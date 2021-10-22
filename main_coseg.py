from __future__ import print_function

import os
import sys
import argparse
import time
import math
import pickle

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import save_model
from util import get_gpu_memory_map
from networks.unet_con import SupConUnet, LocalConUnet2, LocalConUnet3
from loss_functions.supcon_loss import SupConSegLoss, LocalConLoss, BlockConLoss
from datasets.two_dim.NumpyDataLoader import NumpyDataSet

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')
    parser.add_argument('--pretrained_model_path', type=str, default=None,
                        help='where to find the pretrained model')
    parser.add_argument('--head', type=str, default="cls",
                        help='head mode, cls or mlp')
    parser.add_argument('--stride', type=int, default=4,
                        help='number of stride when doing downsampling')
    parser.add_argument('--mode', type=str, default="stride",
                        help='how to downsample the feature maps, stride or block')


    # optimization
    parser.add_argument('--optimizer', type=str, default="adam",
                        help='optimization method')
    parser.add_argument('--learning_rate', type=float, default=0.00001,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.1,
                        help='momentum')

    # model dataset
    parser.add_argument('--dataset', type=str, default='mmwhs',
                        help='dataset')
    parser.add_argument('--resume', type=str, default=None,
                        help="path to the stored checkpoint")
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')
    parser.add_argument('--split_dir', type=str, default=None, help='path to split pickle file')
    parser.add_argument('--fold', type=int, default=0, help='parameter for splits')
    parser.add_argument('--train_sample', type=float, default=1.0, help='parameter for sampling rate of training set')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = 'data'
    else:
        opt.data_folder = os.path.join(opt.data_folder, opt.dataset, 'preprocessed')

    if opt.split_dir is None:
        opt.split_dir = os.path.join('./data', opt.dataset)
    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_fold_{}_lr_{}_decay_{}_bsz_{}_temp_{}_train_{}_{}'.\
        format(opt.method, opt.dataset, opt.optimizer, opt.fold, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.train_sample, opt.mode)

    if opt.mode =="stride":
        opt.model_name = '{}_stride_{}'.format(opt.model_name, opt.stride)
    elif opt.mode == "block":
        opt.model_name = '{}_block_{}'.format(opt.model_name, opt.block_size)

    if opt.pretrained_model_path is not None:
        opt.model_name = '{}_pretrained'.format(opt.model_name)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_loader(opt):
    # construct data loader
    pkl_dir = opt.split_dir
    with open(os.path.join(pkl_dir, "splits.pkl"), 'rb') as f:
        splits = pickle.load(f)

    if opt.train_sample == 1:
        tr_keys = splits[opt.fold]['train'] + splits[opt.fold]['val'] + splits[opt.fold]['test']
    else:
        tr_keys = splits[opt.fold]['train']
        tr_size = int(len(tr_keys) * opt.train_sample)
        tr_keys = tr_keys[0:tr_size]

    train_loader = NumpyDataSet(opt.data_folder, target_size=64, batch_size=opt.batch_size,
                                keys=tr_keys, do_reshuffle=True, mode="supcon")

    return train_loader


def set_model(opt):
    model = SupConUnet(num_classes=128, mode=opt.head)
    # model = LocalConUnet3(num_classes=128)
    # criterion = SupConSegLoss(temperature=opt.temp)
    if opt.mode == "block":
        criterion = BlockConLoss(temperature=opt.temp, block_size=opt.block_size)
    elif opt.mode == "stride":
        criterion = LocalConLoss(temperature=opt.temp, stride=opt.stride)
    else:
        raise NotImplementedError("The feature downsampling mode is not supported yet!")

    if opt.resume is not None:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            ckpt = torch.load(opt.resume)
            model.load_state_dict(ckpt['model'])
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
            criterion = torch.nn.DataParallel(criterion)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        if opt.pretrained_model_path is not None:
            state_dict = torch.load(opt.pretrained_model_path)
            model.load_state_dict(state_dict, strict=False)
            print("checkpoint state dict:", state_dict.keys())
            print("model state dict:", model.state_dict().keys())
            print("loaded pretrained model:", opt.pretrained_model_path)


    return model, criterion


def set_optimizer(opt, model):
    if opt.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(),
                              lr=opt.learning_rate,
                              momentum=opt.momentum,
                              weight_decay=opt.weight_decay
                              )
    elif opt.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(),
                               lr=opt.learning_rate,
                               weight_decay=opt.weight_decay)
    else:
        raise NotImplementedError("The optimizer is not supported.")
    return optimizer


def train(train_loader, model, criterion, logger, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, data_batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        data1 = data_batch[0]['data'][0].float()
        target1 = data_batch[0]['seg'][0].long()

        data2 = data_batch[1]['data'][0].float()
        target2 = data_batch[1]['seg'][0].long()

        imgs = torch.cat([data1, data2], dim=0)
        labels = torch.cat([target1, target2], dim=0).squeeze(dim=1)  # of shape [2B, 512, 512]

        # images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            imgs = imgs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0] // 2
        img_size = imgs.shape[-1]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        features = model(imgs)
        features = F.normalize(features, p=2, dim=1)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)  # [bsz, n_view, c, img_size, img_size]
        l1, l2 = torch.split(labels, [bsz, bsz], dim=0)
        labels = torch.cat([l1.unsqueeze(1), l2.unsqueeze(1)], dim=1)
        # print(idx, features.device, labels.device)
        loss = criterion(features)


        """
        # compute loss
        inf_time = time.time()
        features = model(imgs)  # of shape [2b, c, 512, 512]
        features = F.normalize(features, p=2, dim=1)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)  # [bsz, n_view, c, img_size, img_size]
        print(features.shape)
        l1, l2 = torch.split(labels, [bsz, bsz], dim=0)
        labels = torch.cat([l1.unsqueeze(1), l2.unsqueeze(1)], dim=1)
        gpu_map = get_gpu_memory_map()
        print("model inference time:", time.time() - inf_time)
        gpu_map = get_gpu_memory_map()
        loss_time = time.time()
        loss = criterion(features, labels)
        print("loss time:", time.time() - loss_time)
        gpu_map = get_gpu_memory_map()
        exit(0)
        """

        if loss.mean() == 0:
            continue
        mask = (loss != 0)
        mask = mask.int().cuda()
        loss = (loss * mask).sum() / mask.sum()

        if torch.isinf(loss):
            print(data_batch[0]['fnames'])
            print(data_batch[0]['slice_idx'])
            print(imgs.max().item(), imgs.min().item())
        losses.update(loss.item(), img_size)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # features = model(imgs)

        # loss_time = time.time() - start
        # print(loss_time)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            num_iteration = idx + 1 + (epoch-1)*len(train_loader)
            logger.add_scalar("train_loss", losses.avg, num_iteration)
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg


def main():
    opt = parse_option()

    # build data loader
    train_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    logger = SummaryWriter(opt.tb_folder)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, logger, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.add_scalar('loss', loss, epoch)
        logger.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    # save_file = os.path.join(
    #   opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    print(torch.cuda.device_count())
    main()
