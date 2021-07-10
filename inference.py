import os
import pickle
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from configs.Config import get_config
from configs.Config_mmwhs import get_config
from datasets.two_dim.NumpyDataLoader import NumpyDataSet

from networks.unet_con import SupConUnetInfer
from loss_functions.supcon_loss import SupConSegLoss, LocalConLoss, BlockConLoss
from loss_functions.metrics import SegmentationMetric
from util import AverageMeter


class InferenceExperiment(object):
    def __init__(self, config):
        self.config = config
        pkl_dir = self.config.split_dir
        with open(os.path.join(pkl_dir, "splits.pkl"), 'rb') as f:
            splits = pickle.load(f)

        self.train_keys = splits[self.config.fold]['train'][0:2]
        self.val_keys = splits[self.config.fold]['val'][0:2]

        self.test_data_loader = NumpyDataSet(self.config.data_dir, target_size=self.config.img_size,
                                             batch_size=2, keys=self.train_keys, do_reshuffle=False, mode="test")
        self.model = SupConUnetInfer(num_classes=self.config.num_classes)
        self.criterion = SupConSegLoss(temperature=0.7)
        self.criterion1 = LocalConLoss(temperature=0.7)
        self.criterion2 = BlockConLoss(temperature=0.7)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.model = nn.DataParallel(self.model)

        self.device = torch.device(self.config.device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.criterion.to(self.device)
        self.criterion1.to(self.device)
        self.criterion2.to(self.device)

        # self.load_checkpoint()

        self.save_folder = os.path.join(self.config.base_dir, "infer_" + self.config.name + str(datetime.now())[0:16])
        if not os.path.exists(self.save_folder):
            os.mkdir(self.save_folder)

    def load_checkpoint(self):
        if self.config.saved_model_path is None:
            print('checkpoint_dir is empty, please provide directory to load checkpoint.')
            exit(0)
        else:
            state_dict = torch.load(self.config.saved_model_path)['model']
            self.model.load_state_dict(state_dict, strict=False)
            # self.model.load_state_dict(state_dict)

    def binfer(self):
        self.model.eval()
        co_losses = AverageMeter()
        local_co_losses = AverageMeter()
        block_co_losses = AverageMeter()
        metric_val = SegmentationMetric(self.config.num_classes)
        metric_val.reset()
        bsz = 2

        with torch.no_grad():
            for (i, data_batch) in enumerate(self.test_data_loader):
                """
                data = data_batch['data'][0].float().to(self.device)
                labels = data_batch['seg'][0].long().to(self.device)
                fnames = data_batch['fnames']
                slice_idx = data_batch['slice_idxs']
                """
                data1 = data_batch[0]['data'][0].float()
                target1 = data_batch[0]['seg'][0].long()

                data2 = data_batch[1]['data'][0].float()
                target2 = data_batch[1]['seg'][0].long()

                data = torch.cat([data1, data2], dim=0)
                labels = torch.cat([target1, target2], dim=0).squeeze(dim=1)  # of shape [2B, 512, 512]

                features, output = self.model(data)
                output_softmax = F.softmax(output, dim=1)
                pred = torch.argmax(output_softmax, dim=1)
                metric_val.update(labels, output_softmax)
                # self.save_data(pred, fnames, slice_idx, 'seg')

                features = F.normalize(features, p=2, dim=1)
                # print(features.shape, labels.shape)
                f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)  # [bsz, n_view, c, img_size, img_size]
                l1, l2 = torch.split(labels, [bsz, bsz], dim=0)
                labels = torch.cat([l1.unsqueeze(1), l2.unsqueeze(1)], dim=1)
                labels = labels.cuda()
                # print(features.device, labels.device)
                co_loss = self.criterion(features, labels)
                local_co_loss = self.criterion1(features, labels)
                block_co_loss = self.criterion2(features, labels)
                if co_loss == 0:
                    continue
                co_losses.update(co_loss, bsz)
                if local_co_loss == 0:
                    continue
                local_co_losses.update(local_co_loss, bsz)
                if block_co_loss == 0:
                    continue
                block_co_losses.update(block_co_loss, bsz)
                # self.save_data(features, fnames, slice_idx, 'features')

                if i % 10 == 0:
                    _, _, Dice = metric_val.get()
                    print("Index:%d, mean Dice:%.4f" % (i, Dice))
                    print("Index:%d, mean contrastive loss:%.4f" % (i, co_losses.avg))

        print("=====Inference Finished=====")
        _, _, Dice = metric_val.get()
        print("mean Dice:", Dice)
        print("mean contrastive loss:", co_losses.avg.item())
        print("mean local contrastive loss:", local_co_losses.avg.item())
        print("mean block contrastive loss:", block_co_losses.avg.item())

    def inference(self):
        self.model.eval()
        co_losses = AverageMeter()
        metric_val = SegmentationMetric(self.config.num_classes)
        metric_val.reset()
        bsz = 4

        with torch.no_grad():
            for k in range(2):
                key = self.val_keys[k:k+1]
                data_loader = NumpyDataSet(self.config.data_dir, target_size=self.config.img_size,
                                             batch_size=bsz, keys=key, do_reshuffle=False, mode="test")
                feature_map = []
                prediction = []
                for (i, data_batch) in enumerate(data_loader):
                    data = data_batch['data'][0].float().to(self.device)
                    labels = data_batch['seg'][0].long().to(self.device)
                    slice_idx = data_batch['slice_idxs']

                    features, output = self.model(data)
                    # print(output.shape, labels.shape)
                    output_softmax = F.softmax(output, dim=1)
                    pred = torch.argmax(output_softmax, dim=1)
                    metric_val.update(labels.squeeze(), output_softmax)
                    # self.save_data(pred, fnames, slice_idx, 'seg')

                    features = F.normalize(features, p=2, dim=1)
                    for j in range(features.shape[0]):
                        # feature_map.append(features[j].cpu().numpy())
                        prediction.append(pred[j].cpu().numpy())
                    # print(features.shape, labels.shape)

                    """
                    if i == 30:
                        print(slice_idx)
                        self.save_data(features.cpu().numpy(), key[0], 'features')
                        self.save_data(labels.cpu().numpy(), key[0], "labels")
                    """

                    if i % 10 == 0:
                        _, _, Dice = metric_val.get()
                        print("Index:%d, mean Dice:%.4f" % (i, Dice))

                # feature_map = np.stack(feature_map)
                prediction = np.stack(prediction)
                # self.save_data(feature_map, key, 'features')
                self.save_data(prediction, key[0], 'prediction')

        print("=====Inference Finished=====")
        _, _, Dice = metric_val.get()
        print("mean Dice:", Dice)

    def save_data(self, data, key, mode):

        if not os.path.exists(os.path.join(self.save_folder, mode)):
            os.mkdir(os.path.join(self.save_folder, mode))

        save_path = os.path.join(self.save_folder, mode + '_' + key)
        np.save(save_path, data)

        """
        for k in range(bsz):
            slice = slice_idx[k][0].numpy()
            file_name = fnames[k][0].split("preprocessed/")[1]
            save_path = os.path.join(self.save_folder, mode, str(slice) + '_' + file_name)
            np.save(save_path, data[k])
        """


if __name__ == "__main__":
    c = get_config()
    c.saved_model_path = os.path.abspath("output_experiment") + "/20210227-065712_Unet_mmwhs/" \
                        + "checkpoint/" + "checkpoint_last.pth.tar"
    # c.saved_model_path = os.path.abspath('save') + '/SupCon/mmwhs_models/' \
    #                     + 'SupCon_mmwhs_adam_fold_0_lr_0.0001_decay_0.0001_bsz_4_temp_0.1_train_0.4_mlp_block_pretrained/' \
    #                     + 'ckpt.pth'
    c.fold = 0
    print(c)
    exp = InferenceExperiment(config=c)
    exp.load_checkpoint()
    exp.inference()

