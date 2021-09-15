from collections import defaultdict
from batchgenerators.augmentations.utils import resize_image_by_padding

from medpy.io import load
import os
import numpy as np
import shutil
import torch
import torch.nn.functional as F


def preprocess_data(root_dir, target_size=None):
    image_dir = os.path.join(root_dir, 'imagesTr')
    label_dir = os.path.join(root_dir, 'labelsTr')
    output_dir = os.path.join(root_dir, 'preprocessed')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('Created' + output_dir + '...')

    class_stats = defaultdict(int)
    total = 0
    nii_files = subfiles(image_dir, suffix=".nii.gz", join=False)

    for f in nii_files:
        if f.startswith("."):
            os.remove(os.path.join(image_dir, f))
            continue
        file_dir = os.path.join(output_dir, f.split('.')[0]+'.npy')
        if not os.path.exists(file_dir):
            image, _ = load(os.path.join(image_dir, f))
            label, _ = load(os.path.join(label_dir, f.replace('image', 'label')))


            # normalize images
            image = (image - image.min()) / (image.max() - image.min())

            print(label.max())
            print(label.min())
            total += image.shape[2]

            image = image[:, :, 0].transpose((0, 2, 1))

            """
            # modify the label
            label[label == 500] = 1
            label[label == 600] = 2
            label[label == 420] = 3
            label[label == 550] = 4
            label[label == 205] = 5
            label[label == 820] = 6
            label[label == 850] = 7
            """

            if target_size is not None:
                image = resize_image_by_padding(image, (target_size, target_size, image.shape[2]), "constant",
                                     kwargs={'constant_values': image.min()})
                label = resize_image_by_padding(label, (target_size, target_size, image.shape[2]), "constant",
                                     kwargs={'constant_values': label.min()})
            print(image.shape, label.shape)

            # result = np.stack((image, label)).transpose((3, 0, 1, 2))
            # result = reshape_array(result)
            # print(result.shape)

            # np.save(os.path.join(output_dir, f.split('.')[0] + '.npy'), result)
            # print(f)

    print(total)


def padding_imgs(orig_img, append_value=-1024, new_shape=(512, 512, 512)):
    reshaped_image = np.zeros(new_shape)
    reshaped_image[...] = append_value
    x_offset = 0
    y_offset = 0  # (new_shape[1] - orig_img.shape[1]) // 2
    z_offset = 0  # (new_shape[2] - orig_img.shape[2]) // 2

    reshaped_image[x_offset:orig_img.shape[0]+x_offset, y_offset:orig_img.shape[1]+y_offset, z_offset:orig_img.shape[2]+z_offset] = orig_img
    # insert temp_img.min() as background value

    return reshaped_image


def reshape_2d_data(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('Created' + output_dir + '...')

    files_list = os.listdir(input_dir)

    for f in files_list:
        target_dir = os.path.join(output_dir, f)
        if not os.path.exists(target_dir):
            data = np.load(os.path.join(input_dir, f))

            image = data[:, 0]
            label = data[:, 1]

            image_tensor = torch.from_numpy(image)
            label_tensor = torch.from_numpy(label)

            new_image = F.interpolate(image_tensor[None], size=(160, 160), mode="bilinear")
            new_image = new_image.squeeze().cpu().numpy()

            new_label = F.interpolate(label_tensor[None], size=(160, 160), mode="bilinear")
            new_label = new_label.squeeze().cpu().numpy()

            new_data = np.concatenate((new_image[:, None], new_label[:, None]), axis=1)

            print(new_data.shape)
            np.save(target_dir, new_data)


def reshape_three_dim_data(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('Created' + output_dir + '...')

    files_list = os.listdir(input_dir)

    for f in files_list:
        target_dir = os.path.join(output_dir, f)
        if not os.path.exists(target_dir):
            data = np.load(os.path.join(input_dir, f))

            image = data[:, 0]
            label = data[:, 1]

            image_tensor = torch.from_numpy(image)
            label_tensor = torch.from_numpy(label)

            new_image = F.interpolate(image_tensor[None, None], size=(160, 160), mode="bilinear")
            new_image = new_image.squeeze().cpu().numpy()

            new_label = F.interpolate(label_tensor[None, None], size=(160, 160), mode="bilinear")
            new_label = new_label.squeeze().cpu().numpy()

            new_data = np.concatenate((new_image[None], new_label[None]))

            print(new_data.shape)
            np.save(target_dir, new_data)


def subfiles(folder, join=True, prefix=None, suffix=None, sort=True):
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y   # lambda is another simplified way of defining a function
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isfile(os.path.join(folder, i))
            and (prefix is None or i.startswith(prefix))
            and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res

if __name__ == "__main__":
    root_dir = "../../data/Prostate"
    input_dir = "../../data/mmwhs/orig"
    target_dir = "../../data/mmwhs/preprocessed"
    target_dir_1 = os.path.join(root_dir, "cardiac/labels")
    src_dir = os.path.join(root_dir, 'ct_train1')
    src_dir_1 = os.path.join(root_dir, "ct_train2")
    k = 5;
    j = 10 -k;

    # preprocess_data(root_dir)

    reshape_2d_data(input_dir, target_dir)


"""
    files = os.listdir(src_dir)
    for i in range(k):
        f = np.random.choice(files)
        files.remove(f)
        print(f)
        shutil.copyfile(os.path.join(src_dir, f), os.path.join(target_dir, f))

    files = os.listdir(src_dir_1)
    for i in range(j):
        f = np.random.choice(files)
        files.remove(f)
        print(f)
        shutil.copyfile(os.path.join(src_dir_1, f), os.path.join(target_dir, f))

"""



