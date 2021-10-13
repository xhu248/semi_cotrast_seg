from collections import defaultdict
from batchgenerators.augmentations.utils import resize_image_by_padding

from medpy.io import load
import os
import numpy as np
import shutil
import torch
import torch.nn.functional as F


def preprocess_data(root_dir):
    image_dir = os.path.join(root_dir, 'imgs')
    label_dir = os.path.join(root_dir, 'labels')
    output_dir = os.path.join(root_dir, 'orig')

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
            # modify the label for MMWHS dataset
            label[label == 500] = 1
            label[label == 600] = 2
            label[label == 420] = 3
            label[label == 550] = 4
            label[label == 205] = 5
            label[label == 820] = 6
            label[label == 850] = 7
            """

            print(image.shape, label.shape)

            result = np.stack((image, label)).transpose((3, 0, 1, 2))
            print(result.shape)

            np.save(os.path.join(output_dir, f.split('.')[0] + '.npy'), result)
            print(f)

    print(total)


def reshape_2d_data(input_dir, output_dir, target_size=(64, 64)):
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

            new_image = F.interpolate(image_tensor[None], size=target_size, mode="bilinear")
            new_image = new_image.squeeze().cpu().numpy()

            new_label = F.interpolate(label_tensor[None], size=target_size, mode="bilinear")
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
    root_dir = "../../data/Hippocampus"
    input_dir = "../../data/Hippocampus/orig"
    target_dir = "../../data/Hippocampus/preprocessed"

    preprocess_data(root_dir)

    reshape_2d_data(input_dir, target_dir)




