#!/usr/bin/env python

import os
import shutil

from utilities.file_and_folder_operations import subfiles


def rearrange_dir(root_dir):
    image_dir = os.path.join(root_dir, 'images')
    label_dir = os.path.join(root_dir, 'labels')


    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
        print('Created' + image_dir + '...')

    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
        print('Created' + label_dir + '...')

    nii_files = subfiles(root_dir, suffix=".nii.gz", join=False)

    for i in range(0, len(nii_files)):
        src_dir = os.path.join(root_dir, nii_files[i])
        if 'image' in nii_files[i]:
            shutil.move(src_dir, os.path.join(image_dir, nii_files[i]))
        elif 'label' in nii_files[i]:
            shutil.move(src_dir, os.path.join(label_dir, nii_files[i]))

        print('moving' + nii_files[i] + '...')

    files = subfiles(root_dir, suffix=".nii.gz", join=False)
    if files == []:
        print("rearrange directory finished")

