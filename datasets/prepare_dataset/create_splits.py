#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pickle
from utilities.file_and_folder_operations import subfiles

import os
import numpy as np


def create_splits(output_dir, image_dir):
    npy_files = subfiles(image_dir, suffix=".npy", join=False)

    trainset_size = len(npy_files)*50//100
    valset_size = len(npy_files)*25//100
    testset_size = len(npy_files)*25//100

    splits = []
    for split in range(0, 5):
        image_list = npy_files.copy()
        trainset = []
        valset = []
        testset = []
        for i in range(0, trainset_size):
            patient = np.random.choice(image_list)
            image_list.remove(patient)
            trainset.append(patient)
        for i in range(0, valset_size):
            patient = np.random.choice(image_list)
            image_list.remove(patient)
            valset.append(patient)
        for i in range(0, testset_size):
            patient = np.random.choice(image_list)
            image_list.remove(patient)
            testset.append(patient)
        split_dict = dict()
        split_dict['train'] = trainset
        split_dict['val'] = valset
        split_dict['test'] = testset

        splits.append(split_dict)

    with open(os.path.join(output_dir, 'splits.pkl'), 'wb') as f:
        pickle.dump(splits, f)


# some dataset may include an independent test set
def create_splits_1(output_dir, image_dir, test_dir):
    npy_files = subfiles(image_dir, suffix=".npy", join=False)
    test_files = subfiles(test_dir, suffix=".npy", join=False)

    trainset_size = len(npy_files) * 3 // 4
    valset_size = len(npy_files) - trainset_size

    splits = []
    for split in range(0, 5):
        image_list = npy_files.copy()
        trainset = []
        valset = []
        for i in range(0, trainset_size):
            patient = np.random.choice(image_list)
            image_list.remove(patient)
            trainset.append(patient)
        for i in range(0, valset_size):
            patient = np.random.choice(image_list)
            image_list.remove(patient)
            valset.append(patient)
        split_dict = dict()
        split_dict['train'] = trainset
        split_dict['val'] = valset
        split_dict['test'] = test_files

        splits.append(split_dict)

    with open(os.path.join(output_dir, 'splits.pkl'), 'wb') as f:
        pickle.dump(splits, f)


# divide the processed .npy data into three parts, and do it four times