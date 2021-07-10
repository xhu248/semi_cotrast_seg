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

import os
from os.path import exists
import tarfile

from google_drive_downloader import GoogleDriveDownloader as gdd

dataset_id = {
    "Task01_BrainTumour": '1A2IU8Sgea1h3fYLpYtFb2v7NYdMjvEhU',
    "Task02_Heart": '1wEB2I6S6tQBVEPxir8cA5kFB8gTQadYY',
    "Task03_Liver": '1jyVGUGyxKBXV6_9ivuZapQS8eUJXCIpu',
    "Task04_Hippocampus": '1RzPB1_bqzQhlWvU-YGvZzhx2omcDh38C',
    "Task05_Prostate": "1Ff7c21UksxyT4JfETjaarmuKEjdqe1-a",
    "Task07_Pancreas": "1YZQFSonulXuagMIfbJkZeTFJ6qEUuUxL",
    "Task10_Colon": "1m7tMpE9qEcQGQjL_BdMD-Mvgmc44hG1Y"
}

def download_dataset(dest_path, dataset):
    tar_path = os.path.join(dest_path, dataset) + '.tar'
    id = dataset_id[dataset]
    gdd.download_file_from_google_drive(file_id=id,
                                        dest_path=tar_path, overwrite=False,
                                        unzip=False)

    if not exists(os.path.join(dest_path, dataset)):
        print('Extracting data [STARTED]')
        tar = tarfile.open(tar_path)
        tar.extractall(dest_path)
        print('Extracting data [DONE]')
    else:
        print('Data already downloaded. Files are not extracted again.')

    return


if __name__ == "__main__":

    dest_path = "../../data"
    dataset_name = "Task07_Pancreas"

    download_dataset(dest_path, dataset=dataset_name)
