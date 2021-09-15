import numpy as np
from batchgenerators.transforms import Compose, MirrorTransform
from batchgenerators.transforms.crop_and_pad_transforms import CenterCropTransform, RandomCropTransform
from batchgenerators.transforms.spatial_transforms import ResizeTransform, SpatialTransform
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgenerators.transforms.color_transforms import BrightnessTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform

from torchvision import transforms


def get_transforms(mode="train", target_size=128):
    tranform_list = []

    if mode == "train":
        tranform_list = [# CenterCropTransform(crop_size=target_size),
                         ResizeTransform(target_size=(target_size,target_size), order=1),    # resize
                         MirrorTransform(axes=(1,)),
                         SpatialTransform(patch_size=(target_size, target_size), random_crop=False,
                                          patch_center_dist_from_border=target_size // 2,
                                          do_elastic_deform=True, alpha=(0., 1000.), sigma=(40., 60.),
                                          do_rotation=True, p_rot_per_sample=0.5,
                                          angle_x=(-0.1, 0.1), angle_y=(0, 1e-8), angle_z=(0, 1e-8),
                                          scale=(0.5, 1.9), p_scale_per_sample=0.5,
                                          border_mode_data="nearest", border_mode_seg="nearest"),
                         ]

    elif mode == "val":
        tranform_list = [# CenterCropTransform(crop_size=target_size),
                         ResizeTransform(target_size=target_size, order=1),
                         ]

    elif mode == "test":
        tranform_list = [# CenterCropTransform(crop_size=target_size),
                         ResizeTransform(target_size=target_size, order=1),
                         ]

    elif mode == "supcon":
        tranform_list = [
            BrightnessTransform(mu=1, sigma=1, p_per_sample=0.5),
            GammaTransform(p_per_sample=0.5),
            GaussianNoiseTransform(p_per_sample=0.5),
            # SpatialTransform(patch_size=(target_size, target_size)
            #                   do_elastic_deform=True, alpha=(0., 1000.), sigma=(40., 60.),
            #                do_rotation=True, p_rot_per_sample=0.5,
            #                 angle_z=(0, 2 * np.pi),
            #                 scale=(0.7, 1.25), p_scale_per_sample=0.5,
            #                 border_mode_data="nearest", border_mode_seg="nearest"),
        ]

        tranform_list.append(NumpyToTensor())

        return TwoCropTransform(Compose(tranform_list))

    elif mode == "simclr":
        tranform_list = [
            BrightnessTransform(mu=1, sigma=1, p_per_sample=0.5),
            GammaTransform(p_per_sample=0.5),
            GaussianNoiseTransform(p_per_sample=0.5),
            SpatialTransform(patch_size=(target_size, target_size), random_crop=True,
                             do_elastic_deform=True, alpha=(0., 1000.), sigma=(40., 60.),
                             do_rotation=True, p_rot_per_sample=0.5,
                             angle_z=(0, 2 * np.pi),
                             scale=(0.7, 1.25), p_scale_per_sample=0.5,
                             border_mode_data="nearest", border_mode_seg="nearest"),
            NumpyToTensor(),
        ]

        return TwoCropTransform(Compose(tranform_list))

    tranform_list.append(NumpyToTensor())

    return Compose(tranform_list)


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, **x):
        return [self.transform(**x), self.transform(**x)]
