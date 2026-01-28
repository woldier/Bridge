# -*- coding:utf-8 -*-
"""
 @FileName   : dcas_transform.py
 @Time       : 9/24/24 4:08 PM
 @Author     : Woldier Wong
 @Description:
 https://github.com/lhoyer/DAFormer/blob/master/mmseg/models/utils/dacs_transforms.py
 Obtained from: https://github.com/vikolss/DACS
 Copyright (c) 2020 vikolss. Licensed under the MIT License
 A copy of the license is available at resources/license_dacs
"""

import kornia
import numpy as np
import torch
import torch.nn as nn
from .img_utils import denorm_, renorm_


def strong_transform(param, data=None, target=None):
    """strong_transform
    DACS 强图像增强
    Args:
        param: 强变化图像增强 参数
        data: 图像 shape [b, 3, h, w]
        target: 标签 shape [b, 1, h, w]

    Returns:
        返回DACS变换后的mix shape [b, 3, h, w]，图像以及 mix 标签 shape [b, h, w]
    """
    assert ((data is not None) or (target is not None))
    data, target = one_mix(mask=param['mix'], data=data, target=target)
    data, target = color_jitter(
        color_jitter=param['color_jitter'],
        s=param['color_jitter_s'],
        p=param['color_jitter_p'],
        mean=param['mean'],
        std=param['std'],
        data=data,
        target=target)
    data, target = gaussian_blur(blur=param['blur'], data=data, target=target)
    return data, target


def color_jitter(color_jitter, mean, std, data=None, target=None, s=.25, p=.2):
    # s is the strength of colorjitter
    if not (data is None):
        if data.shape[-3] == 3:  # 支持 C H W 或者 B C H W
            if color_jitter > p:
                if isinstance(s, dict):
                    seq = nn.Sequential(kornia.augmentation.ColorJitter(**s))
                else:
                    seq = nn.Sequential(
                        kornia.augmentation.ColorJitter(
                            brightness=s, contrast=s, saturation=s, hue=s))
                denorm_(data, mean, std)
                data = seq(data)
                renorm_(data, mean, std)
    return data, target


def gaussian_blur(blur, data=None, target=None):
    if not (data is None):
        if data.shape[-3] == 3:  # 支持 C H W 或者 B C H W
            if blur > 0.5:
                sigma = np.random.uniform(0.15, 1.15)
                kernel_size_y = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[-2]) - 0.5 +
                        np.ceil(0.1 * data.shape[-2]) % 2))
                kernel_size_x = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[-1]) - 0.5 +
                        np.ceil(0.1 * data.shape[-1]) % 2))
                kernel_size = (kernel_size_y, kernel_size_x)
                seq = nn.Sequential(
                    kornia.filters.GaussianBlur2d(
                        kernel_size=kernel_size, sigma=(sigma, sigma)))
                data = seq(data)
    return data, target


def get_class_masks(labels, ignore_index=255, rcs_cla=None):
    """
    在一些数据集中, 其可能存在标签255即边界类别. 这种类别在
    """
    class_masks = []
    for b, label in enumerate(labels):
        classes = torch.unique(label)
        classes = classes[classes != ignore_index]
        nclasses = classes.shape[0]
        class_choice = np.random.choice(nclasses, int((nclasses + nclasses % 2) / 2), replace=False)
        classes = classes[torch.Tensor(class_choice).long()]
        if rcs_cla is not None and rcs_cla[b] not in class_choice: class_choice[0] = rcs_cla[b]  # assure RCS MIX
        class_masks.append(generate_class_mask(label, classes).unsqueeze(0))
    return class_masks


def generate_class_mask(label, classes):
    label, classes = torch.broadcast_tensors(label, classes.unsqueeze(1).unsqueeze(2))
    class_mask = label.eq(classes).sum(0, keepdims=True)
    return class_mask


def one_mix(mask, data=None, target=None):
    if mask is None:
        return data, target
    if not (data is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], data[0])
        data = (stackedMask0 * data[0] +
                (1 - stackedMask0) * data[1]).unsqueeze(0)
    if not (target is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], target[0])
        target = (stackedMask0 * target[0] +
                  (1 - stackedMask0) * target[1]).unsqueeze(0)
    return data, target
