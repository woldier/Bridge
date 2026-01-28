# -*- coding:utf-8 -*-
"""
 @FileName   : img_utils.py
 @Time       : 12/15/24 7:06 PM
 @Author     : Woldier Wong
 @Description: img utils
"""

def denorm(img, mean, std):
    return img.mul(std).add(mean) / 255.0


def denorm_(img, mean, std):
    img.mul_(std).add_(mean).div_(255.0)


def renorm_(img, mean, std):
    img.mul_(255.0).sub_(mean).div_(std)
