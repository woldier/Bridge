# -*- coding:utf-8 -*-
"""
 @FileName   : base_teacher.py
 @Time       : 1/1/25 8:04 PM
 @Author     : Woldier Wong
 @Description: base teacher
"""
import torch, torch.nn as nn
from abc import ABCMeta, abstractmethod
from typing import Union, Dict, Optional
from mmseg.registry import MODELS
from mmseg.models import EncoderDecoder
from mmseg.models.utils import resize
from torch.nn.modules.dropout import _DropoutNd


class BaseTeacher(nn.Module, metaclass=ABCMeta):
    """Teacher
    Teacher model 定义了对model的包装，并定义了通过model来进行推理得到预测值、标签和权重的方法

    Args:
        model: 接收确切的模型 或者初始化模型的cfg文件
    """

    def __init__(self,
                 model: Union[nn.Module, Dict],
                 pseudo_threshold: Optional[float] = None):
        super().__init__()
        self.pseudo_threshold = pseudo_threshold
        self.model = self._init_model(model)
        for p in self.model.parameters():
            p.requires_grad = False
        # eval mod
        self.model.eval()

    @staticmethod
    def _init_model(model: Union[nn.Module, Dict]) -> nn.Module:
        if isinstance(model, nn.Module):
            return model
        elif isinstance(model, dict):
            model = MODELS.build(model)
            return model  # type: ignore
        else:
            raise TypeError('model should be a nn.Module object or dict, '
                            f'but got {model}')

    def get_model(self) -> EncoderDecoder:
        return self.model

    def _model_dropout(self, mode: bool = False):
        timm_flag = True
        try:
            from timm.models.layers import DropPath
        except ImportError:
            timm_flag = False
        for m in self.model.modules():
            if isinstance(m, _DropoutNd):
                m.training = mode
            if timm_flag and isinstance(m, DropPath):
                m.training = mode

    def logic(self, img: torch.Tensor, dropout: bool = False, return_feature: bool = False):
        self._model_dropout(dropout)
        feat = self.get_model().extract_feat(img)
        # logic = self.get_model().decode_head.forward(feat)
        # align_corners = self.get_model().align_corners
        # logic = resize(input=logic, size=img.shape[2:], mode='bilinear', align_corners=align_corners)
        logic = self.get_model().decode_head.predict(feat, [ dict(img_shape=_img.shape[-2:]) for _img in img], {})
        align_corners = self.get_model().align_corners
        logic = resize(input=logic, size=img.shape[2:], mode='bilinear', align_corners=align_corners)
        if return_feature:
            return logic, feat
        return logic

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass
