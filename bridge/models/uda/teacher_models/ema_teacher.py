# -*- coding:utf-8 -*-
"""
 @FileName   : ema_teacher.py
 @Time       : 1/1/25 8:38 PM
 @Author     : Woldier Wong
 @Description: TODO
"""
from typing import Union, Dict, Optional
import torch
from torch import nn as nn
from mmseg.registry import MODELS

from .base_teacher import BaseTeacher


@MODELS.register_module()
class EMATeacher(BaseTeacher):
    """
    Args:
        model: 接收确切的模型 或者初始化模型的cfg文件
        pseudo_threshold: 可选参数，默认为None。某像素点预测的类别可以作为标签的概率阈值。 此参数用于计算权重。DAFormer和DACS中是0.968
        alpha: EMA更新的α值， 默认为0.99
    """  # noqa: E501

    def __init__(self,
                 model: Union[nn.Module, Dict],
                 pseudo_threshold: Optional[float] = None,
                 alpha: float = 0.99,
                 ):
        self.alpha = alpha
        super().__init__(model, pseudo_threshold)

    def _init_ema_weights(self, model):
        """
        初始化ema的权重
        """
        for param in self.get_model().parameters():
            param.detach_()
        mp = list(model.parameters())
        mcp = list(self.get_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, model, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_model().parameters(),
                                    model.parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def update_weights(self, model, iter):
        # Init/update ema model
        if iter == 0:
            self._init_ema_weights(model)
            # assert _params_equal(self.get_ema_model(), self.get_model())
        if iter > 0:
            self._update_ema(model, iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training

    def get_pseudo_label_and_weight(self, logits):
        ema_softmax = torch.softmax(logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        if self.pseudo_threshold is not None:
            ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
            ps_size = logits.shape[-1] * logits.shape[-2]
            pseudo_weight = torch.sum(ps_large_p).item() / ps_size
            pseudo_weight = pseudo_weight * torch.ones(pseudo_prob.shape, device=logits.device)
        else:
            pseudo_weight = torch.ones(pseudo_prob.shape, device=logits.device)
        return pseudo_label, pseudo_weight

    def __call__(self, img, dropout: bool = False):
        # Generate pseudo-label
        ema_logits = self.logic(img)
        # compute weight
        pseudo_label, pseudo_weight = self.get_pseudo_label_and_weight(ema_logits)
        return pseudo_label, pseudo_weight


@MODELS.register_module()
class EMATeacher2Vehicles(EMATeacher):
    """EMATeacher2Vehicles
    对于 Intelligent Vehicles 通常会忽略掉 top 和 bottom 的一些区域， 因为这些区域是不太重要， 且总是为确定的。
    refer: https://github.com/lhoyer/DAFormer/blob/master/configs/daformer/gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0.py
    
    Args:
        model: config of model in :class:`EMATeacher`
        pseudo_threshold: config of pseudo_threshold in :class:`EMATeacher`
        alpha: config of alpha in :class:`EMATeacher`
        pseudo_weight_ignore_top: 为了保持在缺省情况下与父类 `EMATeache` 保持相同的行为默认为0. 在DAFormer 中 给定的是10
        pseudo_weight_ignore_bottom: 为了保持在缺省情况下与父类 `EMATeache` 保持相同的行为默认为0. 在DAFormer中给定的是120
    """  # noqa: E501

    def __init__(self,
                 model: Union[nn.Module, Dict], pseudo_threshold: Optional[float] = None, alpha: float = 0.99,
                 pseudo_weight_ignore_top: int = 0, pseudo_weight_ignore_bottom: int = 0,
                 ):

        super().__init__(model, pseudo_threshold, alpha)
        self.pseudo_weight_ignore_top = pseudo_weight_ignore_top
        self.pseudo_weight_ignore_bottom = pseudo_weight_ignore_bottom

    def filter_valid_pseudo_region(self, pseudo_weight, valid_pseudo_mask):
        if self.pseudo_weight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            assert valid_pseudo_mask is None
            pseudo_weight[:, :self.pseudo_weight_ignore_top, :] = 0
        if self.pseudo_weight_ignore_bottom > 0:
            assert valid_pseudo_mask is None
            pseudo_weight[:, -self.pseudo_weight_ignore_bottom:, :] = 0
        if valid_pseudo_mask is not None:
            pseudo_weight *= valid_pseudo_mask.squeeze(1)
        return pseudo_weight

    def __call__(self, img, dropout: bool = False, valid_pseudo_mask=None):
        pseudo_label, pseudo_weight = super().__call__(img, dropout)
        pseudo_weight = self.filter_valid_pseudo_region(pseudo_weight, valid_pseudo_mask)
        return pseudo_label, pseudo_weight
