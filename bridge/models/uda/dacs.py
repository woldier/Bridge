# -*- coding:utf-8 -*-
"""
 @FileName   : TestUDA.py
 @Time       : 12/28/24 2:13 PM
 @Author     : Woldier Wong
 @Description: "DACS: Domain Adaptation via Cross-domain Mixed Sampling"
 refer https://github.com/vikolss/DACS
"""
from typing import Union, Dict
import torch, random
from mmengine.optim import OptimWrapper

from .uda_decorator import OPTStr
from .uda_with_teacher import UDAWithTeacher
from mmseg.registry import MODELS
from mmseg.utils import ConfigType, OptConfigType
from mmengine.structures import PixelData
from mmseg.utils import add_prefix


@MODELS.register_module()
class DACS(UDAWithTeacher):
    """DACS
    
    Args:
        blur: 是否进行 gaussian_blur. 默认为Fasle，即不进行。
        color_jitter_strength:  color_jitter 的强度。 默认为 0.25
        color_jitter_probability: color_jitter的概率 默认为0.2
        **cfg:
    """  # noqa: E501

    def __init__(self,
                 blur: bool = False,
                 color_jitter_strength: float = .25,
                 color_jitter_probability: float = .2,
                 **cfg):
        super().__init__(**cfg)
        self.blur = blur
        self.color_jitter_s = color_jitter_strength
        self.color_jitter_p = color_jitter_probability

    def _prepare_strong_transform_param(self, means, stds):
        """
        img_metas: 图片的元数据
        dev: 设备id
        """

        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }
        return strong_parameters

    @staticmethod
    def _img_mix(img, gt_semantic_seg,
                 target_img, pseudo_label, pseudo_weight,
                 batch_size, strong_parameters, rcs_cla=None):
        from ..utils.dacs_transform import get_class_masks, strong_transform
        gt_pixel_weight = torch.ones_like(pseudo_weight)  # 对于source, 其损失权重总是为1
        mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
        mixed_seg_weight = pseudo_weight.clone()
        mix_masks = get_class_masks(gt_semantic_seg, rcs_cla=rcs_cla)
        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks[i]
            mixed_img[i], mixed_lbl[i] = strong_transform(
                strong_parameters,
                data=torch.stack((img[i], target_img[i])),
                target=torch.stack((gt_semantic_seg[i][0], pseudo_label[i][0])))
            _, mixed_seg_weight[i] = strong_transform(
                strong_parameters,
                target=torch.stack((gt_pixel_weight[i], mixed_seg_weight[i])))
        mixed_img = torch.cat(mixed_img)
        mixed_lbl = torch.cat(mixed_lbl)
        return mix_masks, mixed_img, mixed_lbl, mixed_seg_weight

    def _inner_train_step(self, data: Union[dict, tuple, list], optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        tgt_key_prefix = data['tgt_key_prefix']
        img, data_samples, target_img, target_data_samples = \
            data['inputs'], data['data_samples'], data[f"{tgt_key_prefix}inputs"], data[f"{tgt_key_prefix}data_samples"]
        loss_var_total = {}
        # ================================ Source Train =========================================
        with optim_wrapper.optim_context(self.model):
            losses = self.get_model()._run_forward(dict(inputs=img, data_samples=data_samples), mode='loss')
        parsed_losses, log_vars = self.parse_losses(losses)
        loss_total = parsed_losses
        loss_var_total.update(add_prefix(log_vars, 'src'))
        # ================================ Target Train =========================================
        ## 1.Pseudo Label Generate
        pseudo_label, weight = self.get_ema_model().__call__(target_img)
        # 组装标签
        for p_label, tgt_sample in zip(pseudo_label.unsqueeze(1), target_data_samples):
            tgt_sample.gt_sem_seg = PixelData(data=p_label)
        batch_size = pseudo_label.shape[0]

        means = [torch.as_tensor(self.data_preprocessor.mean).cuda() for _ in range(batch_size)]
        stds = [torch.as_tensor(self.data_preprocessor.std).cuda() for _ in range(batch_size)]

        strong_parameters = self._prepare_strong_transform_param(means, stds)
        gt_sem_seg = self._stack_batch_gt(data_samples)  # 凭借source 标签
        mix_masks, mixed_img, mixed_lbl, mixed_seg_weight = \
            self._img_mix(img, gt_sem_seg, target_img, pseudo_label.unsqueeze(1),
                          weight, batch_size, strong_parameters)
        # 组装mix标签
        mix_data_samples = []
        for m_label, tgt_sample, m_weight in zip(mixed_lbl, target_data_samples, mixed_seg_weight):
            mix_data_samples.append(tgt_sample.new(gt_sem_seg=PixelData(data=m_label), seg_weight=m_weight))
        with optim_wrapper.optim_context(self.model):
            losses_tgt = self.get_model()._run_forward(
                dict(inputs=mixed_img, data_samples=mix_data_samples),
                mode='loss')
        parsed_losses, log_vars = self.parse_losses(losses_tgt)
        loss_total += parsed_losses
        loss_var_total.update(add_prefix(log_vars, 'tgt'))
        optim_wrapper.update_params(loss_total)
        return loss_var_total
