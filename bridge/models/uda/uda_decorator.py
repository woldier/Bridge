# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# refer https://github.com/lhoyer/DAFormer/blob/master/mmseg/models/uda/uda_decorator.py
# refer https://github.com/woldier/SiamSeg/blob/master/mmseg/models/uda/uda_decorator.py
# many changes
# ---------------------------------------------------------------
from typing import Tuple, List, Union, Dict

import torch
from copy import deepcopy
from mmengine.model.wrappers import MMDistributedDataParallel
from mmengine.optim import OptimWrapper
from torch import Tensor
from mmengine.structures import PixelData

from mmseg.models import BaseSegmentor, build_segmentor, EncoderDecoder
from mmseg.utils import OptSampleList, SampleList
from mmseg.utils import OptConfigType, ConfigType
from typing import Optional

OPTStr = Optional[str]


def get_module(module):
    """Get `nn.ModuleDict` to fit the `MMDistributedDataParallel` interface.
    Args:
        module (MMDistributedDataParallel | nn.ModuleDict): The input
            module that needs processing.

    Returns:
        nn.ModuleDict: The ModuleDict of multiple networks.
    """
    if isinstance(module, MMDistributedDataParallel):
        return module.module

    return module


class UDADecorator(BaseSegmentor):
    """UDADecorator


    默认会使用cfg.model 的配置作为 model 的配置文件
    并且本UDADecorator 默认会使用cfg.model 的train_cfg 和test_cfg 文件

    UDA Decorator提供了对 Segmentor 的封装， 支持UDA 训练。
    需要注意的是， UDA Decorator 需要重写 train_step方法，完成对UDA训练逻辑的定义。

    除此之外，考虑到UDA有对抗学习的方法，
    为了支持分布式训练，我们使用 MMSeparateDistributedDataParallel 封装UDA Decorator。
    Parameters:
        model (dict): model 的配置
        work_dir (str|None): 当前工作目录用于需要保存临时文件的需求，如果没设置在init UDA model 时会设置与  cfg.work_dir 相同
        cmap (str): 方法中最常见的保存临时文件的需求是可视化训练过程的图片，而cmap定义了mask 的color.
                    支持的cmap详见mmseg/models/utils/visualization.py
        data_preprocessor (dict, optional): The pre-process config of :class:`.BaseDataPreprocessor`.
    """  # noqa: E501

    def __init__(self, model: ConfigType, data_preprocessor: OptConfigType = None,
                 work_dir: OPTStr = None,  # cmap: OPTStr = 'isprs',
                 **cfg):
        super(BaseSegmentor, self).__init__(data_preprocessor=data_preprocessor)

        self.model = build_segmentor(deepcopy(model))
        self.train_cfg = model['train_cfg']
        self.test_cfg = model['test_cfg']
        self.num_classes = model['decode_head']['num_classes']
        self.work_dir = work_dir
        # self.cmap = cmap

    def get_model(self) -> EncoderDecoder:
        return get_module(self.model)

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        return self.get_model().extract_feat(inputs)

    def encode_decode(self, inputs: Tensor, batch_data_samples: List[dict]) -> Tensor:
        return self.get_model().encode_decode(inputs, batch_data_samples)

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        return self.get_model().loss(inputs, data_samples)

    def predict(self, inputs: Tensor, data_samples: OptSampleList = None) -> SampleList:
        return self.get_model().predict(inputs, data_samples)

    def _forward(self, inputs: Tensor, data_samples: OptSampleList = None) -> Tensor:
        return self.get_model()._forward(inputs, data_samples)

    def train_step(self, data: Union[dict, tuple, list], optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        """

        Args:
            data:
            optim_wrapper:

        Returns:

        Examples:

        >>> with optim_wrapper.optim_context(self):
        >>>     data = self.data_preprocessor(data, True)
        >>>     losses = self._run_forward(data, mode='loss')  # type: ignore
        >>> parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        >>> optim_wrapper.update_params(parsed_losses)
        >>> return log_vars
        """
        # 实现自定义的UDA 训练逻辑
        # return super().train_step(data, optim_wrapper)
        raise NotImplementedError

    @staticmethod
    def _stack_batch_gt(batch_data_samples: SampleList) -> Tensor:
        """
        从batch_data_samples中组装gt_semantic_segs并且返回tensor
        Args:
            batch_data_samples:

        Returns:

        """
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_semantic_segs, dim=0)

    @staticmethod
    def _unwrap_data(data: dict):
        """
        从data dict 中解包 出 source 和 target 数据
        我们约定 data 中必须包含 key `inputs` 作为 source 的img,
        `data_samples` 作为source 的 data_samples.

        除此之外, 还必须有 key `tgt_key_prefix`, 指明target data 的前缀.
        Args:
            data:

        Returns:
            source 图像, source data samples, target 图像, target data samples
        """
        tgt_key_prefix = data['tgt_key_prefix']
        img, data_samples, target_img, target_data_samples = \
            data['inputs'], data['data_samples'], data[f"{tgt_key_prefix}inputs"], data[f"{tgt_key_prefix}data_samples"]
        return img, data_samples, target_img, target_data_samples

    @staticmethod
    def _assemble_label(batch_label: Tensor, data_samples: SampleList,
                        empty_check=False, new_sample=False) -> SampleList:
        """

        对于target 数据, 没有GT标签是可以获得的, 因此在data_samples中没有gt_sem_seg

        当我们生成伪标签后, 可以通过本方法将伪标签 设置到 data_samples中

        Args:
            batch_label: 需要设置的标签 [b 1 h w]
            data_samples: data_samples
            empty_check: 是否检查 SampleList 中的每个sample 有gt_sem_seg 成员
            new_sample: 是否重新生成一份data_sample. 从而保证对传入的data_samples不产生额外影响.

        Returns:

        """
        if len(batch_label.shape) == 3:
            batch_label = batch_label.unsqueeze(1)
        assert len(batch_label.shape) == 4 and batch_label.shape[1] == 1
        res_data_samples = []
        for label, sample in zip(batch_label, data_samples):
            assert not empty_check or sample.get('gt_sem_seg', None) is None, \
                f"when {empty_check=}, sample.gt_sem_seg must `None`"
            if new_sample:
                sample = sample.new(gt_sem_seg=PixelData(data=label))  # type: ignore
            else:
                sample.gt_sem_seg = PixelData(data=label)
            res_data_samples.append(sample)
        return res_data_samples

    def _get_mean_std(self, batch_size):
        means = [torch.as_tensor(self.data_preprocessor.mean).cuda() for _ in range(batch_size)]
        stds = [torch.as_tensor(self.data_preprocessor.std).cuda() for _ in range(batch_size)]
        return torch.stack(means, dim=0), torch.stack(stds, dim=0)
