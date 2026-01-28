# -*- coding:utf-8 -*-
"""
 @FileName   : uda_with_teacher.py
 @Time       : 1/1/25 9:36 PM
 @Author     : Woldier Wong
 @Description: UDA decorator with teacher model
"""
from copy import deepcopy
from typing import Union, Dict

import torch
from mmengine.optim import OptimWrapper

from .teacher_models import EMATeacher
from .uda_decorator import UDADecorator, OPTStr
from mmseg.utils import ConfigType, OptConfigType
from mmseg.registry import MODELS


class UDAWithTeacher(UDADecorator):
    """UDAWithTeacher
    包含Teacher model 的 UDA Decorator
    在 train_step 中每次迭代都会更新EMA teacher。
    因为更新EMA teacher 是含有教师model 的UDA方法的共性， 因此我们将更新操作进行了抽象。

    然后扩展了 _inner_train_step 方法， 在_inner_train_step中完成UDA的训练逻辑

    Args:
        teacher: teacher model 的配置文件
    """  # noqa: E501

    def __init__(self, model: ConfigType, teacher: ConfigType,
                 data_preprocessor: OptConfigType = None, work_dir: OPTStr = None, **cfg):
        teacher.model = deepcopy(model)
        super().__init__(model, data_preprocessor, work_dir, **cfg)
        self.ema_model = self._init_ema_teacher(teacher)
        self.register_buffer('local_iter', torch.tensor(0, dtype=torch.long))

    @property
    def iter(self):
        return self.local_iter

    @staticmethod
    def _init_ema_teacher(teacher: ConfigType):
        return MODELS.build(teacher)

    def get_ema_model(self) -> EMATeacher:
        return self.ema_model

    def update_weights(self, iter: int):
        self.get_ema_model().update_weights(self.get_model(), iter)

    def train_step(self, data: Union[dict, tuple, list], optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        """
        抽象所有teacher model 的UDA 方法都需要更新teacher 权重的逻辑
        具体的训练逻辑 由_inner_train_step完成
        Args:
            data:
            optim_wrapper:

        Returns:

        """
        self.update_weights(self.iter)
        data = self.data_preprocessor(data, True)  # 调用 UDA 的 data processor
        out = self._inner_train_step(data, optim_wrapper)
        self.local_iter += 1 # local iter acc
        return out

    def _inner_train_step(self, data: Union[dict, tuple, list], optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        """
        UDA with teacher 的训练逻辑在此进行实现
        Args:
            data: 数据
            optim_wrapper: 优化器的包装

        Returns:
            loss的记录值

        Examples:

        >>> with optim_wrapper.optim_context(self.model):
        >>>     data = self.model.data_preprocessor(data, True)
        >>>     losses = self.model._run_forward(data, mode='loss')  # type: ignore
        >>> parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        >>> optim_wrapper.update_params(parsed_losses)
        >>> return log_vars
        """
        raise NotImplementedError
