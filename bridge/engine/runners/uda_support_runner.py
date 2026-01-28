# -*- coding:utf-8 -*-
"""
 @FileName   : uda_support_runner.py
 @Time       : 12/31/24 3:02 PM
 @Author     : Woldier Wong
 @Description: support uda running
"""
from typing import Union, Dict, Optional, List
import copy
from mmengine.evaluator import Evaluator
from mmengine.hooks import Hook
from mmengine.optim import OptimWrapper, _ParamScheduler

from mmengine.runner import Runner
from mmengine.runner.runner import ConfigType
from mmengine.visualization import Visualizer
from torch import nn as nn
from torch.utils.data import DataLoader


from mmseg.registry import RUNNERS, MODELS


@RUNNERS.register_module()
class UDASupportedRunner(Runner):
    """UDASupportedRunner
    用于支持UDA的Runner。
    在设计UDA MODEL时， 为了保持EncoderDecoder的封装性，并遵循开闭原则，因此UDA model的实现使用的是装饰器模式。

    在UDA中内嵌EncoderDecoder，支持与其相同的行为（除了训练过程外）。

    同理为了不破坏Config 文件的封装性， 我们额外的使用了
    uda config 来配置uda。 装载模型时，再封装成UDA model 需求的config 文件格式。

    例如：
    model=dict(type='model_name',xxx=xxx)
    uda=dict(type='uda_name',xxx=xx)

    在装载模型时 会自动的组装成如下

    >>> uda=dict(
    >>>     type='uda_name',xxx=xx  # UDA 原有的
    >>>     # 初始化 encoder decoder 的配置文件
    >>>     model=dict(type='model_name',xxx=xxx)
    >>>     # 如果uda没有设置自己的data_preprocessor那么使用的是 UDASegDataPreProcessor
    >>>     data_preprocessor = copy.deepcopy(model["data_preprocessor"])
    >>>     uda['data_preprocessor']['type'] = 'UDASegDataPreProcessor'
    >>>     # work_dir 用于保存训练中间的一些数据到指定路径，如可视化等。 默认与self.work_dir相同，也支持自定义
    >>>     work_dir=self.work_dir)

    此配置文件用于初始化UDA model，此实现方式不会破坏原有配置文件的封装。
    在测试阶段， 也支持直接装载EncoderDecoder。
    """

    def __init__(self, uda: Union[nn.Module, Dict], model: Union[nn.Module, Dict], work_dir: str,
                 train_dataloader: Optional[Union[DataLoader, Dict]] = None,
                 val_dataloader: Optional[Union[DataLoader, Dict]] = None,
                 test_dataloader: Optional[Union[DataLoader, Dict]] = None, train_cfg: Optional[Dict] = None,
                 val_cfg: Optional[Dict] = None, test_cfg: Optional[Dict] = None, auto_scale_lr: Optional[Dict] = None,
                 optim_wrapper: Optional[Union[OptimWrapper, Dict]] = None,
                 param_scheduler: Optional[Union[_ParamScheduler, Dict, List]] = None,
                 val_evaluator: Optional[Union[Evaluator, Dict, List]] = None,
                 test_evaluator: Optional[Union[Evaluator, Dict, List]] = None,
                 default_hooks: Optional[Dict[str, Union[Hook, Dict]]] = None,
                 custom_hooks: Optional[List[Union[Hook, Dict]]] = None,
                 data_preprocessor: Union[nn.Module, Dict, None] = None, load_from: Optional[str] = None,
                 resume: bool = False, launcher: str = 'none', env_cfg: Dict = dict(dist_cfg=dict(backend='nccl')),
                 log_processor: Optional[Dict] = None, log_level: str = 'INFO',
                 visualizer: Optional[Union[Visualizer, Dict]] = None, default_scope: str = 'mmengine',
                 randomness: Dict = dict(seed=None), experiment_name: Optional[str] = None,
                 cfg: Optional[ConfigType] = None):
        self.uda = uda
        # TODO 支持 work dir 的动态生成
        # TODO 支持多卡训练时, scale 缩放 训练轮数, lr调度器步数 等参数

        # 为UDA 设置特有的data_preprocessor
        if getattr(uda, "data_preprocessor", None) is None:
            uda['data_preprocessor'] = copy.deepcopy(model["data_preprocessor"])
            uda['data_preprocessor']['type'] = 'UDASegDataPreProcessor'

        super().__init__(model, work_dir, train_dataloader, val_dataloader, test_dataloader, train_cfg, val_cfg,
                         test_cfg, auto_scale_lr, optim_wrapper, param_scheduler, val_evaluator, test_evaluator,
                         default_hooks, custom_hooks, data_preprocessor, load_from, resume, launcher, env_cfg,
                         log_processor, log_level, visualizer, default_scope, randomness, experiment_name, cfg)

    def build_model(self, model: Union[nn.Module, Dict]) -> nn.Module:
        cp_model = copy.deepcopy(model)
        uda = copy.deepcopy(self.uda)
        setattr(uda, "model", cp_model)
        return MODELS.build(uda)

    @classmethod
    def from_cfg(cls, cfg: ConfigType) -> 'Runner':
        """Build a runner from config.

                Args:
                    cfg (ConfigType): A config used for building runner. Keys of
                        ``cfg`` can see :meth:`__init__`.

                Returns:
                    Runner: A runner build from ``cfg``.
                """
        cfg = copy.deepcopy(cfg)
        runner = cls(
            uda=cfg['uda'],
            model=cfg['model'],
            work_dir=cfg['work_dir'],
            train_dataloader=cfg.get('train_dataloader'),
            val_dataloader=cfg.get('val_dataloader'),
            test_dataloader=cfg.get('test_dataloader'),
            train_cfg=cfg.get('train_cfg'),
            val_cfg=cfg.get('val_cfg'),
            test_cfg=cfg.get('test_cfg'),
            auto_scale_lr=cfg.get('auto_scale_lr'),
            optim_wrapper=cfg.get('optim_wrapper'),
            param_scheduler=cfg.get('param_scheduler'),
            val_evaluator=cfg.get('val_evaluator'),
            test_evaluator=cfg.get('test_evaluator'),
            default_hooks=cfg.get('default_hooks'),
            custom_hooks=cfg.get('custom_hooks'),
            data_preprocessor=cfg.get('data_preprocessor'),
            load_from=cfg.get('load_from'),
            resume=cfg.get('resume', False),
            launcher=cfg.get('launcher', 'none'),
            env_cfg=cfg.get('env_cfg', dict(dist_cfg=dict(backend='nccl'))),
            log_processor=cfg.get('log_processor'),
            log_level=cfg.get('log_level', 'INFO'),
            visualizer=cfg.get('visualizer'),
            default_scope=cfg.get('default_scope', 'mmengine'),
            randomness=cfg.get('randomness', dict(seed=None)),
            experiment_name=cfg.get('experiment_name'),
            cfg=cfg,
        )

        return runner
