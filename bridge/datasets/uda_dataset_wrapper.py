# -*- coding:utf-8 -*-
"""
 @FileName   : uda_dataset_wrapper.py
 @Time       : 12/26/24 5:21 PM
 @Author     : Woldier Wong
 @Description: support uda training
"""
import warnings
import copy

import mmengine
from mmengine.dataset import BaseDataset, force_full_init
from mmengine.registry import DATASETS



@DATASETS.register_module()
class UDADataset:
    """ A wrapper of UDA source and target dataset.
    The length of uda dataset will be source dataset 'times' target dataset.

    for idx, which means we need data ``idx//len(source)`` form source
    and ``idx%len(source)`` from target

    返回的数据格式为

    >>>  dict(
    >>>     inputs=inputs,
    >>>     taget_inputs=taget_inputs,
    >>>     data_samples=dict(
    >>>         gt_sem_seg=gt_sem_seg, # source 的 label
    >>>         metainfo=dict(
    >>>                xxx=xxx # source 的 meta 信息
    >>>             ),
    >>>     ),
    >>>     {{self._target_prefix}}data_samples=dict(
    >>>         gt_sem_seg=gt_sem_seg, # target 的 label
    >>>         metainfo=dict(
    >>>                xxx=xxx # target 的 meta 信息
    >>>             ),
    >>>     ),
    >>>  )

    Args:
        source: source 的config 文件
        target: target 的config 文件
        lazy_init: 懒加载模式
        target_prefix: 用于设置target 数据key的前缀。
            默认为 tgt, 因此， 此时装载的 target 数据 其 key 为 tgt开头，如tgt_input
    """

    def __init__(self, source, target, lazy_init=False, target_prefix='tgt', **kwargs):
        self._build_dataset(source=source, target=target)
        # 记录原数据集的元信息
        if not target_prefix.endswith('_'):
            target_prefix = target_prefix + '_'
        self._target_prefix = target_prefix
        self._metainfo = self.source.metainfo
        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

    @property
    def target_prefix(self):
        return self._target_prefix

    @property
    def source(self) -> BaseDataset:
        return self._source

    @property
    def target(self) -> BaseDataset:
        return self._target

    def _tgt_key(self, key: str) -> str:
        return self._target_prefix + key

    def _build_dataset(self, **dataset_dict):
        for k, dataset in dataset_dict.items():
            # 构建原数据集（self.dataset）
            if isinstance(dataset, dict):
                _dataset = DATASETS.build(dataset)
                setattr(self, '_' + k, _dataset)
            elif isinstance(dataset, BaseDataset):
                setattr(self, '_' + k, dataset)
            else:
                raise TypeError(
                    'elements in datasets sequence should be config or '
                    f'`BaseDataset` instance, but got {type(dataset)}')

    def full_init(self):
        if self._fully_initialized:
            return

        # 将原数据集完全初始化
        self.source.full_init()
        self.target.full_init()

        self._fully_initialized = True

    @force_full_init
    def _get_ori_dataset_idx(self, idx: int):

        s, t = len(self.source), len(self.target)
        ori_idx_s = idx // t  
        ori_idx_t = idx % t
        ori_idx = (ori_idx_s, ori_idx_t)
        return ori_idx

    # 提供与 `self.dataset` 一样的对外接口。
    @force_full_init
    def get_data_info(self, idx):
        sample_idx_s, sample_idx_t = self._get_ori_dataset_idx(idx)
        return self.source.get_data_info(sample_idx_s)

    # 提供与 `self.dataset` 一样的对外接口。
    def __getitem__(self, idx):
        if not self._fully_initialized:
            warnings.warn('Please call `full_init` method manually to '
                          'accelerate the speed.')
            self.full_init()

        sample_idx_s, sample_idx_t = self._get_ori_dataset_idx(idx)
        out = self.source[sample_idx_s]
        out_t = self.target[sample_idx_t]
        tgt_inputs, tgt_data_samples = out_t["inputs"], out_t['data_samples']
        # 为了保证在UDA过程中不会错误的访问到 target data 的GT label;
        tgt_data_samples.pop('gt_sem_seg', None) # tgt_data_samples.gt_sem_seg  # other ops
        out[self._tgt_key("inputs")] = tgt_inputs
        out[self._tgt_key("data_samples")] = tgt_data_samples
        out["tgt_key_prefix"] = self.target_prefix  # 设置tgt_key_prefix，便于后续在后续处理中，加载相应的key
        return out

        # 提供与 `self.dataset` 一样的对外接口。

    @force_full_init
    def __len__(self):
        len_wrapper = len(self.source) * len(self.target)
        return len_wrapper

    # 提供与 `self.dataset` 一样的对外接口。
    @property
    def metainfo(self):
        return copy.deepcopy(self._metainfo)

