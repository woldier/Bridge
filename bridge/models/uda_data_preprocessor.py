# -*- coding:utf-8 -*-
"""
 @FileName   : uda_data_preprocessor.py
 @Time       : 1/2/25 2:54 PM
 @Author     : Woldier Wong
 @Description: 支持UDA 训练的 data pre-processor
"""
from sys import prefix
from typing import Dict, Any
import torch

import mmengine
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmseg.utils import stack_batch

from mmseg.registry import MODELS



@MODELS.register_module()
class UDASegDataPreProcessor(SegDataPreProcessor):
    """UDASegDataPreProcessor
    本处理器提供对UDA训练中数据预处理的支持。

    在UDA训练中，除了需要源域数据外，还需要目标域数据。
    我们约定，源域数据的key与SegDataPreProcessor接收的key是相同的。
    而对于目标域数据，其key有前缀如 target_inputs target_data_samples。
    在这种情况下，目标域数据key的命名与源域数据的命名格式相同，唯一的区别就是存在前缀。
    数据格式的定义详见 Wrapper类 UDADataset

    而在UDA的测试过程中（val 和 test），初始化的数据集是目标域数据集， 而不是Wrapper类 UDADataset
    因此，这种情况下还要保留 SegDataPreProcessor 的行为
    """

    def forward(self, data: dict, training: bool = False) -> Dict[str, Any]:
        """Perform normalization、padding and bgr2rgb conversion based on
               ``SegDataPreProcessor``.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            Dict: Data in the same format as the model input.
       """
        data = self.cast_data(data)  # type: ignore
        inputs = data['inputs']
        data_samples = data.get('data_samples', None)
        # # TODO: whether normalize should be after stack_batch
        # 为了在training中用公用操作处理source data , 此处将其注释了
        # 并且, 为了保持在非 training 情况下行为的一致性, 在else 中执行了这行代码
        # inputs = self._inputs_pre_process(inputs)

        if training:
            # 在训练条件下，有target data需要处理
            tgt_key_prefix = data.setdefault('tgt_key_prefix',None)
            assert tgt_key_prefix is not None, 'During UDA training, `tgt_key_prefix` must be define in dataset.'
            tgt_key_prefix = tgt_key_prefix[0] if isinstance(tgt_key_prefix, list) else tgt_key_prefix
            assert data_samples is not None, 'During training, `data_samples` must be define.'
            extra_fields = data.get('extra_fields', [])  # 对于提供 extra_fields 的情况下, 支持处理更多额外的数据
            # 用于是批次数据, sampler会将很多样本的extra_fields 会拼接在一起形成 list 或者tuple 因此需要额外的处理
            if mmengine.is_list_of(extra_fields, expected_type=(list,tuple)): extra_fields = [i[0] for i in extra_fields]
            extra_fields.append('')  # 添加处理处理source的prefix
            if tgt_key_prefix not in extra_fields: extra_fields.append(tgt_key_prefix)

            out= dict(tgt_key_prefix=tgt_key_prefix)
            for prefix_name in extra_fields:
                a_inputs = data[f"{prefix_name}inputs"]
                a_data_samples = data[f'{prefix_name}data_samples']
                # inputs pre-process
                a_inputs = self._inputs_pre_process(a_inputs)
                # stack samples
                a_inputs, a_data_samples = stack_batch(
                    inputs=a_inputs, data_samples=a_data_samples,
                    size=self.size, size_divisor=self.size_divisor,
                    pad_val=self.pad_val, seg_pad_val=self.seg_pad_val # type: ignore
                )
                if self.batch_augments is not None:
                    a_inputs, a_data_samples = self.batch_augments(a_inputs, a_data_samples)
                out[f'{prefix_name}inputs'] = a_inputs
                out[f'{prefix_name}data_samples'] = a_data_samples

            # # stack source
            # inputs, data_samples = stack_batch(
            #     inputs=inputs, data_samples=data_samples,
            #     size=self.size, size_divisor=self.size_divisor,
            #     pad_val=self.pad_val, seg_pad_val=self.seg_pad_val)  # type: ignore
            #
            # if self.batch_augments is not None:
            #     inputs, data_samples = self.batch_augments(inputs, data_samples)
            # out = dict(inputs=inputs, data_samples=data_samples)
            #
            # # 获取target inputs
            # target_inputs = data[tgt_key_prefix + 'inputs']
            # # 获取target data_samples 用于stack_batch
            # target_data_samples = data[tgt_key_prefix + 'data_samples']
            # # target inputs pre-process
            # target_inputs = self._inputs_pre_process(target_inputs)
            # # stack target
            # target_inputs, target_data_samples = stack_batch(
            #     inputs=target_inputs, data_samples=target_data_samples,
            #     size=self.size, size_divisor=self.size_divisor,
            #     pad_val=self.pad_val, seg_pad_val=self.seg_pad_val)  # type: ignore
            # if self.batch_augments is not None:
            #     target_inputs, target_data_samples = self.batch_augments(target_inputs, target_data_samples)
            # out[f'{tgt_key_prefix}inputs'] = target_inputs
            # out[f'{tgt_key_prefix}data_samples'] = target_data_samples
            # out['tgt_key_prefix'] = tgt_key_prefix
            return out
        else:
            inputs = self._inputs_pre_process(inputs)  # norm 保证 training 和test 行为的一致性
            img_size = inputs[0].shape[1:]
            assert all(input_.shape[1:] == img_size for input_ in inputs), \
                'The image size in a batch should be the same.'
            # pad images when testing
            if self.test_cfg:
                inputs, padded_samples = stack_batch(
                    inputs=inputs,
                    size=self.test_cfg.get('size', None),
                    size_divisor=self.test_cfg.get('size_divisor', None),
                    pad_val=self.pad_val,
                    seg_pad_val=self.seg_pad_val)
                for data_sample, pad_info in zip(data_samples, padded_samples):
                    data_sample.set_metainfo({**pad_info})
            else:
                inputs = torch.stack(inputs, dim=0)

            return dict(inputs=inputs, data_samples=data_samples)

    def _inputs_pre_process(self, inputs):
        if self.channel_conversion and inputs[0].size(0) == 3:
            inputs = [_input[[2, 1, 0], ...] for _input in inputs]
        inputs = [_input.float() for _input in inputs]
        if self._enable_normalize:
            inputs = [(_input - self.mean) / self.std for _input in inputs]
        return inputs
