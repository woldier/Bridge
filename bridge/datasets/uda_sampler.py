# -*- coding:utf-8 -*-
"""
 @FileName   : uda_sampler.py
 @Time       : 12/26/24 9:55 PM
 @Author     : Woldier Wong
 @Description: UDAInfiniteSampler
"""
import itertools
import math
from typing import Iterator, Optional, Sized

import torch
from torch.utils.data import Sampler

from mmengine.dist import get_dist_info, sync_random_seed
from mmengine.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class UDAInfiniteSampler(Sampler):
    """It's designed for iteration-based runner and yields a mini-batch indices
    each time. For UDA use.
    
    首先, 如果直接使用InfiniteSampler进行采样的话, 也是ok的.
    但是这时候面临如下问题.
    
    由于dataset的长度是len(dataset.source)*len(dataset.target),  这就导致了采样有可能是不均匀的.
    因为对于idx索引, 获取的source data 是 idx//len(dataset.source), 获取的target data 是 idx%len(dataset.source)
    
    假设source 和 target 的长度都是2048, 那么dataset 的总长度就是2048*2048.
    假设迭代次数是40k. 且batch size 是6. 
    
    (2048*2048)/(40000*6)≈17 
    那么假设在不shuffle的情况下, 从最不利的情况下 智能覆盖到source data 的1/17. 
    就算多卡的情况下, 假设8张卡并行, 也有一半的source data无法获取到. 这就是不合理的.
    
    因此我们需要一个更加适合UDA dataset 的采样方式, 至少能够保证source data 或者 target data 被hint的次数是比较平均的.
    
    这里我们的实现是, 将source 或者target 当作采样主体, 然后让另剩下的 domain 数据去适配它.

    Args:
        dataset (Sized): The dataset.
        shuffle (bool): Whether shuffle the dataset or not. Defaults to True.
        seed (int, optional): Random seed. If None, set a random seed.
            Defaults to None.
    """  # noqa: W605

    def __init__(self,
                 dataset: Sized,
                 shuffle: bool = True,
                 seed: Optional[int] = None) -> None:
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        assert hasattr(dataset, "source") and hasattr(dataset, "target"), "dataset mast have source and target"
        self.world_size = world_size
        self.rank = rank
        self.shuffle = shuffle
        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
        self.size = len(dataset)  # 这里的dataset 是 UDA Wrapper 其数据集长度约定为 s * t
        self.indices = self._indices_of_rank()

    def _infinite_indices(self, size) -> Iterator[int]:
        """Infinitely yield a sequence of indices."""
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            if self.shuffle:
                yield from torch.randperm(size, generator=g).tolist()
            else:
                yield from torch.arange(size).tolist()

    def _indices_of_rank(self) -> Iterator[int]:
        """Slice the infinite indices by rank."""
        s , t = len(self.dataset.source), len(self.dataset.target)
        i1 = itertools.islice(self._infinite_indices(s), self.rank, None,
                              self.world_size)
        i2 = itertools.islice(self._infinite_indices(t), self.rank, None,
                              self.world_size)
        while True:
            yield next(i1) * t + next(i2)

    def __iter__(self) -> Iterator[int]:
        """Iterate the indices."""
        yield from self.indices

    def __len__(self) -> int:
        """Length of base dataset."""
        return self.size

    def set_epoch(self, epoch: int) -> None:
        """Not supported in iteration-based runner."""
        pass
