# -*- coding:utf-8 -*-
optimizer = dict(type='AdamW', lr=0.0006, betas=(0.9, 0.999), weight_decay=0.01,)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
