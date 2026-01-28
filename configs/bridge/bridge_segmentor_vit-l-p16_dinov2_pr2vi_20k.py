_base_ = [
    '../_base_/datasets/uda_potsdamRGB_2_vaihingenIRRG_512x512.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/poly_schedule_20k.py',
    '../_base_/schedules/adamw.py',
    './bridge_dinov2_segmentor.py'
]
# # =====================UDA==============================
uda = dict(
    type='DACS',
    teacher=dict(type='EMATeacher', pseudo_threshold=0.968),
)
# =====================runner==============================
runner_type = 'UDASupportedRunner'
# # =====================dataset==============================
train_dataloader = dict(batch_size=2, )
optim_wrapper = dict( # 使用Amp
    type='AmpOptimWrapper',
    paramwise_cfg=dict(  # 为不同的位置设置不同的
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            norm=dict(decay_mult=0.0),
            domain_invariant_token=dict(decay_mult=0.0),
            level_embed=dict(decay_mult=0.0),
        )
    ),
    optimizer=dict(lr=6e-5, betas=(0.9, 0.999), weight_decay=0.05),
    constructor='PEFTOptimWrapperConstructor'
)
# ========================IterLoopConfig==================================
train_cfg = dict(type='IterBasedTrainLoop', val_interval=2000)

model = dict(decode_head=dict(num_classes={{_base_.num_classes}},))