size = (512, 512)
model = dict(
    type="EncoderDecoder",
    data_preprocessor=dict(
        type="SegDataPreProcessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        size=size,
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
    ),
    backbone=dict(
        type='BridgePretrainVisionTransformer',
        arch='large',
        img_size=512,  # Default in DinoV2 is 518, change to 512.
        patch_size=16,  # Default in DinoV2 is 14, change to 16.
        layer_scale_init_value=1e-5,  # use Layer Scale
        out_type='featmap',
        out_indices=[7, 11, 15, 23],
        # layer_cfgs=dict(mem_eff_attn=True), # use mem eff attn
        bridge_config=dict(
            num_token=100, embed_dims=1024,
            num_layers=24, patch_size=16,
        ),
        init_cfg=dict(type="Pretrained", checkpoint="pretrained/vit-large-p16_dinov2_converted.pth",
                      prefix='backbone.'),
    ),
    decode_head=dict(
        type='SegmenterMaskTransformerHead',
        in_channels=1024,
        channels=1024,
        num_classes=6,
        num_layers=2,
        num_heads=16,
        embed_dims=1024,
        dropout_ratio=0.0,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)
