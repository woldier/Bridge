import math
import torch
import torch.nn as nn
from torch import Tensor
import torch.utils.checkpoint as cp
import torch.nn.functional as F
from functools import reduce
from operator import mul
from typing import Union, List, Optional, Tuple
from mmengine.model import BaseModule
from ..utils.layer_scale import LayerScale
from mmseg.utils import OptConfigType, OptMultiConfig
from mmcv.cnn import build_conv_layer, build_norm_layer
from .vision_transformer import PretrainVisionTransformer
from mmseg.registry import MODELS


class SpatialPriorModule(BaseModule):
    """
    SpatialPriorModule
    extract spatial prior

    Args:
        in_planes (int): Number of input image channels. Default: 3.
        planes (int): Number of stem channels. Default: 64.
        embed_dim (int): Number of Transformer embedding channels. Default: 384.
        conv_cfg (dict | None): Dictionary to construct and config conv layer.
            When conv_cfg is None, cfg will be set to dict(type='Conv2d').
            Default: None.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True).
    """

    def __init__(self, in_planes: int = 3, planes: int = 64, embed_dims: int = 384,
                 conv_cfg=dict(type='Conv2d'), norm_cfg=dict(type='BN'), with_cp: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.with_cp = with_cp

        ## ===============build stem========================

        stem_layer = [
            build_conv_layer(conv_cfg, in_planes, planes, 3, stride=2, padding=1, bias=False),
            build_norm_layer(norm_cfg, planes, postfix=1)[1],
            nn.ReLU(inplace=True),

            build_conv_layer(conv_cfg, planes, planes, 3, stride=1, padding=1, bias=False),
            build_norm_layer(norm_cfg, planes, postfix=1)[1],
            nn.ReLU(inplace=True),

            build_conv_layer(conv_cfg, planes, planes, 3, stride=1, padding=1, bias=False),
            build_norm_layer(norm_cfg, planes, postfix=1)[1],
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        ]
        self.stem = nn.Sequential(*stem_layer)
        ## ===============build conv========================
        conv_kw = dict(kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Sequential(*[
            build_conv_layer(conv_cfg, 2 ** 0 * planes, 2 ** 1 * planes, **conv_kw),
            build_norm_layer(norm_cfg, 2 ** 1 * planes, postfix=1)[1],
            nn.ReLU(inplace=True)
        ])
        self.conv3 = nn.Sequential(*[
            build_conv_layer(conv_cfg, 2 * planes, 2 ** 2 * planes, **conv_kw),
            build_norm_layer(norm_cfg, 2 ** 2 * planes, postfix=1)[1],
            nn.ReLU(inplace=True)

        ])
        self.conv4 = nn.Sequential(*[
            build_conv_layer(conv_cfg, 2 ** 2 * planes, 2 ** 2 * planes, **conv_kw),
            build_norm_layer(norm_cfg, 2 ** 2 * planes, postfix=1)[1],
            nn.ReLU(inplace=True)
        ])
        ## ===============build fc========================
        # self.fc1 = build_conv_layer(conv_cfg, planes, embed_dims, 1, stride=1, padding=0, bias=True)
        self.fc2 = build_conv_layer(conv_cfg, 2 * planes, embed_dims, 1, stride=1, padding=0, bias=True)
        self.fc3 = build_conv_layer(conv_cfg, 4 * planes, embed_dims, 1, stride=1, padding=0, bias=True)
        self.fc4 = build_conv_layer(conv_cfg, 4 * planes, embed_dims, 1, stride=1, padding=0, bias=True)

    def forward(self, x):

        def _inner_forward(x):
            c1 = self.stem(x)
            c2 = self.conv2(c1)
            c3 = self.conv3(c2)
            c4 = self.conv4(c3)
            # c1 = self.fc1(c1)
            c2 = self.fc2(c2)
            c3 = self.fc3(c3)
            c4 = self.fc4(c4)

            # bs, dim, _, _ = c1.shape
            # c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s
            # c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
            # c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
            # c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s

            # return c1, c2, c3, c4
            return c2, c3, c4

        if self.with_cp and x.requires_grad:
            outs = cp.checkpoint(_inner_forward, x)
        else:
            outs = _inner_forward(x)
        return outs


class SinePositionalEncoding(BaseModule):
    """Position encoding with sine and cosine functions.

    See `End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times of this value.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Defaults to 10000.
        normalize (bool, optional): Whether to normalize the position
            embedding. Defaults to False.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Defaults to 2*pi.
        eps (float, optional): A value added to the denominator for
            numerical stability. Defaults to 1e-6.
        offset (float): offset add to embed when do the normalization.
            Defaults to 0.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None
    """

    def __init__(self,
                 num_feats: int,
                 temperature: int = 10000,
                 normalize: bool = False,
                 scale: float = 2 * math.pi,
                 eps: float = 1e-6,
                 offset: float = 0.,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        if normalize:
            assert isinstance(scale, (float, int)), 'when normalize is set,' \
                                                    'scale should be provided and in float or int type, ' \
                                                    f'found {type(scale)}'
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask: Tensor, input: Optional[Tensor] = None) -> Tensor:
        """Forward function for `SinePositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].
            input (Tensor, optional): Input image/feature Tensor.
                Shape [bs, c, h, w]

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        assert not (mask is None and input is None)

        if mask is not None:
            B, H, W = mask.size()
            device = mask.device
            # For convenience of exporting to ONNX,
            # it's required to convert
            # `masks` from bool to int.
            mask = mask.to(torch.int)
            not_mask = 1 - mask  # logical_not
            y_embed = not_mask.cumsum(1, dtype=torch.float32)
            x_embed = not_mask.cumsum(2, dtype=torch.float32)
        else:
            # single image or batch image with no padding
            B, _, H, W = input.shape
            device = input.device
            x_embed = torch.arange(
                1, W + 1, dtype=torch.float32, device=device)
            x_embed = x_embed.view(1, 1, -1).repeat(B, H, 1)
            y_embed = torch.arange(
                1, H + 1, dtype=torch.float32, device=device)
            y_embed = y_embed.view(1, -1, 1).repeat(B, 1, W)
        if self.normalize:
            y_embed = (y_embed + self.offset) / \
                      (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / \
                      (x_embed[:, :, -1:] + self.eps) * self.scale
        dim_t = torch.arange(
            self.num_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        # use `view` instead of `flatten` for dynamically exporting to ONNX

        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, -1)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, -1)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def __repr__(self) -> str:
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'temperature={self.temperature}, '
        repr_str += f'normalize={self.normalize}, '
        repr_str += f'scale={self.scale}, '
        repr_str += f'eps={self.eps})'
        return repr_str


class DomainInvariantTokenAdaptation(BaseModule):
    """
        由于 ViT 的预训练是基于自然图像（ImageNet 等），它内部的表示能力偏向自然图像统计特性。
        直接用会导致遥感特征“读不懂”或“解释不准”。
        但是，无论是自然图像还是遥感图像，他们都有一些共性（Domain Invariant）。
        那么我们可以将其定义为可学习的Token，并且它在训练过程中自适应地“吸收”遥感先验，
        从而在自然图像 → 遥感图像 的迁移中起到 adapter 的作用。
        强调 latent token 自身是 domain-invariant，通过 cross attention 学会遥感 prior。

        Arg: 
            attn_cfg (:obj:`ConfigDict` or dict, optional): Config for attention.
    """  # noqa

    def __init__(self, layer_num=24, patch_size=16, token_length=100, embed_dims=256,
                 layer_scale_init_value: float = .15, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dims = embed_dims
        self.domain_invariant_token = nn.Parameter(torch.empty([layer_num, token_length, embed_dims]))
        self.project = nn.Linear(self.embed_dims, self.embed_dims)
        self.feat_project = nn.Linear(self.embed_dims, self.embed_dims)
        self._init_token()
        nn.init.kaiming_uniform_(self.project.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.feat_project.weight, a=math.sqrt(5))

        if layer_scale_init_value > 0:
            self.gamma = LayerScale(embed_dims, layer_scale_init_value=layer_scale_init_value)
        else:
            self.gamma = nn.Identity()

    def _init_token(self):
        """
        Xavier initialization.
        """
        val = math.sqrt(6.0 / float(3 * reduce(mul, (self.patch_size, self.patch_size), 1) + self.embed_dims))
        nn.init.uniform_(self.domain_invariant_token.data, -val, val)

    def forward(self, feats: Tensor, index=-1):

        token = self.domain_invariant_token[index]
        attn = torch.einsum("mc,bnc->bmn", token, feats)

        # if self.use_softmax:
        attn = attn * (self.embed_dims ** -0.5)
        attn = F.softmax(attn, dim=-1)

        feat_proj = self.feat_project(feats)
        delta_token = torch.einsum('bmn,bnc->bmc', attn, feat_proj)

        delta_t = self.gamma(self.project(token + delta_token))

        delta_token = delta_t + token

        return delta_token


class DomainAwareInjection(BaseModule):
    """
    Backbone Feature Adaptation
    ViT 中间层的自然图像特征向遥感域收缩/对齐。
    进一步缓解 自然图像 → 遥感图像 的 gap。
    """  # noqa

    def __init__(self, embed_dims, layer_scale_init_value=0.15, use_softmax=True, **kwargs):
        super().__init__(**kwargs)
        # self.use_softmax = use_softmax
        # self.scale_init = scale_init
        self.embed_dims = embed_dims
        # self.scale = nn.Parameter(torch.tensor(self.scale_init))
        self.mlp_token2feat = nn.Linear(self.embed_dims, self.embed_dims)
        self.mlp_delta_f = nn.Linear(self.embed_dims, self.embed_dims)
        nn.init.kaiming_uniform_(self.mlp_delta_f.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mlp_token2feat.weight, a=math.sqrt(5))

        if layer_scale_init_value > 0:
            self.gamma = LayerScale(embed_dims, layer_scale_init_value=layer_scale_init_value)
        else:
            self.gamma = nn.Identity()

    def forward(self, feats: Tensor, token: Tensor, batch_first=False, has_cls_token=True):
        if batch_first:
            feats = feats.permute(1, 0, 2)
        if has_cls_token:
            cls_token, feats = torch.tensor_split(feats, [1], dim=0)

        attn = torch.einsum("nbc,bmc->nbm", feats, token)
        # if self.use_softmax:
        attn = attn * (self.embed_dims ** -0.5)
        attn = F.softmax(attn, dim=-1)

        delta_f = torch.einsum("nbm,bmc->nbc", attn[:, :, 1:], self.mlp_token2feat(token[:, 1:, :]))
        delta_f = self.mlp_delta_f(delta_f + feats)

        # delta_feat = delta_f * self.scale
        delta_feat = self.gamma(delta_f)

        feats = feats + delta_feat

        if has_cls_token:
            feats = torch.cat([cls_token, feats], dim=0)
        if batch_first:
            feats = feats.permute(1, 0, 2)
        return feats


class SpatialPriorRefinement(BaseModule):
    """
    Source-to-Target Adaptation
    让源域遥感特征 注入已经适配过的 backbone 语义信息，从而逐步引导 源域 → 目标域 对齐。
    """  # noqa

    def __init__(self, embed_dims, layer_scale_init_value=0.15, use_softmax=True, **kwargs):
        super().__init__(**kwargs)
        # self.use_softmax = use_softmax
        # self.scale_init = scale_init
        self.embed_dims = embed_dims
        # self.scale = nn.Parameter(torch.tensor(self.scale_init))
        self.mlp_vit_feat_project = nn.Linear(self.embed_dims, self.embed_dims)
        self.mlp_delta_f = nn.Linear(self.embed_dims, self.embed_dims)
        nn.init.kaiming_uniform_(self.mlp_delta_f.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mlp_vit_feat_project.weight, a=math.sqrt(5))

        if layer_scale_init_value > 0:
            self.gamma = LayerScale(embed_dims, layer_scale_init_value=layer_scale_init_value)
        else:
            self.gamma = nn.Identity()

    def forward(self, feat_vit: Tensor, feat_sp: Tensor, has_cls_token=True):
        if has_cls_token:
            _, feat_vit = torch.tensor_split(feat_vit, [1], dim=1)

        attn = torch.einsum("blc,bnc->bln", feat_sp, feat_vit)
        # if self.use_softmax:
        attn = attn * (self.embed_dims ** -0.5)
        attn = F.softmax(attn, dim=-1)

        delta_f = torch.einsum("bln,bnc->blc", attn, self.mlp_vit_feat_project(feat_vit))
        delta_f = self.mlp_delta_f(delta_f + feat_sp)

        # delta_feat = delta_f * self.scale
        delta_feat = self.gamma(delta_f)

        feats = feat_sp + delta_feat
        return feats


class BridgeV2(BaseModule):
    """

    Args:
        strides (list[int] | tuple[int]): Output strides of feature from
            backbone.
        positional_encoding (:obj:`ConfigDict` or dict): Config for
            transformer encoder position encoding. Defaults to
            dict(num_feats=128, normalize=True).
    """
    MODEL_MODE = ['spm', 'dita', 'dai', 'spr']

    def __init__(self, num_token, num_layers: int, patch_size: int, embed_dims: int,
                 strides: Union[List[int], Tuple[int]] = [8, 16, 32],
                 positional_encoding: OptConfigType = dict(num_feats=128, normalize=True),
                 spm_cfg: OptConfigType = dict(),
                 dita_config: OptMultiConfig = dict(), dai_config: OptConfigType = dict(),
                 spr_config: OptMultiConfig = dict(),
                 **kwargs):
        super().__init__(**kwargs)
        self.embed_dims = embed_dims

        # =======================Spatial Prior Module================================
        spm_cfg['embed_dims'] = embed_dims
        self.spm = SpatialPriorModule(**spm_cfg)

        positional_encoding.update(num_feats=embed_dims // 2)
        self.postional_encoding = SinePositionalEncoding(**positional_encoding)
        self.level_encoding = nn.Embedding(3, embed_dims)  # 3 level for spm feature

        _dita_config = dict(patch_size=patch_size, embed_dims=embed_dims,
                            token_length=num_token, layer_num=num_layers)
        dita_config.update(_dita_config)
        self.dita = DomainInvariantTokenAdaptation(**dita_config)

        # =======================Domain Aware Injection================================
        dai_config.update(embed_dims=embed_dims)

        self.dai = DomainAwareInjection(**dai_config)

        # =======================Spatial Prior Refinement================================
        # spr_config.update(embed_dims=embed_dims)
        # self.spr = SpatialPriorRefinement(**spr_config)

    def forward(self, *arg, mode='None', **kwargs):
        """
        ['spm', 'dita', 'dai', 'spr']
        """
        if mode == 'spm':
            return self.spm(*arg, **kwargs)
        elif mode == 'dita':
            return self.dita(*arg, **kwargs)
        elif mode == 'dai':
            return self.dai(*arg, **kwargs)
        elif mode == 'spr':
            return self.spr(*arg, **kwargs)
        else:
            raise NotImplementedError


@MODELS.register_module()
class BridgePretrainVisionTransformer(PretrainVisionTransformer):
    def __init__(self,
                 bridge_config: OptConfigType,
                 # arch='base', img_size=224, patch_size=16, in_channels=3, out_indices=-1, drop_rate=0.,
                 # drop_path_rate=0., qkv_bias=True, norm_cfg=dict(type='LN', eps=1e-6), final_norm=True,
                 # out_type='cls_token', with_cls_token=True, frozen_stages=-1, interpolate_mode='bicubic',
                 # layer_scale_init_value=0., patch_cfg=dict(), layer_cfgs=dict(), pre_norm=False, init_cfg=None
                 **kwargs):
        super().__init__(**kwargs)
        if self.frozen_stages == -1:
            self.frozen_stages = len(self.layers)
            self._freeze_stages()
        self.bridge: BridgeV2 = BridgeV2(**bridge_config)

    def forward(self, x):
        B = x.shape[0]
        # ====================================SPM=============================================
        # c2, c3, c4 = self.bridge.spm(x)
        c2, c3, c4 = self.bridge(x, mode='spm')
        feat_sp = [c2, c3, c4]
        device = feat_sp[0].device
        feat_sp_input_list = []
        feat_sp_level_positional_encoding_list = []
        # feat_sp_spatial_shapes = []
        for level_idx, feat_sp_i in enumerate(feat_sp):
            feat_hw = torch._shape_as_tensor(feat_sp_i)[2:].to(device)
            padding_mask = feat_sp_i.new_zeros((B,) + feat_sp_i.shape[-2:], dtype=torch.bool)
            pos_embed = self.bridge.postional_encoding(padding_mask)
            level_embed = self.bridge.level_encoding.weight[level_idx]
            level_pos_embed = level_embed.view(1, -1, 1, 1) + pos_embed

            # shape (batch_size, c, h_i, w_i) -> ( batch_size, h_i * w_i, c)
            feat_sp_i = feat_sp_i.flatten(2).permute(0, 2, 1)
            level_pos_embed = level_pos_embed.flatten(2).permute(0, 2, 1)

            feat_sp_input_list.append(feat_sp_i)
            # feat_sp_padding_mask_list.append(padding_mask)
            feat_sp_level_positional_encoding_list.append(level_pos_embed)

            # feat_sp_spatial_shapes.append(feat_hw)

        # shape (total_num_queries, batch_size, c)
        feat_sp = torch.cat(feat_sp_input_list, dim=1)
        feat_sp_level_positional_encodings = torch.cat(feat_sp_level_positional_encoding_list, dim=1)
        feat_sp_pos_encoded = feat_sp + feat_sp_level_positional_encodings

        # feat_sp_spatial_shapes = torch.cat(feat_sp_spatial_shapes).view(-1, 2)
        # feat_sp_level_start_index = torch.cat((feat_sp_spatial_shapes.new_zeros((1,)),
        #                       feat_sp_spatial_shapes.prod(1).cumsum(0)[:-1]))
        # =====================================ViT patch && pos embed================================================

        x, patch_resolution = self.patch_embed(x)

        if self.cls_token is not None:
            # stole cls_tokens impl from Phil Wang, thanks
            cls_token = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        x = x + self.resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens)
        x = self.drop_after_pos(x)

        x = self.pre_norm(x)

        # ============================================================
        # this impl mIOU 77.44
        outs = []
        for i, layer in enumerate(self.layers):

            x = layer(x)

            # dita
            domain_spec_token = self.bridge(feat_sp_pos_encoded, mode='dita', index=i, )

            # dai
            x = self.bridge(x, domain_spec_token, mode='dai', batch_first=True, has_cls_token=True)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.ln1(x)

            if i in self.out_indices:
                outs.append(self._format_output(x, patch_resolution))

        return tuple(outs)
