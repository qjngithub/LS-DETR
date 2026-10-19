import torch
import torch.nn as nn
from typing import Optional, Sequence

def make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original TensorFlow repo.
    It ensures that all layers have a channel number that is divisible by 8
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    return new_v


class Convls(nn.Module):
    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p: Optional[int] = None,
        g: int = 1,
        d: int = 1,
        bn: bool = True,
        act: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, padding=autopad(k, p, d), dilation=d, groups=g, bias=False
        )
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


def autopad(k, p=None, d=1):
    # 自动计算 padding，考虑 dilation
    if p is None:
        p = (k // 2) * d
    return p


# class LSMBlock(nn.Module):
#     expansion = 1

#     def __init__(
#         self,
#         ch_in: int,
#         ch_out: int,
#         stride: int = 1,
#         shortcut: bool = False,
#         variant: str = 'b',
#         act: bool = True,
#         kernel_sizes: Sequence[int] = (3, 5, 7, 9, 11),
#         dilations: Sequence[int] = (1, 1, 1, 1, 1),
#         expansion: float = 0.5,
#         fc_ratio: int = 16,
#         init_cfg: Optional[dict] = None
#     ):
#         super().__init__()
#         self.shortcut = shortcut
#         self.ch_out = ch_out

#         # 计算隐藏通道
#         hidden_channels = make_divisible(int(ch_out * expansion), 8)
#         # hidden_channels = ch_in//4
#         # concat_channels = 2 * hidden_channels

#         # 分支1：大核卷积分支
#         # 1x1 卷积降维
#         self.large_kernel_pre = Convls(
#             c1=ch_in,
#             c2=hidden_channels,
#             k=1,
#             s=1,
#             p=0,
#             bn=True,
#             act=True,
#             g=1
#         )

#         # 多尺度大核卷积分支
#         self.dw_conv0 = Convls(
#             c1=hidden_channels,
#             c2=hidden_channels,
#             k=kernel_sizes[0],
#             s=1,
#             d=dilations[0],
#             g=hidden_channels,
#             bn=False,
#             act=False
#         )
#         self.dw_conv1 = Convls(
#             c1=hidden_channels,
#             c2=hidden_channels,
#             k=kernel_sizes[1],
#             s=1,
#             d=dilations[1],
#             g=hidden_channels,
#             bn=False,
#             act=False
#         )
#         self.dw_conv2 = Convls(
#             c1=hidden_channels,
#             c2=hidden_channels,
#             k=kernel_sizes[2],
#             s=1,
#             d=dilations[2],
#             g=hidden_channels,
#             bn=False,
#             act=False
#         )
#         self.dw_conv3 = Convls(
#             c1=hidden_channels,
#             c2=hidden_channels,
#             k=kernel_sizes[3],
#             s=1,
#             d=dilations[3],
#             g=hidden_channels,
#             bn=False,
#             act=False
#         )
#         self.dw_conv4 = Convls(
#             c1=hidden_channels,
#             c2=hidden_channels,
#             k=kernel_sizes[4],
#             s=1,
#             d=dilations[4],
#             g=hidden_channels,
#             bn=False,
#             act=False
#         )

#         # 后处理卷积
#         self.large_kernel_post = Convls(
#             c1=hidden_channels,
#             c2=hidden_channels,
#             k=1,
#             s=1,
#             p=0,
#             bn=True,
#             act=True
#         )
#         self.final_conv = Convls(
#             hidden_channels, ch_out, k=1, s=1, p=0, bn=True, g=1, act=True
#         )

#     def forward(self, x):
#         # identity = x

#         # 分支1：大核卷积分支
#         large_kernel_x = self.large_kernel_pre(x)
#         identity = large_kernel_x
#         # 多尺度卷积并叠加
#         x_dw = (
#             self.dw_conv0(large_kernel_x)
#             + self.dw_conv1(large_kernel_x)
#             + self.dw_conv2(large_kernel_x)
#             + self.dw_conv3(large_kernel_x)
#             + self.dw_conv4(large_kernel_x)
#         )

#         large_kernel_out = self.large_kernel_post(x_dw)

#         large_kernel_out = large_kernel_out +identity
#         # 最终输出卷积
#         out = self.final_conv(large_kernel_out)

#         # 残差连接（可选）
        
#         # out = out + identity

#         return out
# 移除了最大的核之后的
class LSMBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        stride: int = 1,
        shortcut: bool = False,
        variant: str = 'b',
        act: bool = True,
        kernel_sizes: Sequence[int] = (3,5, 7, 9,11),
        dilations: Sequence[int] = (1, 1, 1, 1,1),
        expansion: float = 0.5,
        fc_ratio: int = 16,
        init_cfg: Optional[dict] = None
    ):
        super().__init__()
        self.shortcut = shortcut
        self.ch_out = ch_out

        # 计算隐藏通道
        hidden_channels = make_divisible(int(ch_out * expansion), 8)
        # hidden_channels = ch_in//4
        # concat_channels = 2 * hidden_channels

        # 分支1：大核卷积分支
        # 1x1 卷积降维
        self.large_kernel_pre = Convls(
            c1=ch_in,
            c2=hidden_channels,
            k=1,
            s=1,
            p=0,
            bn=True,
            act=True,
            g=1
        )

        # 多尺度大核卷积分支
        self.dw_conv0 = Convls(
            c1=hidden_channels,
            c2=hidden_channels,
            k=kernel_sizes[0],
            s=1,
            d=dilations[0],
            g=hidden_channels,
            bn=False,
            act=False
        )
        self.dw_conv1 = Convls(
            c1=hidden_channels,
            c2=hidden_channels,
            k=kernel_sizes[1],
            s=1,
            d=dilations[1],
            g=hidden_channels,
            bn=False,
            act=False
        )
        self.dw_conv2 = Convls(
            c1=hidden_channels,
            c2=hidden_channels,
            k=kernel_sizes[2],
            s=1,
            d=dilations[2],
            g=hidden_channels,
            bn=False,
            act=False
        )
        self.dw_conv3 = Convls(
            c1=hidden_channels,
            c2=hidden_channels,
            k=kernel_sizes[3],
            s=1,
            d=dilations[3],
            g=hidden_channels,
            bn=False,
            act=False
        )
        self.dw_conv4 = Convls(
            c1=hidden_channels,
            c2=hidden_channels,
            k=kernel_sizes[4],
            s=1,
            d=dilations[4],
            g=hidden_channels,
            bn=False,
            act=False
        )

        # 后处理卷积
        self.large_kernel_post = Convls(
            c1=hidden_channels,
            c2=hidden_channels,
            k=1,
            s=1,
            p=0,
            bn=True,
            act=True
        )
        self.final_conv = Convls(
            hidden_channels, ch_out, k=1, s=1, p=0, bn=True, g=1, act=True
        )

    def forward(self, x):
        # identity = x

        # 分支1：大核卷积分支
        large_kernel_x = self.large_kernel_pre(x)
        identity = large_kernel_x
        # 多尺度卷积并叠加
        x_dw = (
            self.dw_conv0(large_kernel_x)
            + self.dw_conv1(large_kernel_x)
            + self.dw_conv2(large_kernel_x)
            + self.dw_conv3(large_kernel_x)
            + self.dw_conv4(large_kernel_x)
        )

        large_kernel_out = self.large_kernel_post(x_dw)

        large_kernel_out = large_kernel_out +identity
        # 最终输出卷积
        out = self.final_conv(large_kernel_out)

        # 残差连接（可选）
        
        # out = out + identity

        return out
    

class DropPath(torch.nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output
    
    
"""
reference
- https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py

Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from .common import FrozenBatchNorm2d
from ..core import register
import logging

# Constants for initialization
kaiming_normal_ = nn.init.kaiming_normal_
zeros_ = nn.init.zeros_
ones_ = nn.init.ones_

__all__ = ['HGNetv2']

class FrozenBatchNorm2d(nn.Module):
    """copy and modified from https://github.com/facebookresearch/detr/blob/master/models/backbone.py
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """
    def __init__(self, num_features, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        n = num_features
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps
        self.num_features = n

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

    def extra_repr(self):
        return (
            "{num_features}, eps={eps}".format(**self.__dict__)
        )

def freeze_batch_norm2d(module: nn.Module) -> nn.Module:
    if isinstance(module, nn.BatchNorm2d):
        module = FrozenBatchNorm2d(module.num_features)
    else:
        for name, child in module.named_children():
            _child = freeze_batch_norm2d(child)
            if _child is not child:
                setattr(module, name, _child)
    return module


def get_activation(act: str, inplace: bool=True):
    """get activation
    """
    if act is None:
        return nn.Identity()

    elif isinstance(act, nn.Module):
        return act

    act = act.lower()

    if act == 'silu' or act == 'swish':
        m = nn.SiLU()

    elif act == 'relu':
        m = nn.ReLU()

    elif act == 'leaky_relu':
        m = nn.LeakyReLU()

    elif act == 'silu':
        m = nn.SiLU()

    elif act == 'gelu':
        m = nn.GELU()

    elif act == 'hardsigmoid':
        m = nn.Hardsigmoid()

    else:
        raise RuntimeError('')

    if hasattr(m, 'inplace'):
        m.inplace = inplace

    return m

class LearnableAffineBlock(nn.Module):
    def __init__(
            self,
            scale_value=1.0,
            bias_value=0.0
    ):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([scale_value]), requires_grad=True)
        self.bias = nn.Parameter(torch.tensor([bias_value]), requires_grad=True)

    def forward(self, x):
        return self.scale * x + self.bias


class ConvBNAct(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size,
            stride=1,
            groups=1,
            padding='',
            use_act=True,
            use_lab=False
    ):
        super().__init__()
        self.use_act = use_act
        self.use_lab = use_lab
        if padding == 'same':
            self.conv = nn.Sequential(
                nn.ZeroPad2d([0, 1, 0, 1]),
                nn.Conv2d(
                    in_chs,
                    out_chs,
                    kernel_size,
                    stride,
                    groups=groups,
                    bias=False
                )
            )
        else:
            self.conv = nn.Conv2d(
                in_chs,
                out_chs,
                kernel_size,
                stride,
                padding=(kernel_size - 1) // 2,
                groups=groups,
                bias=False
            )
        self.bn = nn.BatchNorm2d(out_chs)
        if self.use_act:
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()
        if self.use_act and self.use_lab:
            self.lab = LearnableAffineBlock()
        else:
            self.lab = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.lab(x)
        return x


class LightConvBNAct(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size,
            groups=1,
            use_lab=False,
    ):
        super().__init__()
        self.conv1 = ConvBNAct(
            in_chs,
            out_chs,
            kernel_size=1,
            use_act=False,
            use_lab=use_lab,
        )
        self.conv2 = ConvBNAct(
            out_chs,
            out_chs,
            kernel_size=kernel_size,
            groups=out_chs,
            use_act=True,
            use_lab=use_lab,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class StemBlock(nn.Module):
    # for HGNetv2
    def __init__(self, in_chs, mid_chs, out_chs, use_lab=False):
        super().__init__()
        self.stem1 = ConvBNAct(
            in_chs,
            mid_chs,
            kernel_size=3,
            stride=2,
            use_lab=use_lab,
        )
        self.stem2a = ConvBNAct(
            mid_chs,
            mid_chs // 2,
            kernel_size=2,
            stride=1,
            use_lab=use_lab,
        )
        self.stem2b = ConvBNAct(
            mid_chs // 2,
            mid_chs,
            kernel_size=2,
            stride=1,
            use_lab=use_lab,
        )
        self.stem3 = ConvBNAct(
            mid_chs * 2,
            mid_chs,
            kernel_size=3,
            stride=2,
            use_lab=use_lab,
        )
        self.stem4 = ConvBNAct(
            mid_chs,
            out_chs,
            kernel_size=1,
            stride=1,
            use_lab=use_lab,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)

    def forward(self, x):
        x = self.stem1(x)
        x = F.pad(x, (0, 1, 0, 1))
        x2 = self.stem2a(x)
        x2 = F.pad(x2, (0, 1, 0, 1))
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


# class EseModule(nn.Module):
#     def __init__(self, chs):
#         super().__init__()
#         self.conv = nn.Conv2d(
#             chs,
#             chs,
#             kernel_size=1,
#             stride=1,
#             padding=0,
#         )
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         identity = x
#         x = x.mean((2, 3), keepdim=True)
#         x = self.conv(x)
#         x = self.sigmoid(x)
#         return torch.mul(identity, x)+identity
import torch
import torch.nn as nn

class EseModule(nn.Module):
    def __init__(self, dim, fc_ratio=16, act_layer=nn.ReLU6):
        """
        改进版 ESE 模块：轻量通道注意力模块
        :param dim: 输入通道数
        :param fc_ratio: 压缩比，用于控制中间通道数
        :param act_layer: 激活函数，默认为 ReLU6（适合移动端部署）
        """
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim // fc_ratio, kernel_size=1, bias=False),
            act_layer(),
            nn.Conv2d(dim // fc_ratio, dim, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x
        y = self.avgpool(x)
        y = self.fc(y)
        return x * y + identity

class HG_Block(nn.Module):
    def __init__(
        self,
        in_chs,
        mid_chs,
        out_chs,
        layer_num,
        kernel_size=3,
        residual=False,
        light_block=False,
        use_lab=False,
        agg='ese',
        drop_path=0.,
        use_lsm_branch=True,  # 是否使用 LSM 分支
        lsm_expansion=0.5,     # LSM 的通道扩展比例
    ):
        super().__init__()
        self.residual = residual
        self.use_lsm_branch = use_lsm_branch

        # 主分支：原来的 dense layers
        self.layers = nn.ModuleList()
        for i in range(layer_num):
            if light_block:
                self.layers.append(
                    LightConvBNAct(
                        in_chs if i == 0 else mid_chs,
                        mid_chs,
                        kernel_size=kernel_size,
                        use_lab=use_lab,
                    )
                )
            else:
                self.layers.append(
                    ConvBNAct(
                        in_chs if i == 0 else mid_chs,
                        mid_chs,
                        kernel_size=kernel_size,
                        stride=1,
                        use_lab=use_lab,
                    )
                )

        # 大核卷积分支（LSM）
        if self.use_lsm_branch:
            self.lsm_branch = LSMBranch(
                in_chs=in_chs,
                out_chs=out_chs,
                expansion=lsm_expansion
            )

        # feature aggregation
        total_chs = in_chs + layer_num* mid_chs

        # 如果使用 LSM 分支，则总通道数增加
        if self.use_lsm_branch:
            total_chs = in_chs + layer_num * mid_chs + out_chs

        if agg == 'se':
            aggregation_squeeze_conv = ConvBNAct(
                total_chs,
                out_chs // 2,
                kernel_size=1,
                stride=1,
                use_lab=use_lab,
            )
            aggregation_excitation_conv = ConvBNAct(
                out_chs // 2,
                out_chs,
                kernel_size=1,
                stride=1,
                use_lab=use_lab,
            )
            self.aggregation = nn.Sequential(
                aggregation_squeeze_conv,
                aggregation_excitation_conv,
            )
        else:
            aggregation_conv = ConvBNAct(
                total_chs,
                out_chs,
                kernel_size=1,
                stride=1,
                use_lab=use_lab,
            )
            att = EseModule(out_chs)
            self.aggregation = nn.Sequential(
                aggregation_conv,
                att,
            )

        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        identity = x

        # 主分支：dense 拼接
        output = [x]
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        main_branch = torch.cat(output, dim=1)

        # 大核卷积分支（LSM）
        if self.use_lsm_branch:
            lsm_feat = self.lsm_branch(identity)
            combined = torch.cat([main_branch, lsm_feat], dim=1)
        else:
            combined = main_branch

        # 融合输出
        x = self.aggregation(combined)

        # 残差连接,这个原本就没加
        # if self.residual:
        #     x = self.drop_path(x) + identity

        return x

class LSMBranch(nn.Module):
    def __init__(self, in_chs, out_chs, expansion=0.5):
        super().__init__()
        self.conv = LSMBlock(
            ch_in=in_chs,
            ch_out=out_chs,
            stride=1,
            shortcut=False,
            expansion=expansion,
            act=True
        )

    def forward(self, x):
        return self.conv(x)
class HG_Stage(nn.Module):
    def __init__(
            self,
            in_chs,
            mid_chs,
            out_chs,
            block_num,
            layer_num,
            downsample=True,
            light_block=False,
            kernel_size=3,
            use_lab=False,
            agg='se',
            drop_path=0.,
    ):
        super().__init__()
        self.downsample = downsample
        if downsample:
            self.downsample = ConvBNAct(
                in_chs,
                in_chs,
                kernel_size=3,
                stride=2,
                groups=in_chs,
                use_act=False,
                use_lab=use_lab,
            )
        else:
            self.downsample = nn.Identity()

        blocks_list = []
        for i in range(block_num):
            blocks_list.append(
                HG_Block(
                    in_chs if i == 0 else out_chs,
                    mid_chs,
                    out_chs,
                    layer_num,
                    residual=False if i == 0 else True,
                    kernel_size=kernel_size,
                    light_block=light_block,
                    use_lab=use_lab,
                    agg=agg,
                    drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path,
                )
            )
        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x



@register()
class HGNetv2(nn.Module):
    """
    HGNetV2
    Args:
        stem_channels: list. Number of channels for the stem block.
        stage_type: str. The stage configuration of HGNet. such as the number of channels, stride, etc.
        use_lab: boolean. Whether to use LearnableAffineBlock in network.
        lr_mult_list: list. Control the learning rate of different stages.
    Returns:
        model: nn.Layer. Specific HGNetV2 model depends on args.
    """

    arch_configs = {
        'B0': {
            'stem_channels': [3, 16, 16],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [16, 16, 64, 1, False, False, 3, 3],
                "stage2": [64, 32, 256, 1, True, False, 3, 3],
                "stage3": [256, 64, 512, 2, True, True, 5, 3],
                "stage4": [512, 128, 1024, 1, True, True, 5, 3],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth'
        },
        'B1': {
            'stem_channels': [3, 24, 32],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [32, 32, 64, 1, False, False, 3, 3],
                "stage2": [64, 48, 256, 1, True, False, 3, 3],
                "stage3": [256, 96, 512, 2, True, True, 5, 3],
                "stage4": [512, 192, 1024, 1, True, True, 5, 3],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B1_stage1.pth'
        },
        'B2': {
            'stem_channels': [3, 24, 32],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [32, 32, 96, 1, False, False, 3, 4],
                "stage2": [96, 64, 384, 1, True, False, 3, 4],
                "stage3": [384, 128, 768, 3, True, True, 5, 4],
                "stage4": [768, 256, 1536, 1, True, True, 5, 4],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B2_stage1.pth'
        },
        'B3': {
            'stem_channels': [3, 24, 32],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [32, 32, 128, 1, False, False, 3, 5],
                "stage2": [128, 64, 512, 1, True, False, 3, 5],
                "stage3": [512, 128, 1024, 3, True, True, 5, 5],
                "stage4": [1024, 256, 2048, 1, True, True, 5, 5],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B3_stage1.pth'
        },
        'B4': {
            'stem_channels': [3, 32, 48],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [48, 48, 128, 1, False, False, 3, 6],
                "stage2": [128, 96, 512, 1, True, False, 3, 6],
                "stage3": [512, 192, 1024, 3, True, True, 5, 6],
                "stage4": [1024, 384, 2048, 1, True, True, 5, 6],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B4_stage1.pth'
        },
        'B5': {
            'stem_channels': [3, 32, 64],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [64, 64, 128, 1, False, False, 3, 6],
                "stage2": [128, 128, 512, 2, True, False, 3, 6],
                "stage3": [512, 256, 1024, 5, True, True, 5, 6],
                "stage4": [1024, 512, 2048, 2, True, True, 5, 6],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B5_stage1.pth'
        },
        'B6': {
            'stem_channels': [3, 48, 96],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [96, 96, 192, 2, False, False, 3, 6],
                "stage2": [192, 192, 512, 3, True, False, 3, 6],
                "stage3": [512, 384, 1024, 6, True, True, 5, 6],
                "stage4": [1024, 768, 2048, 3, True, True, 5, 6],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B6_stage1.pth'
        },
    }

    def __init__(self,
                 name,
                 use_lab=False,
                 return_idx=[1, 2, 3],
                 freeze_stem_only=True,
                 freeze_at=0,
                 freeze_norm=True,
                 pretrained=True,
                 local_model_dir='weight/hgnetv2/'):
        super().__init__()
        self.use_lab = use_lab
        self.return_idx = return_idx

        stem_channels = self.arch_configs[name]['stem_channels']
        stage_config = self.arch_configs[name]['stage_config']
        download_url = self.arch_configs[name]['url']

        self._out_strides = [4, 8, 16, 32]
        self._out_channels = [stage_config[k][2] for k in stage_config]

        # stem
        self.stem = StemBlock(
                in_chs=stem_channels[0],
                mid_chs=stem_channels[1],
                out_chs=stem_channels[2],
                use_lab=use_lab)

        # stages
        self.stages = nn.ModuleList()
        for i, k in enumerate(stage_config):
            in_channels, mid_channels, out_channels, block_num, downsample, light_block, kernel_size, layer_num = stage_config[
                k]
            self.stages.append(
                HG_Stage(
                    in_channels,
                    mid_channels,
                    out_channels,
                    block_num,
                    layer_num,
                    downsample,
                    light_block,
                    kernel_size,
                    use_lab))

        if freeze_at >= 0:
            self._freeze_parameters(self.stem)
            if not freeze_stem_only:
                for i in range(min(freeze_at + 1, len(self.stages))):
                    self._freeze_parameters(self.stages[i])

        if freeze_norm:
            self._freeze_norm(self)

        if pretrained:
            RED, GREEN, RESET = "\033[91m", "\033[92m", "\033[0m"
            try:
                model_path = local_model_dir + 'PPHGNetV2_' + name + '_stage1.pth'
                if os.path.exists(model_path):
                    state = torch.load(model_path, map_location='cpu')
                    print(f"Loaded stage1 {name} HGNetV2 from local file.")
                else:
                    # If the file doesn't exist locally, download from the URL
                    if torch.distributed.get_rank() == 0:
                        print(GREEN + "If the pretrained HGNetV2 can't be downloaded automatically. Please check your network connection." + RESET)
                        print(GREEN + "Please check your network connection. Or download the model manually from " + RESET + f"{download_url}" + GREEN + " to " + RESET + f"{local_model_dir}." + RESET)
                        state = torch.hub.load_state_dict_from_url(download_url, map_location='cpu', model_dir=local_model_dir)
                        torch.distributed.barrier()
                    else:
                        torch.distributed.barrier()
                        state = torch.load(local_model_dir)

                    print(f"Loaded stage1 {name} HGNetV2 from URL.")

                self.load_state_dict(state)

            except (Exception, KeyboardInterrupt) as e:
                if torch.distributed.get_rank() == 0:
                    print(f"{str(e)}")
                    logging.error(RED + "CRITICAL WARNING: Failed to load pretrained HGNetV2 model" + RESET)
                    logging.error(GREEN + "Please check your network connection. Or download the model manually from " \
                                + RESET + f"{download_url}" + GREEN + " to " + RESET + f"{local_model_dir}." + RESET)
                exit()




    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs
# def count_parameters(model):
#     """计算模型总参数和可训练参数数量"""
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     return total_params, trainable_params

# def get_layer_info(model, input_size=(3, 640, 640)):
#     """收集模型各层的详细信息"""
#     layer_info = []
#     input_tensor = torch.randn(1, *input_size)  # 创建随机输入张量
    
#     # 用于存储各层的输入输出形状
#     layer_shapes = {}
    
#     # 注册钩子函数获取各层输入输出形状
#     def hook_fn(name):
#         def hook(module, input, output):
#             input_shape = tuple(input[0].shape) if input else None
#             output_shape = tuple(output.shape) if output is not None else None
#             layer_shapes[name] = {
#                 'input_shape': input_shape,
#                 'output_shape': output_shape
#             }
#         return hook
    
#     # 为每个模块注册钩子
#     hooks = []
#     for name, module in model.named_modules():
#         if name:  # 跳过模型本身
#             hooks.append(module.register_forward_hook(hook_fn(name)))
    
#     # 前向传播以触发钩子
#     with torch.no_grad():
#         model(input_tensor)
    
#     # 移除钩子
#     for hook in hooks:
#         hook.remove()
    
#     # 收集各层信息
#     for name, module in model.named_modules():
#         if not name:  # 跳过模型本身
#             continue
            
#         # 获取层类型
#         layer_type = module.__class__.__name__
        
#         # 获取输入输出形状
#         shapes = layer_shapes.get(name, {})
#         input_shape = shapes.get('input_shape')
#         output_shape = shapes.get('output_shape')
        
#         # 计算参数数量
#         params = sum(p.numel() for p in module.parameters())
#         trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        
#         layer_info.append({
#             'name': name,
#             'type': layer_type,
#             'input_shape': input_shape,
#             'output_shape': output_shape,
#             'total_params': params,
#             'trainable_params': trainable_params
#         })
    
#     return layer_info

# def save_model_info(model_name, total_params, trainable_params, layer_info, save_dir='model_info'):
#     """保存模型信息到文件"""
#     # 创建保存目录
#     os.makedirs(save_dir, exist_ok=True)
    
#     # 保存参数摘要
#     summary_path = os.path.join(save_dir, f'{model_name}_summary.txt')
#     with open(summary_path, 'w', encoding='utf-8') as f:
#         f.write(f"模型名称: {model_name}\n")
#         f.write(f"总参数数量: {total_params:,}\n")
#         f.write(f"可训练参数数量: {trainable_params:,}\n")
#         f.write(f"可训练参数占比: {trainable_params/total_params*100:.2f}%\n")
#         f.write(f"参数量 (MB): {total_params * 4 / 1024 / 1024:.2f} MB\n")  # 假设每个参数4字节
    
#     # 保存详细的层信息（JSON格式）
#     json_path = os.path.join(save_dir, f'{model_name}_layers.json')
#     with open(json_path, 'w', encoding='utf-8') as f:
#         json.dump(layer_info, f, indent=4, ensure_ascii=False)
    
#     # 保存易读的层信息表格（文本格式）
#     txt_path = os.path.join(save_dir, f'{model_name}_layers.txt')
#     with open(txt_path, 'w', encoding='utf-8') as f:
#         # 表头
#         f.write(f"{'层名称':<60} {'层类型':<30} {'输入形状':<20} {'输出形状':<20} {'总参数':<15} {'可训练参数':<15}\n")
#         f.write("-" * 180 + "\n")
        
#         # 写入各层信息
#         for layer in layer_info:
#             name = layer['name']
#             # 过长的名称截断处理
#             if len(name) > 57:
#                 name = name[:57] + "..."
            
#             f.write(
#                 f"{name:<60} {layer['type']:<30} "
#                 f"{str(layer['input_shape']):<20} {str(layer['output_shape']):<20} "
#                 f"{layer['total_params']:<15,} {layer['trainable_params']:<15,}\n"
#             )
    
#     print(f"模型信息已保存至: {os.path.abspath(save_dir)}")

# import json
# def main():
#     """主函数：创建模型、收集信息并保存"""
#     # 可以测试的模型配置
#     model_configs = ['B0']  # 可扩展为['B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6']
#     save_dir = 'hgnetv2_info'
    
#     for config in model_configs:
#         print(f"\n===== 处理模型: HGNetv2-{config} =====")
        
#         # 创建模型
#         model = HGNetv2(
#             name=config,
#             use_lab=False,
#             return_idx=[1, 2, 3],
#             freeze_stem_only=True,
#             freeze_at=0,
#             freeze_norm=True,
#             pretrained=False  # 不加载预训练权重，加快测试速度
#         )
#         model.eval()  # 设置为评估模式
        
#         # 计算参数数量
#         total_params, trainable_params = count_parameters(model)
#         print(f"总参数数量: {total_params:,}")
#         print(f"可训练参数数量: {trainable_params:,}")
#         print(f"可训练参数占比: {trainable_params/total_params*100:.2f}%")
        
#         # 获取层信息
#         print("收集各层信息...")
#         layers = get_layer_info(model)
        
#         # 保存信息
#         save_model_info(f'HGNetv2-{config}', total_params, trainable_params, layers, save_dir)

# if __name__ == "__main__":
#     main()