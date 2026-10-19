"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
"""

import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .utils import get_activation

from ..core import register

__all__ = ['HybridEncoder']


class ConvNormLayer_fuse(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size-1)//2 if padding is None else padding
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            groups=g,
            padding=padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)
        self.ch_in, self.ch_out, self.kernel_size, self.stride, self.g, self.padding, self.bias = \
            ch_in, ch_out, kernel_size, stride, g, padding, bias

    def forward(self, x):
        if hasattr(self, 'conv_bn_fused'):
            y = self.conv_bn_fused(x)
        else:
            y = self.norm(self.conv(x))
        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv_bn_fused'):
            self.conv_bn_fused = nn.Conv2d(
                self.ch_in,
                self.ch_out,
                self.kernel_size,
                self.stride,
                groups=self.g,
                padding=self.padding,
                bias=True)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv_bn_fused.weight.data = kernel
        self.conv_bn_fused.bias.data = bias
        self.__delattr__('conv')
        self.__delattr__('norm')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor()

        return kernel3x3, bias3x3

    def _fuse_bn_tensor(self):
        kernel = self.conv.weight
        running_mean = self.norm.running_mean
        running_var = self.norm.running_var
        gamma = self.norm.weight
        beta = self.norm.bias
        eps = self.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size-1)//2 if padding is None else padding
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            groups=g,
            padding=padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


# TODO, add activation for cv1 following YOLOv10
# self.cv1 = Conv(c1, c2, 1, 1)
# self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)
class SCDown(nn.Module):
    def __init__(self, c1, c2, k, s, act=None):
        super().__init__()
        self.cv1 = ConvNormLayer_fuse(c1, c2, 1, 1)
        self.cv2 = ConvNormLayer_fuse(c2, c2, k, s, c2)

    def forward(self, x):
        return self.cv2(self.cv1(x))
from pytorch_wavelets import DWTForward
class SCDownWithAttention1(nn.Module):
    def __init__(self, c1, c2, k, s, act='gelu', reduction_ratio=2, wavelet='haar'):
        super().__init__()
        self.main_branch = SCDown(c1, c2, k, s, act)
        self.dwt = DWTForward(wave='haar', J=1, mode='zero')
        self.bg_scale = nn.Parameter(torch.tensor(1.0))
        # 替换原avg_pool和density_conv为小波高频处理分支
        self.high_freq_conv = nn.Sequential(
            nn.Conv2d(c1*4, c1 // reduction_ratio, 3, 1, 1, groups=c1 // reduction_ratio, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(c1 // reduction_ratio, c2, 1, 1, 0, bias=False)
        )
        self.final_act = get_activation(act)

    def wavelet_decomposition(self, x):

        yl, yh = self.dwt(x)



        HL = yh[0][:, :, 0, ::]
        LH = yh[0][:, :, 1, ::]
        HH = yh[0][:, :, 2, ::]

        return yl, LH, HL, HH

    def forward(self, x):
        main_output = self.main_branch(x)

        # 1. 小波分解提取高频分量
        yl, LH, HL, HH = self.wavelet_decomposition(x)


        dwt=torch.cat([yl, LH, HL, HH],dim=1)
        # 2. 高频融合（绝对值求和）
        # high_freq = torch.abs(LH) + torch.abs(HL) + torch.abs(HH)

        # 3. 高频特征卷积处理
        branch_hf = self.high_freq_conv(dwt)

        # 4. 生成空间注意力图
        b, c, h, w = branch_hf.size()
    # 使用 sigmoid 替代 softmax
        attention_map = torch.sigmoid(branch_hf)

        # 5. 上采样注意力图到主分支尺寸
        if attention_map.size()[2:] != main_output.size()[2:]:
            attention_map = F.interpolate(
                attention_map,
                size=main_output.size()[2:],
                mode='bilinear',
                align_corners=False
            )

        background_attention = torch.sigmoid(self.bg_scale * (1.0 - attention_map))


        attended_output = main_output * attention_map +  main_output * background_attention

        return self.final_act(attended_output)  
# class SCDown1(nn.Module):
#     def __init__(self, c1, c2, k, s, g,act=None):
#         super().__init__()
#         self.cv1 = ConvNormLayer_fuse(c1, c2, 1, 1, g)
#         self.cv2 = ConvNormLayer_fuse(c2, c2, k, s, c2)

#     def forward(self, x):
#         return self.cv2(self.cv1(x))

class HWD(nn.Module):
    def __init__(self, c1, c2, k, s, act):
        super(HWD, self).__init__()
        self.wt = DWTForward(J=1, wave='haar', mode='zero')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(c1 * 4, c2, kernel_size=1, stride=1),
            nn.BatchNorm2d(c2),
            nn.GELU(),
        )
        self.main_branch = SCDown(c1, c2, k, s, act)
    def forward(self, x):
        yL, yH = self.wt(x)
        identity=self.main_branch(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)
        return x+identity

class SCDownWithAttention(nn.Module):
    def __init__(self, c1, c2, k, s, act='gelu', reduction_ratio=2, wavelet='haar'):
        super().__init__()
    
        self.dwt = DWTForward(wave=wavelet, J=1, mode='zero')  # 小波变换

    def wavelet_decomposition(self, x):
        yl, yh = self.dwt(x)  # yl: LL分量（低频）；yh: 高频分量集合
        HL = yh[0][:, :, 0, ::]  # 水平高频（HL）
        LH = yh[0][:, :, 1, ::]  # 垂直高频（LH）
        return yl, LH, HL  # 仅保留LL、LH、HL（HH暂不参与）
    def forward(self, x):
 
        yl, LH, HL = self.wavelet_decomposition(x)  # yl尺寸与x相同，通道数c1
        high_freq = torch.add(LH, HL)  # 垂直+水平高频融合（保留方向信息）
        attention_map = F.softmax(high_freq, dim=1)  # 沿通道维度归一化（假设通道维度为特征维度）
        ll_attended = yl * attention_map  # 用高频注意力增强LL中的关键区域
        ll_enhanced = ll_attended + yl  # 残差融合
        return ll_enhanced 
class VGGBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        self.__delattr__('conv1')
        self.__delattr__('conv2')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)

        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=False,
                 act="silu",
                 bottletype=VGGBlock):
        super(CSPLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            bottletype(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer_fuse(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_2 = self.conv2(x)
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        return self.conv3(x_1 + x_2)
    
from typing import Optional, Sequence
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
            c1, c2, k, s, padding=autopad1(k, p, d), dilation=d, groups=g, bias=False
        )
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))
    
def make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original TensorFlow repo.
    It ensures that all layers have a channel number that is divisible by 8
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    return new_v
    
from typing import Optional, Sequence





# transformer
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
class MBSM(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, dilation=1, bias=False, scale=0.3,map_reduce=2):
        super(MBSM, self).__init__()
        self.scale = scale
        # 通道压缩
        inter_planes = in_planes // map_reduce

        # 分支1: 1x3 + 3x1 + 3x3 DW Conv
        self.branch1_dw1 = BasicConv(in_planes, inter_planes, kernel_size=(1, 3), padding=(0, 1),
                                     groups=inter_planes, bias=bias, bn=True, relu=False)
        self.branch1_pw1 = BasicConv(inter_planes, out_planes, kernel_size=(3, 1), padding=(1, 0),
                                     groups=inter_planes, bias=bias, bn=True, relu=False)
        self.branch1_detail = BasicConv(out_planes, out_planes, kernel_size=3, padding=1,
                                        groups=out_planes, bias=bias, bn=True, relu=True)

        # 分支2: 1x5 + 5x1 + 3x3 DW Conv
        self.branch2_dw2 = BasicConv(in_planes, inter_planes, kernel_size=(1, 5), padding=(0, 2),
                                     groups=inter_planes, bias=bias, bn=True, relu=False)
        self.branch2_pw2 = BasicConv(inter_planes, out_planes, kernel_size=(5, 1), padding=(2, 0),
                                     groups=inter_planes, bias=bias, bn=True, relu=False)
        self.branch2_detail = BasicConv(out_planes, out_planes, kernel_size=3, padding=1,
                                        groups=out_planes, bias=bias, bn=True, relu=True)

        # 分支3: 1x7 + 7x1 + 3x3 DW Conv
        self.branch3_dw3 = BasicConv(in_planes, inter_planes, kernel_size=(1, 7), padding=(0, 3),
                                     groups=inter_planes, bias=bias, bn=True, relu=False)
        self.branch3_pw3 = BasicConv(inter_planes, out_planes, kernel_size=(7, 1), padding=(3, 0),
                                     groups=inter_planes, bias=bias, bn=True, relu=False)
        self.branch3_detail = BasicConv(out_planes, out_planes, kernel_size=3, padding=1,
                                        groups=out_planes, bias=bias, bn=True, relu=True)

        # 分支4: 3x3 DW Conv（局部信息分支）
        self.branch4_detail = BasicConv(out_planes, out_planes, kernel_size=3, padding=1,
                                        groups=out_planes, bias=False, bn=True, relu=False)

        self.fusion_bn = BasicConv(4 * out_planes, out_planes, kernel_size=1, stride=1, groups=8,relu=False)

        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # res = self.residual(x)

        # 分支1
        x1 = self.branch1_dw1(x)
        x1 = self.branch1_pw1(x1)
        x1 = self.branch1_detail(x1)

        # 分支2
        x2 = self.branch2_dw2(x)
        x2 = self.branch2_pw2(x2)
        x2 = self.branch2_detail(x2)

        # 分支3
        x3 = self.branch3_dw3(x)
        x3 = self.branch3_pw3(x3)
        x3 = self.branch3_detail(x3)

        # 分支4
        x4 = self.branch4_detail(x)

        # 拼接融合
        out = torch.cat((x1, x2, x3, x4), dim=1)

        # 融合后的BN + ReLU + 残差
        out = self.fusion_bn(out)
        out = out  + x

        return out
    
class RepNCSPELAN4(nn.Module):
    # csp-elan
    def __init__(self, c1, c2, c3, c4, n=3,
                 bias=False,
                 act="silu"):
        super().__init__()
        self.c = c3//2
        self.cv1 = ConvNormLayer_fuse(c1, c3, 1, 1, bias=bias, act=act)
        self.cv2 = nn.Sequential(CSPLayer(c3//2, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock), ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act))
        self.cv3 = nn.Sequential(CSPLayer(c4, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock), ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act))
        self.cv4 = ConvNormLayer_fuse(c3+(2*c4), c2, 1, 1, bias=bias, act=act)

    def forward_chunk(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

class Conv(nn.Module):
    """标准卷积层：Conv + BN + Activation"""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class PAD(nn.Module):
    """多尺度深度卷积注意力下采样模块"""

    def __init__(self, c1, c2, stride=2, act=True):
        super().__init__()
        # 多尺度特征提取
        self.conv1 = nn.Conv2d(c1, c1, 3, 1, 1, groups=c1, bias=False)
        self.conv2 = Conv(c1, c1, k=1, s=1)
        self.conv3 = nn.Conv2d(c1, c1, 5, 1, 2, groups=c1, bias=False)
        self.conv4 = Conv(c1, c1, 1, 1)
        self.conv5 = nn.Conv2d(c1, c1, 7, 1, 3, groups=c1, bias=False)

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Conv2d(c1, c1, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

        # 深度可分离卷积下采样
        self.dw= Conv(c1, c1, k=3, s=2, g=c1)


    def forward(self, x):
        # 多尺度特征提取
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        # 生成注意力图
        attn_map = self.attention(x5)

        # 应用注意力并与原始输入相加
        x_attended = x * attn_map + x  # 注意力增强

        # 深度可分离卷积下采样
        x_down = self.dw(x_attended)

        return x_down
def autopad1(k, p=None, d=1):
    """自动填充以保持输出形状相同"""
    if d > 1:
        # 实际的卷积核大小
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        # 自动填充
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p
class Conv_MS(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad1(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
class ConvEdgeFusion(nn.Module):
    def __init__(self, inc, ouc) -> None:
        super().__init__()
        
        self.conv_channel_fusion = Conv_MS(sum(inc), ouc // 2, k = 1)
        self.conv_3x3_feature_extract = Conv_MS(ouc // 2, ouc // 2, 3,g=ouc // 2)
        self.conv_1x1 = Conv_MS(ouc // 2, ouc, 1)
    
    def forward(self, x):
        x = torch.cat(x, dim=1)
        x = self.conv_1x1(self.conv_3x3_feature_extract(self.conv_channel_fusion(x)))
        return x 
    

class BGELAN4(nn.Module):
    def __init__(self, c1, c2, c3, c4, n=3, bias=False, act="silu"):
        super().__init__()
        self.c = c3 // 2
        self.cv1 = ConvNormLayer_fuse(c1, c3, 1, 1, bias=bias, act=act)
        self.cv2 = nn.Sequential(
            CSPLayer(c3 // 2, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock),
            ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act)
        )
        self.cv3 = nn.Sequential(
            CSPLayer(c4, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock),
            ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act)
        )
        self.cv4 = ConvNormLayer_fuse(c3 // 2 + 2 * c4, c2, 1, 1, bias=bias, act=act)
        self.norm = nn.BatchNorm2d(c2)
        # 背景增强模块
        self.foreground_attention = nn.Sequential(
            nn.Conv2d(2 * c4, 1, kernel_size=1),  # 使用1x1卷积减少通道数到1
            nn.Sigmoid()  # 输出为前景注意力图 [B,1,H,W]
        )
        self.local_detail = nn.Sequential(
            nn.Conv2d(c1, c2, 1, 1, bias=bias),
            nn.BatchNorm2d(c2),
            nn.SiLU() if act == "silu" else nn.ReLU()
        )
        # self.bg_refinement = nn.Sequential(
        #      nn.Conv2d(c3//2, c3//2, 3, padding=1, groups=c3//2, bias=False),
        #      nn.Conv2d(c3//2, c3//2, 1, bias=False),
        #      nn.BatchNorm2d(c3//2),
        #      nn.SiLU()
        #          )
        # 可学习背景增强权重
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # 初始卷积并拆分
        # x = channel_shuffle(x, groups=4)
        y = list(self.cv1(x).split((self.c, self.c), 1))  # [y0, y1], 各c3//2通道

        # 对y[1]分支保持原始操作
        conv_out1 = self.cv2(y[1])  # [B, c4, H, W]
        conv_out2 = self.cv3(conv_out1)  # [B, c4, H, W]

        # 拼接conv_out1和conv_out2作为前景特征
        foreground_features = torch.cat([conv_out1, conv_out2], dim=1)  # [B, 2*c4, H, W]

        # 生成前景注意力图
        foreground_atten = self.foreground_attention(foreground_features)  # [B,1,H,W]

        # 反转得到背景注意力图
        background_atten = 1 - foreground_atten

        # 应用可学习的背景增强权重
        bg_weight = torch.clamp(self.alpha, 0.5, 1.5)
        beta = torch.sigmoid(self.beta)
        # 对背景区域进行增强
        enhanced_background = y[0] * background_atten * bg_weight
        # print(bg_weight)

        merged = torch.cat([enhanced_background, conv_out1, conv_out2], dim=1)
        local_feat = self.local_detail(x)

        # print(beta)
        out = self.norm(self.cv4(merged) + beta * local_feat)
        return out
    
#下面这个采用了多尺度的AIFI
def autopadSM(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
class ConvSM(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopadSM(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
class MutilScal(nn.Module):
    def __init__(self, dim=512, fc_ratio=4, dilation=[3, 5, 7], pool_ratio=16):
        super(MutilScal, self).__init__()
        self.conv0_1 = ConvSM(dim, dim//fc_ratio)
        self.conv0_2 = ConvSM(dim//fc_ratio, dim//fc_ratio, 3, d=dilation[-3], g=dim//fc_ratio)
        self.conv0_3 = ConvSM(dim//fc_ratio, dim, 1)

        self.conv1_2 = ConvSM(dim//fc_ratio, dim//fc_ratio, 3, d=dilation[-2], g=dim // fc_ratio)
        self.conv1_3 = ConvSM(dim//fc_ratio, dim, 1)

        self.conv2_2 = ConvSM(dim//fc_ratio, dim//fc_ratio, 3, d=dilation[-1], g=dim//fc_ratio)
        self.conv2_3 = ConvSM(dim//fc_ratio, dim, 1)

        self.conv3 = ConvSM(dim, dim, 1)

        self.Avg = nn.AdaptiveAvgPool2d(pool_ratio)

    def forward(self, x):
        u = x.clone()

        attn0_1 = self.conv0_1(x)
        attn0_2 = self.conv0_2(attn0_1)
        attn0_3 = self.conv0_3(attn0_2)

        attn1_2 = self.conv1_2(attn0_1)
        attn1_3 = self.conv1_3(attn1_2)

        attn2_2 = self.conv2_2(attn0_1)
        attn2_3 = self.conv2_3(attn2_2)

        attn = attn0_3 + attn1_3 + attn2_3
        attn = self.conv3(attn)
        attn = attn * u

        pool = self.Avg(attn)

        return pool

class Mutilscal_MHSA(nn.Module):
    def __init__(self, dim, num_heads=8, atten_drop = 0., proj_drop = 0., dilation = [3, 5, 7], fc_ratio=4, pool_ratio=16):
        super(Mutilscal_MHSA, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.atten_drop = nn.Dropout(atten_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.MSC = MutilScal(dim=dim, fc_ratio=fc_ratio, dilation=dilation, pool_ratio=pool_ratio)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim//fc_ratio, kernel_size=1),
            nn.ReLU6(),
            nn.Conv2d(in_channels=dim//fc_ratio, out_channels=dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.kv = ConvSM(dim, 2 * dim, 1)

    def forward(self, x):
        u = x.clone()
        B, C, H, W = x.shape
        kv = self.MSC(x)
        kv = self.kv(kv)

        B1, C1, H1, W1 = kv.shape

        q = rearrange(x, 'b (h d) (hh) (ww) -> (b) h (hh ww) d', h=self.num_heads,
                      d=C // self.num_heads, hh=H, ww=W)
        k, v = rearrange(kv, 'b (kv h d) (hh) (ww) -> kv (b) h (hh ww) d', h=self.num_heads,
                         d=C // self.num_heads, hh=H1, ww=W1, kv=2)

        dots = (q @ k.transpose(-2, -1)) * self.scale
        attn = dots.softmax(dim=-1)
        attn = self.atten_drop(attn)
        attn = attn @ v

        attn = rearrange(attn, '(b) h (hh ww) d -> b (h d) (hh) (ww)', h=self.num_heads,
                         d=C // self.num_heads, hh=H, ww=W)
        c_attn = self.avgpool(x)
        c_attn = self.fc(c_attn)
        c_attn = c_attn * u
        return attn + c_attn
class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
class TransformerEncoderLayer_MSMHSA(nn.Module):

    """Defines a single layer of the transformer encoder."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0, act=nn.GELU(), normalize_before=False):
        """Initialize the TransformerEncoderLayer with specified parameters."""
        super().__init__()
        self.additivetoken = Mutilscal_MHSA(c1)
        # Implementation of Feedforward model
        self.fc1 = nn.Conv2d(c1, cm, 1)
        self.fc2 = nn.Conv2d(cm, c1, 1)

        self.norm1 = LayerNorm(c1)
        self.norm2 = LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.act = act
        self.normalize_before = normalize_before

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Performs forward pass with post-normalization."""
        src2 = self.additivetoken(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Forward propagates the input through the encoder module."""
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)
    


# mona

import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d_mona(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class MonaOp(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=3 // 2, groups=in_features)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=5, padding=5 // 2, groups=in_features)
        self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=7, padding=7 // 2, groups=in_features)

        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1, )

    def forward(self, x):
        identity = x
        conv1_x = self.conv1(x)
        conv2_x = self.conv2(x)
        conv3_x = self.conv3(x)

        x = (conv1_x + conv2_x + conv3_x) / 3.0 + identity

        identity = x

        x = self.projector(x)

        return identity + x

class Mona(nn.Module):
    def __init__(self,
                 in_dim):
        super().__init__()

        self.project1 = nn.Conv2d(in_dim, 64, 1)
        self.nonlinear = F.gelu
        self.project2 = nn.Conv2d(64, in_dim, 1)

        self.dropout = nn.Dropout(p=0.1)

        self.adapter_conv = MonaOp(64)

        self.norm = LayerNorm2d_mona(in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim, 1, 1) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim, 1, 1))

    def forward(self, x, hw_shapes=None):
        identity = x

        x = self.norm(x) * self.gamma + x * self.gammax

        project1 = self.project1(x)

        project1 = self.adapter_conv(project1)

        nonlinear = self.nonlinear(project1)
        nonlinear = self.dropout(nonlinear)
        project2 = self.project2(nonlinear)

        return identity + project2
class TransformerEncoderLayer_Mona(nn.Module):
    """Defines a single layer of the transformer encoder."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0, act=nn.GELU(), normalize_before=False):
        """Initialize the TransformerEncoderLayer with specified parameters."""
        super().__init__()
        # from ...utils.torch_utils import TORCH_1_9
        # if not TORCH_1_9:
        #     raise ModuleNotFoundError(
        #         'TransformerEncoderLayer() requires torch>=1.9 to use nn.MultiheadAttention(batch_first=True).')
        self.ma = nn.MultiheadAttention(c1, num_heads, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)

        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.mona1 = Mona(c1)
        self.mona2 = Mona(c1)

        self.act = act
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor, pos=None):
        """Add position embeddings to the tensor if provided."""
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Performs forward pass with post-normalization."""
        BS, C, H, W = src.size()
        src = src.flatten(2).permute(0, 2, 1)
        q = k = self.with_pos_embed(src, pos)
        src2 = self.ma(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = self.mona1(src.permute(0, 2, 1).view([-1, C, H, W]).contiguous()).flatten(2).permute(0, 2, 1)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
        src = src + self.dropout2(src2)
        return self.mona2(self.norm2(src).permute(0, 2, 1).view([-1, C, H, W]).contiguous()).flatten(2).permute(0, 2, 1)

    def forward_pre(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Performs forward pass with pre-normalization."""
        BS, C, H, W = src.size()
        src = src.flatten(2).permute(0, 2, 1)
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.ma(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.mona1(src.permute(0, 2, 1).view([-1, C, H, W]).contiguous()).flatten(2).permute(0, 2, 1)
        src2 = self.norm2(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src2))))
        return self.mona2((src + self.dropout2(src2)).permute(0, 2, 1).view([-1, C, H, W]).contiguous()).flatten(2).permute(0, 2, 1)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Forward propagates the input through the encoder module."""
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class AIFI_Mona(TransformerEncoderLayer_Mona):
    """Defines the AIFI transformer layer."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0, act=nn.GELU(), normalize_before=False):
        """Initialize the AIFI instance with specified parameters."""
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)

    def forward(self, x):
        """Forward pass for the AIFI transformer layer."""
        c, h, w = x.shape[1:]
        pos_embed = self.build_2d_sincos_position_embedding(w, h, c)
        # Flatten [B, C, H, W] to [B, HxW, C]
        x = super().forward(x, pos=pos_embed.to(device=x.device, dtype=x.dtype))
        return x.permute(0, 2, 1).view([-1, c, h, w]).contiguous()

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
        """Builds 2D sine-cosine position embedding."""
        assert embed_dim % 4 == 0, "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], 1)[None]
    

# CEFFM,两个特征图融合
class ConcatFeatureFusion(nn.Module):
    def __init__(self, in_ch, out_ch, branch_1=nn.Identity(), dilate=1):
        super(ConcatFeatureFusion, self).__init__()
        self.branch_1 = branch_1

        self.dconv1 = Conv_MS(in_ch, in_ch, 3, d=dilate) if dilate else nn.Identity()
        self.fuseconv = Conv_MS(in_ch*2, out_ch, 1)

    def forward(self, x):
        x1 = self.branch_1(x)
        x2 = self.dconv1(x)
        result = self.fuseconv(torch.concat([x1, x2], dim=1))
        return result

class TwoBranchFeatureFusion(nn.Module):
    def __init__(self, inplanes, r=2):
        super(TwoBranchFeatureFusion, self).__init__()
        planes = inplanes // r
        self.branch_1 = nn.Sequential(
            Conv_MS(inplanes, planes, 1),
            Conv_MS(planes, planes, 3),
            Conv_MS(planes, inplanes, 1, act=False)
        )

        self.branch_2 = ConcatFeatureFusion(inplanes, inplanes, branch_1=nn.Sequential(Conv_MS(inplanes, planes, 1),
                                            ParallelAtrousConv6(planes, ratio=[1, 3, 5]),
                                            Conv_MS(planes, inplanes, 1, act=False)), dilate=None)

        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x1, x2 = inputs
        xx = x1 + x2
        xl = self.branch_1(xx)
        xg = self.branch_2(xx)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = x1 * wei + x2 * (1 - wei)
        return xo
    
class ParallelAtrousConv6(nn.Module):
    def __init__(self, inc, ratio=[1, 3, 5, 7]) -> None:
        super().__init__()
        cc = inc//2
        self.rlength = len(ratio)
        self.conv1x1List = nn.ModuleList()
        self.concatFeatureFusionList = nn.ModuleList()

        for idx in range(self.rlength):
            self.conv1x1List.append(Conv_MS(cc, cc, k=1, s=1))
            self.concatFeatureFusionList.append(ConcatFeatureFusion(cc, cc, branch_1=nn.Sequential(Conv_MS(cc, cc, 1)), dilate=ratio[idx]))
        
        self.conv1 = Conv_MS(inc, cc, k=1)
        self.convfuse = Conv_MS(self.rlength*cc, inc, k=1)

    def forward(self, x):
        xx = self.conv1(x)
        ratioResultList = []
        for index in range(self.rlength):  # 按执行顺序遍历网络各层操作
            ratioResultList.append(self.conv1x1List[index](self.concatFeatureFusionList[index](xx)))

        result = self.convfuse(torch.cat(ratioResultList, dim=1))
        return result
@register()
class HybridEncoder(nn.Module):
    __share__ = ['eval_spatial_size', ]

    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 in_channels_tmp=[64,256, 512, 1024],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward = 1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None,
                 version='dfine',
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides



        self.mbsm_blocks = nn.ModuleList([
            MBSM(in_planes=256, out_planes=256, map_reduce=2)
            for _ in in_channels
        ])

        #用于CEFFM
        self.down160=SCDown(hidden_dim, hidden_dim, 3, 2, act=act)
        self.fusion_module = TwoBranchFeatureFusion(inplanes=hidden_dim, r=2) 

        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels_tmp:
            proj = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
                    ('norm', nn.BatchNorm2d(hidden_dim))
                ]))

            self.input_proj.append(proj)

        # encoder transformer
        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_act
            )

        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
        ])
                # 核心修改：用改进的TransformerEncoderLayer_MSMHSA替换原有编码器层
        # 适配激活函数（enc_act映射为nn模块）
        # act_module = nn.GELU() 
        
        # 定义改进的编码器层
        # self.encoder = TransformerEncoderLayer_MSMHSA(
        #     c1=hidden_dim,
        #     cm=dim_feedforward,
        #     num_heads=nhead,
        #     dropout=dropout,
        #     act=act_module,
        #     normalize_before=False)
        # self.encoder = AIFI_Mona(
        #         c1=hidden_dim,
        #          cm=dim_feedforward,
        #         num_heads=nhead,
        #         dropout=dropout,
        #         act=act_module,
        #         normalize_before=False)
        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            # TODO, add activation for those lateral convs
            if version == 'dfine':
                self.lateral_convs.append(ConvNormLayer_fuse(hidden_dim, hidden_dim, 1, 1))
            else:
                self.lateral_convs.append(ConvNormLayer_fuse(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                RepNCSPELAN4(hidden_dim * 2, hidden_dim, hidden_dim * 2, round(expansion * hidden_dim // 2), round(3 * depth_mult), act=act) \
                if version == 'dfine' else CSPLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion, bottletype=VGGBlock)
            )

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                nn.Sequential(SCDown(hidden_dim, hidden_dim, 3, 2, act=act)) \
                if version == 'dfine' else ConvNormLayer_fuse(hidden_dim, hidden_dim, 3, 2, act=act)
            )

            self.pan_blocks.append(
                RepNCSPELAN4(hidden_dim * 2, hidden_dim, hidden_dim * 2, round(expansion * hidden_dim // 2), round(3 * depth_mult), act=act) \
                if version == 'dfine' else CSPLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion, bottletype=VGGBlock)
            )

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)
                # self.register_buffer(f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        """
        """
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats):
        # assert len(feats) == len(self.in_channels)
        # proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # proj_feats = [self.mbsm_blocks[i](proj) for i, proj in enumerate(proj_feats)]
         
        
        
        # 1. 提取160尺寸的特征图到临时变量
        # temp_160_feat = feats[0]  # 保存160的特征图
        # filtered_feats = feats[1:4]  # 此时filtered_feats仅包含[80, 40, 20]

        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        temp_160_feat = proj_feats[0] 
        filtered_feats = proj_feats[1:4]
        proj_feats = [self.mbsm_blocks[i](proj) for i, proj in enumerate(filtered_feats)]
        feat_80 = proj_feats[0]

        proj_feats[0]=self.fusion_module((feat_80,self.down160(temp_160_feat)))

        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)

                memory :torch.Tensor = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()
               


        #下面的是mona代码
        # if self.num_encoder_layers > 0:
        #     for i, enc_ind in enumerate(self.use_encoder_idx):
        #         memory: torch.Tensor = self.encoder(proj_feats[enc_ind])
        #         # 还原特征形状 [B, HxW, C] -> [B, C, H, W]
        #         proj_feats[enc_ind] = memory

        # broadcasting and fusion
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='nearest')
            inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_height], dim=1))
            outs.append(out)


        
        return  outs 


"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
"""

# import copy
# from collections import OrderedDict

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from .utils import get_activation

# from ..core import register

# __all__ = ['HybridEncoder']


# class ConvNormLayer_fuse(nn.Module):
#     def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
#         super().__init__()
#         padding = (kernel_size-1)//2 if padding is None else padding
#         self.conv = nn.Conv2d(
#             ch_in,
#             ch_out,
#             kernel_size,
#             stride,
#             groups=g,
#             padding=padding,
#             bias=bias)
#         self.norm = nn.BatchNorm2d(ch_out)
#         self.act = nn.Identity() if act is None else get_activation(act)
#         self.ch_in, self.ch_out, self.kernel_size, self.stride, self.g, self.padding, self.bias = \
#             ch_in, ch_out, kernel_size, stride, g, padding, bias

#     def forward(self, x):
#         if hasattr(self, 'conv_bn_fused'):
#             y = self.conv_bn_fused(x)
#         else:
#             y = self.norm(self.conv(x))
#         return self.act(y)

#     def convert_to_deploy(self):
#         if not hasattr(self, 'conv_bn_fused'):
#             self.conv_bn_fused = nn.Conv2d(
#                 self.ch_in,
#                 self.ch_out,
#                 self.kernel_size,
#                 self.stride,
#                 groups=self.g,
#                 padding=self.padding,
#                 bias=True)

#         kernel, bias = self.get_equivalent_kernel_bias()
#         self.conv_bn_fused.weight.data = kernel
#         self.conv_bn_fused.bias.data = bias
#         self.__delattr__('conv')
#         self.__delattr__('norm')

#     def get_equivalent_kernel_bias(self):
#         kernel3x3, bias3x3 = self._fuse_bn_tensor()

#         return kernel3x3, bias3x3

#     def _fuse_bn_tensor(self):
#         kernel = self.conv.weight
#         running_mean = self.norm.running_mean
#         running_var = self.norm.running_var
#         gamma = self.norm.weight
#         beta = self.norm.bias
#         eps = self.norm.eps
#         std = (running_var + eps).sqrt()
#         t = (gamma / std).reshape(-1, 1, 1, 1)
#         return kernel * t, beta - running_mean * gamma / std


# class ConvNormLayer(nn.Module):
#     def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
#         super().__init__()
#         padding = (kernel_size-1)//2 if padding is None else padding
#         self.conv = nn.Conv2d(
#             ch_in,
#             ch_out,
#             kernel_size,
#             stride,
#             groups=g,
#             padding=padding,
#             bias=bias)
#         self.norm = nn.BatchNorm2d(ch_out)
#         self.act = nn.Identity() if act is None else get_activation(act)

#     def forward(self, x):
#         return self.act(self.norm(self.conv(x)))


# # TODO, add activation for cv1 following YOLOv10
# # self.cv1 = Conv(c1, c2, 1, 1)
# # self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)
# class SCDown(nn.Module):
#     def __init__(self, c1, c2, k, s, act=None):
#         super().__init__()
#         self.cv1 = ConvNormLayer_fuse(c1, c2, 1, 1)
#         self.cv2 = ConvNormLayer_fuse(c2, c2, k, s, c2)

#     def forward(self, x):
#         return self.cv2(self.cv1(x))


# class VGGBlock(nn.Module):
#     def __init__(self, ch_in, ch_out, act='relu'):
#         super().__init__()
#         self.ch_in = ch_in
#         self.ch_out = ch_out
#         self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
#         self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
#         self.act = nn.Identity() if act is None else get_activation(act)

#     def forward(self, x):
#         if hasattr(self, 'conv'):
#             y = self.conv(x)
#         else:
#             y = self.conv1(x) + self.conv2(x)

#         return self.act(y)

#     def convert_to_deploy(self):
#         if not hasattr(self, 'conv'):
#             self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

#         kernel, bias = self.get_equivalent_kernel_bias()
#         self.conv.weight.data = kernel
#         self.conv.bias.data = bias
#         self.__delattr__('conv1')
#         self.__delattr__('conv2')

#     def get_equivalent_kernel_bias(self):
#         kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
#         kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)

#         return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

#     def _pad_1x1_to_3x3_tensor(self, kernel1x1):
#         if kernel1x1 is None:
#             return 0
#         else:
#             return F.pad(kernel1x1, [1, 1, 1, 1])

#     def _fuse_bn_tensor(self, branch: ConvNormLayer):
#         if branch is None:
#             return 0, 0
#         kernel = branch.conv.weight
#         running_mean = branch.norm.running_mean
#         running_var = branch.norm.running_var
#         gamma = branch.norm.weight
#         beta = branch.norm.bias
#         eps = branch.norm.eps
#         std = (running_var + eps).sqrt()
#         t = (gamma / std).reshape(-1, 1, 1, 1)
#         return kernel * t, beta - running_mean * gamma / std


# class CSPLayer(nn.Module):
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  num_blocks=3,
#                  expansion=1.0,
#                  bias=False,
#                  act="silu",
#                  bottletype=VGGBlock):
#         super(CSPLayer, self).__init__()
#         hidden_channels = int(out_channels * expansion)
#         self.conv1 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
#         self.conv2 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
#         self.bottlenecks = nn.Sequential(*[
#             bottletype(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
#         ])
#         if hidden_channels != out_channels:
#             self.conv3 = ConvNormLayer_fuse(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
#         else:
#             self.conv3 = nn.Identity()

#     def forward(self, x):
#         x_2 = self.conv2(x)
#         x_1 = self.conv1(x)
#         x_1 = self.bottlenecks(x_1)
#         return self.conv3(x_1 + x_2)

# class RepNCSPELAN4(nn.Module):
#     # csp-elan
#     def __init__(self, c1, c2, c3, c4, n=3,
#                  bias=False,
#                  act="silu"):
#         super().__init__()
#         self.c = c3//2
#         self.cv1 = ConvNormLayer_fuse(c1, c3, 1, 1, bias=bias, act=act)
#         self.cv2 = nn.Sequential(CSPLayer(c3//2, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock), ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act))
#         self.cv3 = nn.Sequential(CSPLayer(c4, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock), ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act))
#         self.cv4 = ConvNormLayer_fuse(c3+(2*c4), c2, 1, 1, bias=bias, act=act)

#     def forward_chunk(self, x):
#         y = list(self.cv1(x).chunk(2, 1))
#         y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
#         return self.cv4(torch.cat(y, 1))

#     def forward(self, x):
#         y = list(self.cv1(x).split((self.c, self.c), 1))
#         y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
#         return self.cv4(torch.cat(y, 1))


# # transformer
# class TransformerEncoderLayer(nn.Module):
#     def __init__(self,
#                  d_model,
#                  nhead,
#                  dim_feedforward=2048,
#                  dropout=0.1,
#                  activation="relu",
#                  normalize_before=False):
#         super().__init__()
#         self.normalize_before = normalize_before

#         self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)

#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.activation = get_activation(activation)

#     @staticmethod
#     def with_pos_embed(tensor, pos_embed):
#         return tensor if pos_embed is None else tensor + pos_embed

#     def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
#         residual = src
#         if self.normalize_before:
#             src = self.norm1(src)
#         q = k = self.with_pos_embed(src, pos_embed)
#         src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

#         src = residual + self.dropout1(src)
#         if not self.normalize_before:
#             src = self.norm1(src)

#         residual = src
#         if self.normalize_before:
#             src = self.norm2(src)
#         src = self.linear2(self.dropout(self.activation(self.linear1(src))))
#         src = residual + self.dropout2(src)
#         if not self.normalize_before:
#             src = self.norm2(src)
#         return src


# class TransformerEncoder(nn.Module):
#     def __init__(self, encoder_layer, num_layers, norm=None):
#         super(TransformerEncoder, self).__init__()
#         self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
#         self.num_layers = num_layers
#         self.norm = norm

#     def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
#         output = src
#         for layer in self.layers:
#             output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

#         if self.norm is not None:
#             output = self.norm(output)

#         return output


# @register()
# class HybridEncoder(nn.Module):
#     __share__ = ['eval_spatial_size', ]

#     def __init__(self,
#                  in_channels=[512, 1024, 2048],
#                  feat_strides=[8, 16, 32],
#                  hidden_dim=256,
#                  nhead=8,
#                  dim_feedforward = 1024,
#                  dropout=0.0,
#                  enc_act='gelu',
#                  use_encoder_idx=[2],
#                  num_encoder_layers=1,
#                  pe_temperature=10000,
#                  expansion=1.0,
#                  depth_mult=1.0,
#                  act='silu',
#                  eval_spatial_size=None,
#                  version='dfine',
#                  ):
#         super().__init__()
#         self.in_channels = in_channels
#         self.feat_strides = feat_strides
#         self.hidden_dim = hidden_dim
#         self.use_encoder_idx = use_encoder_idx
#         self.num_encoder_layers = num_encoder_layers
#         self.pe_temperature = pe_temperature
#         self.eval_spatial_size = eval_spatial_size
#         self.out_channels = [hidden_dim for _ in range(len(in_channels))]
#         self.out_strides = feat_strides

#         # channel projection
#         self.input_proj = nn.ModuleList()
#         for in_channel in in_channels:
#             proj = nn.Sequential(OrderedDict([
#                     ('conv', nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
#                     ('norm', nn.BatchNorm2d(hidden_dim))
#                 ]))

#             self.input_proj.append(proj)

#         # encoder transformer
#         encoder_layer = TransformerEncoderLayer(
#             hidden_dim,
#             nhead=nhead,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             activation=enc_act
#             )

#         self.encoder = nn.ModuleList([
#             TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
#         ])

#         # top-down fpn
#         self.lateral_convs = nn.ModuleList()
#         self.fpn_blocks = nn.ModuleList()
#         for _ in range(len(in_channels) - 1, 0, -1):
#             # TODO, add activation for those lateral convs
#             if version == 'dfine':
#                 self.lateral_convs.append(ConvNormLayer_fuse(hidden_dim, hidden_dim, 1, 1))
#             else:
#                 self.lateral_convs.append(ConvNormLayer_fuse(hidden_dim, hidden_dim, 1, 1, act=act))
#             self.fpn_blocks.append(
#                 RepNCSPELAN4(hidden_dim * 2, hidden_dim, hidden_dim * 2, round(expansion * hidden_dim // 2), round(3 * depth_mult), act=act) \
#                 if version == 'dfine' else CSPLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion, bottletype=VGGBlock)
#             )

#         # bottom-up pan
#         self.downsample_convs = nn.ModuleList()
#         self.pan_blocks = nn.ModuleList()
#         for _ in range(len(in_channels) - 1):
#             self.downsample_convs.append(
#                 nn.Sequential(SCDown(hidden_dim, hidden_dim, 3, 2, act=act)) \
#                 if version == 'dfine' else ConvNormLayer_fuse(hidden_dim, hidden_dim, 3, 2, act=act)
#             )
#             self.pan_blocks.append(
#                 RepNCSPELAN4(hidden_dim * 2, hidden_dim, hidden_dim * 2, round(expansion * hidden_dim // 2), round(3 * depth_mult), act=act) \
#                 if version == 'dfine' else CSPLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion, bottletype=VGGBlock)
#             )

#         self._reset_parameters()

#     def _reset_parameters(self):
#         if self.eval_spatial_size:
#             for idx in self.use_encoder_idx:
#                 stride = self.feat_strides[idx]
#                 pos_embed = self.build_2d_sincos_position_embedding(
#                     self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
#                     self.hidden_dim, self.pe_temperature)
#                 setattr(self, f'pos_embed{idx}', pos_embed)
#                 # self.register_buffer(f'pos_embed{idx}', pos_embed)

#     @staticmethod
#     def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
#         """
#         """
#         grid_w = torch.arange(int(w), dtype=torch.float32)
#         grid_h = torch.arange(int(h), dtype=torch.float32)
#         grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
#         assert embed_dim % 4 == 0, \
#             'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
#         pos_dim = embed_dim // 4
#         omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
#         omega = 1. / (temperature ** omega)

#         out_w = grid_w.flatten()[..., None] @ omega[None]
#         out_h = grid_h.flatten()[..., None] @ omega[None]

#         return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

#     def forward(self, feats):
#         assert len(feats) == len(self.in_channels)
#         proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

#         # encoder
#         if self.num_encoder_layers > 0:
#             for i, enc_ind in enumerate(self.use_encoder_idx):
#                 h, w = proj_feats[enc_ind].shape[2:]
#                 # flatten [B, C, H, W] to [B, HxW, C]
#                 src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
#                 if self.training or self.eval_spatial_size is None:
#                     pos_embed = self.build_2d_sincos_position_embedding(
#                         w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
#                 else:
#                     pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)

#                 memory :torch.Tensor = self.encoder[i](src_flatten, pos_embed=pos_embed)
#                 proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

#         # broadcasting and fusion
#         inner_outs = [proj_feats[-1]]
#         for idx in range(len(self.in_channels) - 1, 0, -1):
#             feat_heigh = inner_outs[0]
#             feat_low = proj_feats[idx - 1]
#             feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
#             inner_outs[0] = feat_heigh
#             upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='nearest')
#             inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](torch.concat([upsample_feat, feat_low], dim=1))
#             inner_outs.insert(0, inner_out)

#         outs = [inner_outs[0]]
#         for idx in range(len(self.in_channels) - 1):
#             feat_low = outs[-1]
#             feat_height = inner_outs[idx + 1]
#             downsample_feat = self.downsample_convs[idx](feat_low)
#             out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_height], dim=1))
#             outs.append(out)

#         return outs