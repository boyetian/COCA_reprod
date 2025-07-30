# Copyright 2021 Apple Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple, Union, Dict
import math

def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.norm = norm_layer(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, norm_layer=nn.BatchNorm2d):
        super().__init__()
        hidden_dim = make_divisible(in_channels * expand_ratio)
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(ConvLayer(in_channels, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        
        layers.extend([
            ConvLayer(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileViTBlock(nn.Module):
    def __init__(self, in_channels, transformer_dim, ffn_dim, n_transformer_blocks, patch_size, norm_layer=nn.BatchNorm2d):
        super().__init__()
        # Local representation
        self.local_rep = nn.Sequential(
            ConvLayer(in_channels, in_channels, kernel_size=3, norm_layer=norm_layer),
            ConvLayer(in_channels, transformer_dim, kernel_size=1, norm_layer=norm_layer)
        )

        # Global representation
        self.global_rep = nn.Sequential(
            *[Transformer(transformer_dim, ffn_dim, n_head=2) for _ in range(n_transformer_blocks)]
        )
        self.norm = norm_layer(transformer_dim)

        # Fusion
        self.fusion = ConvLayer(transformer_dim, in_channels, kernel_size=1, norm_layer=norm_layer)
        self.patch_size = patch_size

    def forward(self, x):
        local_rep = self.local_rep(x)
        
        # Unfold to patches
        B, C, H, W = local_rep.shape
        P_H, P_W = self.patch_size
        N_H, N_W = H // P_H, W // P_W
        patches = local_rep.unfold(2, P_H, P_H).unfold(3, P_W, P_W)
        patches = patches.reshape(B, C, N_H * N_W, P_H * P_W).transpose(1, 2) # B, N, C, P

        # Global representation
        global_rep = self.global_rep(patches)
        global_rep = global_rep.transpose(1, 2).reshape(B, C, N_H, N_W, P_H, P_W)
        global_rep = global_rep.permute(0, 1, 2, 4, 3, 5).reshape(B, C, H, W)

        # Fusion
        fused = self.fusion(self.norm(global_rep))
        return fused + x

class Transformer(nn.Module):
    def __init__(self, dim, ffn_dim, n_head):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_head, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.SiLU(inplace=True),
            nn.Linear(ffn_dim, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x

class MobileViT(nn.Module):
    def __init__(self, image_size, dims, channels, num_classes=1000, expansion=4, kernel_size=3, patch_size=(2, 2)):
        super().__init__()
        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        self.conv1 = ConvLayer(3, channels[0], kernel_size=3, stride=2)
        self.mv2 = nn.ModuleList([])
        self.mv2.append(InvertedResidual(channels[0], channels[1], 1, expansion))
        self.mv2.append(InvertedResidual(channels[1], channels[2], 2, expansion))
        self.mv2.append(InvertedResidual(channels[2], channels[3], 1, expansion))
        self.mv2.append(InvertedResidual(channels[2], channels[3], 1, expansion))
        self.mv2.append(InvertedResidual(channels[3], channels[4], 2, expansion))
        
        self.mvit = nn.ModuleList([])
        self.mvit.append(MobileViTBlock(channels[4], dims[0], int(dims[0]*2), n_transformer_blocks=2, patch_size=patch_size))
        self.mvit.append(InvertedResidual(channels[4], channels[5], 2, expansion))
        self.mvit.append(MobileViTBlock(channels[5], dims[1], int(dims[1]*4), n_transformer_blocks=4, patch_size=patch_size))
        self.mvit.append(InvertedResidual(channels[5], channels[6], 2, expansion))
        self.mvit.append(MobileViTBlock(channels[6], dims[2], int(dims[2]*4), n_transformer_blocks=3, patch_size=patch_size))

        self.conv2 = ConvLayer(channels[-2], channels[-1], kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.mv2[0](x)
        x = self.mv2[1](x)
        x = self.mv2[2](x)
        x = self.mv2[3](x)
        x = self.mv2[4](x)
        x = self.mvit[0](x)
        x = self.mv2[5](x)
        x = self.mvit[1](x)
        x = self.mv2[6](x)
        x = self.mvit[2](x)
        x = self.conv2(x)
        x = self.pool(x).view(x.shape[0], -1)
        x = self.fc(x)
        return x

def mobilevit_s(pretrained=False, **kwargs):
    dims = [144, 192, 240]
    channels = [16, 32, 64, 64, 96, 128, 160, 640]
    model = MobileViT((256, 256), dims, channels, num_classes=1000, **kwargs)
    if pretrained:
        # There is no official pretrained weight for this architecture in torchvision.
        # Loading weights would require a source, e.g. a URL to a .pt file.
        # For this example, we will not load pretrained weights.
        print("Warning: pretrained weights not available for mobilevit_s, model is initialized randomly.")
    return model
