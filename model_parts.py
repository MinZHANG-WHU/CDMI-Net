# -*- coding: utf-8 -*-
"""
@author: ZHANG Min, Wuhan University
@email: 007zhangmin@whu.edu.cn
"""

import torch
import torch.nn as nn


class GatedAttentionLayer(nn.Module):
    def __init__(self, in_channels, attention_channels):
        super().__init__()

        self.att = nn.Sequential(
            nn.Linear(in_channels, attention_channels),  # V
            nn.Tanh(),  # tanh(V * H_t)
            nn.Linear(attention_channels, 1)
        )

        self.gate = nn.Sequential(
            nn.Linear(in_channels, attention_channels),  # U
            nn.Sigmoid()  # sigm(U * H_t)
        )

        # W_t * [tanh(V * H_t) * sigm(U * H_t)]
        self.w_t = nn.Linear(attention_channels, 1)

    def forward(self, x):
        a1 = self.att(x)
        a2 = self.gate(x)
        a3 = torch.mul(a1, a2)
        return self.w_t(a3)
