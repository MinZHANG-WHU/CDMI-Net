# -*- coding: utf-8 -*-
"""
@author: ZHANG Min, Wuhan University
@email: 007zhangmin@whu.edu.cn
"""

__author__ = 'ZHANG Min, Wuhan University'

import torch
import torch.nn as nn
import torch.nn.functional as F

from unet_model import UNet
from model_parts import GatedAttentionLayer


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.im_dim = 112
        self.im_channel = 3
        self.feature_channels = 64
        self.attention_channels = 128
        self.unet = UNet(n_channels=self.im_channel)
        self.attention = GatedAttentionLayer(
            self.feature_channels, self.attention_channels)
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_channels, 1),  # 64*1
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        H1 = self.unet(x1)
        H2 = self.unet(x2)

        DI = torch.abs(H1 - H2)
        H = DI.permute(0, 2, 3, 1)
        H = H.view(-1, self.im_dim * self.im_dim, self.feature_channels)

        A = self.attention(H)  # NxK,1x1, W_t*  tanh(V * H_t)

        A = A.permute(0, 2, 1)
        A = F.softmax(A, dim=2)  # softmax over N

        A_3 = A.view(-1, 1, self.im_dim * self.im_dim)
        H_3 = H.view(-1, self.im_dim * self.im_dim,
                     self.feature_channels)
        M = torch.bmm(A_3, H_3)

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()  # y >= 0.5 ? 1 : 0

        return Y_prob, Y_hat, A, DI

    def eval_img(self, X1, X2):
        Y_prob, Y_hat, A, _ = self.forward(X1, X2)
        Y_prob = Y_prob.cpu().detach().numpy()[0, 0]
        Y_hat = Y_hat.cpu().detach().numpy()[0, 0]
        return Y_prob, Y_hat, A

    def calculate_loss(self, X1, X2, Y):
        Y = Y.float()
        Y_prob, Y_hat, A, DI = self.forward(X1, X2)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_prob = -1. * (Y * torch.log(Y_prob) + (1. - Y)
                              * torch.log(1. - Y_prob))

        error = Y_hat.eq(Y).float().cpu().numpy()[0]  #
        error = 1. - error[0]
        return neg_log_prob, A, error

    def print_size(self):
        num_params = 0
        for name, param in self.named_parameters():
            print(name, '->', param.numel())
            num_params += param.numel()
        print(num_params / 1e6)
