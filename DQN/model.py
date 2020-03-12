#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


import torch.nn as nn
from torch.nn import init


def weight_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class DQN(nn.Module):
    def __init__(self, in_ch, n_action):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 8, 4), # [32, 20, 20]
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), # [64, 9, 9]
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), # [64, 7, 7]
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(512, n_action)
        self.apply(weight_init_orthogonal)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1) # Flatten
        x = self.fc1(x)
        q_val = self.fc2(x)
        return q_val