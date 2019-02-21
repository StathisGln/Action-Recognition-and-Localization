import os
import numpy as np
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from action_net import ACT_net

class Model(nn.Module):
    """ 
    action localizatio network which contains:
    -ACT_net : a network for proposing action tubes for 16 frames
    -TCN net : a dilation network which classifies the input tubes
    """
    def __init__(self, actions):
        super(Model, self).__init__()

        self.classes = actions
        self.n_classes = len(actions)

        self.act_net = ACT_net(actions)

        ### tcn options
        ## TODO optimize them

        input_channels = 512
        nhid = 25
        levels = 8
        channel_sizes = [nhid] * levels
        kernel_size = 7
        dropout = 0.05

        self.tcn_net = TCN(input_channels, self.n_classes, channel_sizes, kernel_size = kernel_size, dropout=dropout)

    def forward(input_video, gt_tubes, gt_rois, im_info, num_boxes):

        print('input_video.shape :',input_video.shape)
        print('im_info :',im_info)
        print('num_boxes :',num_boxes)
