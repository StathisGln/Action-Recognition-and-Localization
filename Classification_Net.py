import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from tcn import TCN

class Cls_Net(nn.Module):

    """
    This net gets as input base feats and rois
    and classifies them
    """

    def __init__(self, actions):
        super(Cls_Net, self).__init__()

        self.classes = actions
        self.n_classes = len(actions)
        
        ## general options
        self.sample_duration = 16
        # self.step = int(self.sample_duration/2)
        self.step = 8

        input_channels = 512
        nhid = 25
        levels = 8
        channel_sizes = [nhid] * levels
        kernel_size = 7
        dropout = 0.05

        self.tcn_net = TCN(input_channels, self.n_classes, channel_sizes, kernel_size = kernel_size, dropout=dropout)
        self.base_net

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        truncated = False
        normal_init(self.act_rpn.RPN_Conv, 0, 0.01, truncated)
        normal_init(self.act_rpn.RPN_cls_score, 0, 0.01, truncated)
        normal_init(self.act_rpn.RPN_bbox_pred, 0, 0.01, truncated)
        normal_init(self.act_bbox_pred, 0, 0.001, truncated)
        normal_init(self.act_bbox_single_frame_pred, 0, 0.001, truncated)
        # normal_init(self.act_cls_score, 0, 0.001, truncated)

    def forward(self, input_video, im_info,  tubes):

        batch_size = im_data.size(0)
        im_info = im_info.data

        base_feat = self.act_base(input_video)
        print('base_feat.shape :',base_feat.shape)


    def _init_modules(self):

        resnet_shortcut = 'A'
        sample_size = 112
        self.sample_duration = 16  # len(images)

        model = resnet34(num_classes=400, shortcut_type=resnet_shortcut,
                         sample_size=sample_size, sample_duration=self.sample_duration,
                         last_fc=False)
        model = model
        model = nn.DataParallel(model, device_ids=None)

        self.model_path = '../temporal_localization/resnet-34-kinetics.pth'
        print("Loading pretrained weights from %s" %(self.model_path))
        model_data = torch.load(self.model_path)
        model.load_state_dict(model_data['state_dict'])

        # Build resnet.
        self.act_base = nn.Sequential(model.module.conv1, model.module.bn1, model.module.relu,
          model.module.maxpool,model.module.layer1,model.module.layer2, model.module.layer3)

        self.act_top = nn.Sequential(model.module.layer4)
        self.act_top_s = nn.Sequential(_make_layer(BasicBlock, 512, 3, stride=2, inplanes=256))
        self.act_bbox_single_frame_pred = nn.Linear(512, 4 )
        self.act_bbox_pred = nn.Linear(512, 6 ) # 2 classes bg/ fg

        # Fix blocks
        for p in self.act_base[0].parameters(): p.requires_grad=False
        for p in self.act_base[1].parameters(): p.requires_grad=False

        fixed_blocks = 3
        if fixed_blocks >= 3:
          for p in self.act_base[6].parameters(): p.requires_grad=False
        if fixed_blocks >= 2:
          for p in self.act_base[5].parameters(): p.requires_grad=False
        if fixed_blocks >= 1:
          for p in self.act_base[4].parameters(): p.requires_grad=False

        def set_bn_fix(m):
          classname = m.__class__.__name__
          if classname.find('BatchNorm') != -1:
            for p in m.parameters(): p.requires_grad=False

        self.act_base.apply(set_bn_fix)
        self.act_top.apply(set_bn_fix)

    def _head_to_tail(self, pool5):

        batch_size = pool5.size(0)
        fc7 = self.act_top(pool5)
        fc7 = fc7.mean(4)
        fc7 = fc7.mean(3) # exw (bs,512,16)
        return fc7
