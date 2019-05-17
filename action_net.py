from __future__ import absolute_import

import os
import numpy as np
import glob
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import avg_pool3d
from torch.autograd import Variable

from resnet_3D import resnet34
from resnext import resnet101
from region_net import _RPN 
from human_reg import _Regression_Layer

from proposal_target_layer_cascade import _ProposalTargetLayer
from proposal_target_layer_cascade_single_frame import _ProposalTargetLayer as _ProposalTargetLayer_single
from roi_align_3d.modules.roi_align  import RoIAlignAvg, RoIAlign
from net_utils import _smooth_l1_loss
from bbox_transform import  clip_boxes, bbox_transform_inv, clip_boxes_3d, bbox_transform_inv_3d


class ACT_net(nn.Module):
    """ action tube proposal """
    def __init__(self, actions,sample_duration):
        super(ACT_net, self).__init__()

        self.classes = actions
        self.n_classes = len(actions)
        self.sample_duration = sample_duration

        # loss
        self.act_loss_cls = 0
        self.act_loss_bbox = 0

        # cfg.POOLING_SIZE
        self.pooling_size = 7
        self.spatial_scale = 1.0/16
    
        # define rpn
        self.act_rpn = _RPN(256, sample_duration).cuda()

        self.maxpool3d = nn.MaxPool3d(1, stride=(1,2,2))

        self.act_proposal_target = _ProposalTargetLayer(2).cuda() ## background/ foreground
        self.act_proposal_target_single = _ProposalTargetLayer_single(2) ## background/ foreground for only xy

        self.time_dim =sample_duration
        self.temp_scale = 1.0
        self.act_roi_align = RoIAlign(self.pooling_size, self.pooling_size, self.time_dim, self.spatial_scale, self.temp_scale).cuda()

        self.avgpool = nn.AvgPool3d((1, 7, 7), stride=1)

        self.reg_layer = _Regression_Layer(256, self.sample_duration).cuda()

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,T,H,W = y.size()
        # return F.upsample(x, size=(T,H,W), mode='trilinear', align_corners=False) + y
        return F.interpolate(x, size=(T,H,W), mode='trilinear', align_corners=False) + y

    def forward(self, im_data, im_info, gt_tubes, gt_rois,  start_fr):

        batch_size = im_data.size(0)
        im_info = im_info.data

        if self.training:

            gt_tubes = gt_tubes.data
            gt_rois =  gt_rois.data

            for i in range(batch_size):

              gt_tubes[i,:,2] = gt_tubes[i,:,2] - start_fr[i].type_as(gt_tubes)
              gt_tubes[i,:,5] = gt_tubes[i,:,5] - start_fr[i].type_as(gt_tubes)

            gt_tubes[:,:,:-1] = gt_tubes[:,:,:-1].clamp_(min=0)

        c1 = self.act_layer0(im_data)
        c2 = self.act_layer1(c1)
        c3 = self.act_layer2(c2)
        c4 = self.act_layer3(c3)
        c5 = self.act_layer4(c4)

        # Top-down
        p5 = self.act_toplayer(c5)
        p4 = self._upsample_add(p5, self.act_latlayer1(c4))
        p4 = self.act_smooth1(p4)
        p3 = self._upsample_add(p4, self.act_latlayer2(c3))
        p3 = self.act_smooth2(p3)
        p2 = self._upsample_add(p3, self.act_latlayer3(c2))
        p2 = self.act_smooth3(p2)

        # p6 = self.maxpool3d(p5)

        rpn_feature_maps = [p2, p3, p4, p5]

        rois, rois_16, rpn_loss_cls, rpn_loss_bbox, \
            rpn_loss_cls_16, rpn_loss_bbox_16 = self.act_rpn(rpn_feature_maps, im_info, gt_tubes, None)

        n_rois = rois.size(1)
        f_rois = rois

        if self.training:
          return f_rois,  None, rpn_loss_cls, None, \
            None,None,None,None,None,

        return f_rois,  None, None, None, None, None, \
            None, None, None,

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
        # normal_init(self.act_rpn.RPN_bbox_pred, 0, 0.01, truncated)
        # normal_init(self.reg_layer.Conv, 0, 0.01, truncated)
        # normal_init(self.reg_layer.bbox_pred, 0, 0.01, truncated)

        normal_init(self.act_smooth1, 0, 0.01,  truncated)
        normal_init(self.act_smooth2, 0, 0.01,  truncated)
        normal_init(self.act_smooth3, 0, 0.01,  truncated)
        normal_init(self.act_latlayer1, 0, 0.01,  truncated)
        normal_init(self.act_latlayer2, 0, 0.01,  truncated)
        normal_init(self.act_latlayer3, 0, 0.01,  truncated)



    def _init_modules(self):

        # resnet_shortcut = 'A'
        resnet_shortcut = 'B'

        sample_size = 112

        # model = resnet34(num_classes=400, shortcut_type=resnet_shortcut,
        #                  sample_size=sample_size, sample_duration=self.sample_duration,
        #                  last_fc=False)
        model = resnet101(num_classes=400, shortcut_type=resnet_shortcut,
                         sample_size=sample_size, sample_duration=self.sample_duration,
                         )

        model = nn.DataParallel(model, device_ids=None)
        # self.model_path = '../resnet-34-kinetics.pth'
        self.model_path = '../resnext-101-kinetics.pth'
        print("Loading pretrained weights from %s" %(self.model_path))
        model_data = torch.load(self.model_path)

        model.load_state_dict(model_data['state_dict'])

        # Build resnet.
        self.act_layer0 = nn.Sequential(model.module.conv1, model.module.bn1, model.module.relu,
          model.module.maxpool)
        self.act_layer1 = nn.Sequential(model.module.layer1)
        self.act_layer2 = nn.Sequential(model.module.layer2)
        self.act_layer3 = nn.Sequential(model.module.layer3)
        self.act_layer4 = nn.Sequential(model.module.layer4)

        # Top layer
        self.act_toplayer = nn.Conv3d(2048, 256, kernel_size=1, stride=1, padding=0)  # reduce channel

        # Smooth layers
        self.act_smooth1 = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1)
        self.act_smooth2 = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1)
        self.act_smooth3 = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.act_latlayer1 = nn.Conv3d( 1024, 256, kernel_size=1, stride=1, padding=0)
        self.act_latlayer2 = nn.Conv3d( 512,  256, kernel_size=1, stride=1, padding=0)
        self.act_latlayer3 = nn.Conv3d( 256,  256, kernel_size=1, stride=1, padding=0)

        # ROI Pool feature downsampling
        self.act_roi_feat_ds = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

        # Fix blocks
        for p in self.act_layer0[0].parameters(): p.requires_grad=False
        for p in self.act_layer0[1].parameters(): p.requires_grad=False

        fixed_blocks = 3
        if fixed_blocks >= 3:
          for p in self.act_layer3.parameters(): p.requires_grad=False
        if fixed_blocks >= 2:
          for p in self.act_layer2.parameters(): p.requires_grad=False
        if fixed_blocks >= 1:
          for p in self.act_layer1.parameters(): p.requires_grad=False

        def set_bn_fix(m):
          classname = m.__class__.__name__
          if classname.find('BatchNorm') != -1:
            for p in m.parameters(): p.requires_grad=False

    
        self.act_layer0.apply(set_bn_fix)
        self.act_layer1.apply(set_bn_fix)
        self.act_layer2.apply(set_bn_fix)
        self.act_layer3.apply(set_bn_fix)
        self.act_layer4.apply(set_bn_fix)

