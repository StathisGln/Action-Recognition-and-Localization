from __future__ import absolute_import

import os
import numpy as np
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import avg_pool3d
from torch.autograd import Variable

from resnet_3D import resnet34
from region_net import _RPN
from proposal_target_layer_cascade import _ProposalTargetLayer
from proposal_target_layer_cascade_single_frame import _ProposalTargetLayer as _ProposalTargetLayer_single
from roi_align_3d.modules.roi_align  import RoIAlignAvg
from roi_align.modules.roi_align  import RoIAlignAvg as RoIAlignAvg_s
from net_utils import _smooth_l1_loss
from bbox_transform import  clip_boxes, bbox_transform_inv
## code from resnet

def conv3x3(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=1, bias=False)


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out

def _make_layer( block, planes, blocks, stride=1, inplanes=256):
    downsample = None
    if stride != 1:
      downsample = nn.Sequential(
        nn.Conv2d(inplanes, planes * block.expansion,
              kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(
        inplanes, planes, stride, downsample))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(inplanes, planes))

    return nn.Sequential(*layers)


class ACT_net(nn.Module):
    """ action tube proposal """
    def __init__(self, actions):
        super(ACT_net, self).__init__()

        self.classes = actions
        self.n_classes = len(actions)

        # loss
        self.act_loss_cls = 0
        self.act_loss_bbox = 0

        # cfg.POOLING_SIZE
        self.pooling_size = 7
        self.spatial_scale = 1.0/16.0

        # define rpn
        self.act_rpn = _RPN(256)
        self.act_proposal_target = _ProposalTargetLayer(2) ## background/ foreground
        self.act_proposal_target_single = _ProposalTargetLayer_single(2) ## background/ foreground
        self.time_dim =16
        self.temp_scale = 1.0/4.0
        self.act_roi_align = RoIAlignAvg(self.pooling_size, self.pooling_size, self.time_dim, self.spatial_scale, self.temp_scale)
        self.act_roi_align_s = RoIAlignAvg_s(self.pooling_size, self.pooling_size, self.spatial_scale)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

    def forward(self, im_data, im_info, gt_tubes, gt_rois, num_boxes):

        # print('----------Inside----------')
        batch_size = im_data.size(0)
        im_info = im_info.data
        if self.training:
            gt_tubes = gt_tubes.data
            gt_rois =  gt_rois.data
            num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat = self.act_base(im_data)

        # feed base feature map tp RPN to obtain rois
        rois, rois_16, rpn_loss_cls, rpn_loss_bbox,rpn_loss_cls_16, rpn_loss_bbox_16 = self.act_rpn(base_feat, im_info, gt_tubes, None, num_boxes)
        # if it is training phrase, then use ground trubut bboxes for refining
        # firstly find xy- reggression boxes
        if self.training:
            roi_data = self.act_proposal_target(rois, gt_tubes, num_boxes)

            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data
            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))

            roi_data_16 = self.act_proposal_target(rois_16, gt_tubes, num_boxes)

            rois_16, rois_label_16, rois_target_16, rois_inside_ws_16, rois_outside_ws_16 = roi_data_16
            rois_label_16 = Variable(rois_label_16.view(-1).long())
            rois_target_16 = Variable(rois_target_16.view(-1, rois_target_16.size(2)))
            rois_inside_ws_16 = Variable(rois_inside_ws_16.view(-1, rois_inside_ws_16.size(2)))
            rois_outside_ws_16 = Variable(rois_outside_ws_16.view(-1, rois_outside_ws_16.size(2)))

        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

            rois_label_16 = None
            rois_target_16 = None
            rois_inside_ws_16 = None
            rois_outside_ws_16 = None
            rpn_loss_cls_16 = 0
            rpn_loss_bbox_16 = 0

        rois = Variable(rois)

        # do roi align based on predicted rois
        pooled_feat = self.act_roi_align(base_feat, rois.view(-1,7))

        # # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)
        n_rois = pooled_feat.size(0)

        # # compute bbox offset
        pooled_feat_mean = pooled_feat.mean(2)
        
        bbox_pred = self.act_bbox_pred(pooled_feat_mean)

        act_loss_cls = 0
        act_loss_bbox = 0

        if self.training:
            # # classification loss
            # # print('rois_label :', rois_label)
            # act_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            act_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        ####################################################################################################
        
        ## frame 16
        rois_16 = Variable(rois_16)

        # do roi align based on predicted rois
        # print('base_feat.shape :',base_feat.shape)
        # print('rois_16 :',rois_16.shape)
        pooled_feat_16 = self.act_roi_align(base_feat, rois_16.view(-1,7))
        # # feed pooled features to top model
        # print('pooled_feat.shape :',pooled_feat.shape)
        pooled_feat_16 = self._head_to_tail(pooled_feat_16)
        n_rois_16 = pooled_feat_16.size(0)

        # # compute bbox offset
        pooled_feat_mean_16 = pooled_feat_16.mean(2)
        # print('pooled_feat_16 :',pooled_feat_mean_16.shape)        
        bbox_pred_16 = self.act_bbox_pred(pooled_feat_mean_16)

        act_loss_cls_16 = 0
        act_loss_bbox_16 = 0

        if self.training:
            # # classification loss
            # # print('rois_label :', rois_label)
            # act_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            act_loss_bbox_16 = _smooth_l1_loss(bbox_pred_16, rois_target_16, rois_inside_ws_16, rois_outside_ws_16)


        # # print('bbox_pred.shape :',bbox_pred.shape)
        if self.training:
            return rois, rois_16, bbox_pred, rpn_loss_cls, rpn_loss_bbox, rpn_loss_cls_16, rpn_loss_bbox_16, act_loss_cls, act_loss_bbox, act_loss_cls_16, act_loss_bbox_16

        return rois,  bbox_pred, 

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
        # normal_init(self.act_cls_score, 0, 0.001, truncated)


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
        # self.act_bbox_pred = nn.Linear(512, 6 ) # 2 classes bg/ fg
        # self.act_bbox_pred = nn.Linear(8192, 6 * self.n_classes) # 2 classes bg/ fg
        # self.act_cls_score = nn.Linear(8192, self.n_classes)
        self.act_bbox_single_frame_pred = nn.Linear(512, 4 )
        self.act_bbox_pred = nn.Linear(512, 6 ) # 2 classes bg/ fg
        # self.act_cls_score = nn.Linear(512, self.n_classes)

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
        # print('pool5.shape :',pool5.shape)
        batch_size = pool5.size(0)
        fc7 = self.act_top(pool5)
        fc7 = fc7.mean(4)
        fc7 = fc7.mean(3) # exw (bs,512,16)

        # print('fc7.shape :',fc7.shape)
        return fc7


    
