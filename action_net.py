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
from region_net import _RPN 
from human_reg import _Regression_Layer

from proposal_target_layer_cascade import _ProposalTargetLayer
from proposal_target_layer_cascade_single_frame import _ProposalTargetLayer as _ProposalTargetLayer_single
from roi_align_3d.modules.roi_align  import RoIAlignAvg, RoIAlign
from net_utils import _smooth_l1_loss
from bbox_transform import  clip_boxes, bbox_transform_inv, clip_boxes_3d, bbox_transform_inv_3d
## code from resnet


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
        self.act_proposal_target = _ProposalTargetLayer(2).cuda() ## background/ foreground

        self.time_dim =sample_duration
        self.temp_scale = 1.
        self.act_roi_align = RoIAlignAvg(self.pooling_size, self.pooling_size, self.time_dim, self.spatial_scale, self.temp_scale).cuda()
        # self.act_roi_align = RoIAlign(self.pooling_size, self.pooling_size, self.time_dim, self.spatial_scale, self.temp_scale)
        # self.avgpool = nn.AvgPool3d((1, 7, 7), stride=1)

        
        self.avgpool = nn.AvgPool3d((1, 7, 7), stride=1)
        self.reg_layer = _Regression_Layer(256, self.sample_duration).cuda()

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

    def forward(self, im_data, im_info, gt_tubes, gt_rois,  start_fr):

        # print('----------Inside TPN net----------')
        batch_size = im_data.size(0)
        im_info = im_info.data
        if self.training:
            gt_tubes = gt_tubes.data
            gt_rois =  gt_rois.data

        #### modify gt_tubes:

        for i in range(batch_size):
          
          gt_tubes[i,:,2] = gt_tubes[i,:,2] - start_fr[i].type_as(gt_tubes)
          gt_tubes[i,:,5] = gt_tubes[i,:,5] - start_fr[i].type_as(gt_tubes)

        gt_tubes[:,:,:-1] = gt_tubes[:,:,:-1].clamp_(min=0)
        # print('gt_tubes :',gt_tubes)

        # feed image data to base model to obtain base feature map

        base_feat = self.act_base(im_data)
        rois, rpn_loss_cls, rpn_loss_bbox, \
            rpn_loss_cls_16, rpn_loss_bbox_16 = self.act_rpn(base_feat, im_info, gt_tubes, None)

        # if it is training phrase, then use ground trubut bboxes for refining
        # firstly find xy- reggression boxes
        
        if self.training:
          gt_tubes = torch.cat((gt_tubes,torch.ones(gt_tubes.size(0),gt_tubes.size(1),1).type_as(gt_tubes)),dim=2).type_as(gt_tubes)
          roi_data = self.act_proposal_target(rois, gt_tubes)

          rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data
          rois_label = Variable(rois_label.view(-1).long())
          rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
          rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
          rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
          rois_label = None
          rois_target = None
          rois_inside_ws = None
          rois_outside_ws = None
          rpn_loss_cls = 0
          rpn_loss_bbox = 0

        # rois = Variable(rois)

        # do roi align based on predicted rois
        rois_s = rois[:,:,:7].contiguous()
        pooled_feat_ = self.act_roi_align(base_feat, rois_s.view(-1,7))

        ## regression
        sgl_rois_bbox_pred, sgl_rois_bbox_loss = self.reg_layer(pooled_feat_,rois[:,:,:7], gt_rois) 

        # print('pooled_feat_ :',pooled_feat_.shape)
        pooled_feat = self._head_to_tail(pooled_feat_)
        n_rois = pooled_feat.size(0)

        # # compute bbox offset
        pooled_feat = pooled_feat.mean(2)
        bbox_pred = self.act_bbox_pred(pooled_feat)
        # compute object classification probability

        # act_loss_cls = 0
        act_loss_bbox = 0

        if self.training:
            # bounding box regression L1 loss
            act_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        pooled_feat_ = self.avgpool(pooled_feat_)

        if self.training:
          rois_label = rois_label.view(batch_size, rois.size(1),-1)
          return rois,  bbox_pred, pooled_feat_, \
            rpn_loss_cls, rpn_loss_bbox, act_loss_bbox,\
            rpn_loss_cls_16, rpn_loss_bbox_16, \
            rois_label, sgl_rois_bbox_pred, sgl_rois_bbox_loss


        return rois,  bbox_pred, pooled_feat_, None, None, None, None, None, None, sgl_rois_bbox_pred, None

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
        normal_init(self.reg_layer.Conv, 0, 0.01, truncated)
        normal_init(self.reg_layer.bbox_pred, 0, 0.01, truncated)
        normal_init(self.act_bbox_pred, 0, 0.001, truncated)
        normal_init(self.act_bbox_single_frame_pred, 0, 0.001, truncated)
        # normal_init(self.act_cls_score, 0, 0.001, truncated)


    def _init_modules(self):

        resnet_shortcut = 'A'
        sample_size = 112

        model = resnet34(num_classes=400, shortcut_type=resnet_shortcut,
                         sample_size=sample_size, sample_duration=self.sample_duration,
                         last_fc=False)
        model = model
        model = nn.DataParallel(model, device_ids=None)
        self.model_path = '/gpu-data2/sgal/resnet-34-kinetics.pth'
        print("Loading pretrained weights from %s" %(self.model_path))
        model_data = torch.load(self.model_path)

        # new_state_dict = OrderedDict()
        # for k, v in model_data['state_dict'].items():
        #   name = k[7:] # remove `module.`
        #   new_state_dict[name] = v

        # model.load_state_dict(new_state_dict)
        model.load_state_dict(model_data['state_dict'])

        # Build resnet.
        # self.act_base = nn.Sequential(model.conv1, model.bn1, model.relu,
        #   model.maxpool,model.layer1,model.layer2, model.layer3)
        self.act_base = nn.Sequential(model.module.conv1, model.module.bn1, model.module.relu,
          model.module.maxpool,model.module.layer1,model.module.layer2, model.module.layer3)


        self.act_top = nn.Sequential(model.module.layer4)

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
        # print('pool5.shape :',pool5.shape)
        batch_size = pool5.size(0)
        fc7 = self.act_top(pool5)
        fc7 = fc7.mean(4)
        fc7 = fc7.mean(3) # exw (bs,512,16)
        # print('fc7.shape :',fc7.shape)
        # fc7 = fc7.mean(2) # exw (bs,512,16)
        # print('fc7.shape :',fc7.shape)
        return fc7
    def _head_to_tail_s(self, pool5):
        # print('pool5.shape :',pool5.shape)
        fc7 = self.act_top_s(pool5)
        fc7 = fc7.mean(3)
        fc7 = fc7.mean(2) # exw (bs,512)
        return fc7


    
