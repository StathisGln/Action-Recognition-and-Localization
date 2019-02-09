from __future__ import absolute_import

import os
import numpy as np
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from resnet_3D import resnet34
from region_net import _RPN
from proposal_target_layer_cascade import _ProposalTargetLayer

from roi_align.modules.roi_align  import RoIAlignAvg
from net_utils import _smooth_l1_loss

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
        pooling_size = 7
        # define rpng
        self.act_rpn = _RPN(256)
        self.act_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.time_dim =16
        self.act_roi_align = RoIAlignAvg(pooling_size, pooling_size, 1.0/28.0, self.time_dim)
        # self.act_roi_align = RoIAlignAvg(pooling_size, pooling_size, 1.0/16.0)

        self._init_modules()
        self._init_weights()

    def forward(self, im_data, im_info, gt_tubes, gt_rois, num_boxes):

        batch_size = im_data.size(0)
        # print('batch_size :', batch_size)
        n_rois_batch = gt_rois.size(0)
        # print('n_rois_batch :', n_rois_batch)
        # print('gt_tubes :', gt_tubes.shape)
        # print('gt_tubes :', gt_tubes)
        im_info = im_info.data
        gt_tubes = gt_tubes.data
        gt_rois = gt_rois.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat = self.act_base(im_data)

        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.act_rpn(base_feat, im_info, gt_tubes, gt_rois, num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            # print('gt_rois.shape :',gt_rois.shape)
            # print('gt_rois[0,:,:4] :',gt_rois)
            # print('gt_rois[0,:,:4].contiguous().view(n_rois_batch,-1) :',gt_rois[:,:,:4].contiguous().view(n_rois_batch,-1))
            # print('gt_rois[0,:,:4].contiguous().view(n_rois_batch,-1).shape :',gt_rois[:,:,:4].contiguous().view(n_rois_batch,-1).shape)
            # print('gt_rois[:,0,4] :',gt_tubes[:,:,6])
            # print('gt_rois[:,0,4] :',gt_tubes[:,:,6].permute(1,0).shape)

            gt_rois_reshaped = torch.cat((gt_rois[:,:,:4].contiguous().view(n_rois_batch,-1) , gt_tubes[:,:,6].permute(1,0)),dim=1).unsqueeze(0)
            # print('gt_rois_reshaped.shape :',gt_rois_reshaped.shape)
            roi_data = self.act_proposal_target(rois.unsqueeze(0), gt_rois_reshaped, num_boxes)
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

        rois = Variable(rois)
        # do roi pooling based on predicted rois

        # print('rois.shape :',rois.shape)
        # print('rois.size(2) :',rois.size(2))
        n_dim = rois.size(2)
        pooled_feat = self.act_roi_align(base_feat, rois.view(-1, n_dim))

        # # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)
        # print('pooled_feat.shape :',pooled_feat.shape)
        n_rois = pooled_feat.size(0)
        # print('n_rois :',n_rois)
        # print('pooled_feat.view(n_rois,-1).shape :',pooled_feat.view(n_rois,-1).shape)
        # # compute bbox offset
        bbox_pred = self.act_bbox_pred(pooled_feat.view(n_rois,-1))
        if self.training:
            # # select the corresponding columns according to roi labels
            # print('bbox_pred.shape :',bbox_pred.shape)
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        act_loss_cls = 0
        act_loss_bbox = 0

        if self.training:
            # bounding box regression L1 loss
            act_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)
        # return 0,0,0,0,0,0
        return rois,  bbox_pred, rpn_loss_cls, rpn_loss_bbox,  act_loss_bbox, rois_label

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
        normal_init(self.act_rpn.RPN_time_16, 0, 0.01, truncated)
        normal_init(self.act_rpn.RPN_time_8, 0, 0.01, truncated)
        normal_init(self.act_rpn.RPN_time_4, 0, 0.01, truncated)
        normal_init(self.act_rpn.RPN_cls_score_16, 0, 0.01, truncated)
        normal_init(self.act_rpn.RPN_cls_score_8, 0, 0.01, truncated)
        normal_init(self.act_rpn.RPN_cls_score_4, 0, 0.01, truncated)
        normal_init(self.act_rpn.RPN_bbox_frame_pred_16, 0, 0.01, truncated)
        normal_init(self.act_rpn.RPN_bbox_frame_pred_16, 0, 0.01, truncated)
        normal_init(self.act_rpn.RPN_bbox_frame_pred_16, 0, 0.01, )
        normal_init(self.act_bbox_pred, 0, 0.001, truncated)

    def _init_modules(self):

        resnet_shortcut = 'A'
        sample_size = 112
        sample_duration = 16  # len(images)

        model = resnet34(num_classes=400, shortcut_type=resnet_shortcut,
                         sample_size=sample_size, sample_duration=sample_duration,
                         last_fc=False)
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)

        self.model_path = '../temporal_localization/resnet-34-kinetics.pth'
        print("Loading pretrained weights from %s" %(self.model_path))
        model_data = torch.load(self.model_path)
        model.load_state_dict(model_data['state_dict'])
        # Build resnet.
        self.act_base = nn.Sequential(model.module.conv1, model.module.bn1, model.module.relu,
          model.module.maxpool,model.module.layer1,model.module.layer2, model.module.layer3)

        self.act_top = nn.Sequential(model.module.layer4)

        self.act_bbox_pred = nn.Linear(2048, self.time_dim * 4 * self.n_classes)

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
        fc7 = self.act_top(pool5)
        # print('fc7.shape :',fc7.shape)
        fc7 = fc7.mean(3)
        # print('fc7.shape :',fc7.shape)
        fc7 = fc7.mean(2)
        return fc7

    
