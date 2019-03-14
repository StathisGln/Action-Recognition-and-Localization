from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from config import cfg
from proposal_layer import _ProposalLayer
from anchor_target_layer_mine import _AnchorTargetLayer
from net_utils import _smooth_l1_loss

import numpy as np
import math
import pdb
import time

class _RPN(nn.Module):
    """ region proposal network """
    def __init__(self, din):
        super(_RPN, self).__init__()
        

        self.din = din  # get depth of input feature map, e.g., 512
        self.anchor_scales = [4, 8, 16 ]
        self.anchor_ratios = [0.5, 1, 2]
        self.feat_stride = [16, ]
        # self.anchor_duration = [16,8,4,3] # add
        self.anchor_duration = [16,8] # add 
        
        # # define the convrelu layers processing input feature map

        self.RPN_Conv = nn.Conv3d(self.din, 512, 3, stride=1, padding=1, bias=True)

        # define bg/fg classifcation score layer for each kernel 
        # 2(bg/fg) * 9  (anchors) * 4 (duration : 16,8,4,3)
        self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios) * len(self.anchor_duration) * 2 
        self.RPN_cls_score = nn.Conv3d(512, self.nc_score_out, 1, 1, 0)

        # define anchor box offset prediction layer
        # 6(coords:x1,y1,t1) * 9 (anchors)  * 4 (duration)
        self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * len(self.anchor_duration) * 6
        self.RPN_bbox_pred = nn.Conv3d(512, self.nc_bbox_out, 1, 1, 0) # for regression

        ## temporal regression
        # define proposal layer
        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios, self.anchor_duration)

        # define anchor target layer
        self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios, self.anchor_duration)

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        # print('input_shape.shape :',input_shape)
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3],
            input_shape[4]
        )
        return x

    def forward(self, base_feat, im_info, gt_boxes, gt_rois):

        batch_size = base_feat.size(0)
        # print('base_feat.shape :',base_feat.shape)
        # print('Inside region net')
        rpn_conv1 = F.relu(self.RPN_Conv(base_feat), inplace=True) # 3d convolution
        # rpn_conv1 = rpn_conv1.permute(0,1,3,4,2) # move time dim as last dim
        # print('rpn_conv1.shape :',rpn_conv1.shape)

        # ## get classification score for all anchors
        rpn_cls_score = self.RPN_cls_score(rpn_conv1)  # classification layer
        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv1) # regression layer

        # print('rpn_cls_score shape : ', rpn_cls_score.shape)
        # print('rpn_bbox_pred shape : ', rpn_bbox_pred.shape)

        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)
        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'

        # print('rpn_cls_prob.shape :',rpn_cls_prob.shape)
        rois = self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data,
                                     im_info, cfg_key,16))

        # print('rois.shape :',rois.shape)
        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

        # generating training labels a# nd build the rpn loss
        if self.training:

            assert gt_boxes is not None

            # check if gt_boxes are full empty
            # print(gt_boxes.nonzero().nelement())
            # print('gt_boxes :',gt_boxes)
            rpn_data = self.RPN_anchor_target((rpn_cls_score.data, gt_boxes, im_info, gt_rois)) # time_limit = 16

            # print('rpn_cls_score.shape :',rpn_cls_score.shape) 
            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3,4, 1).contiguous()
            rpn_cls_score = rpn_cls_score.view(batch_size, -1, 2) ## exw [1, 441, 2]
            # print('rpn_cls_score.shape :',rpn_cls_score.shape) 

            rpn_label = rpn_data[0].view(batch_size, -1)
            # print('rpn_label :',rpn_label)
            # print('rpn_label :',rpn_label.shape)
            rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))
            # print('rpn_keep :',rpn_keep)
            # print('rpn_label :',rpn_label.view(-1).shape)
            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1,2), 0, rpn_keep)
            # print('rpn_cls_score.shape :',rpn_cls_score.shape)
            # print('rpn_cls_score :',rpn_cls_score)
            # print('rpn_labels :',rpn_label)
            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
            rpn_label = Variable(rpn_label.long())
            # print('rpn_label :',rpn_label)
            # print('rpn_cls_score.shape :',rpn_cls_score.shape)
            # print('rpn_label.shape :',rpn_label.shape)

            self.rpn_loss_cls =  F.cross_entropy(rpn_cls_score, rpn_label)
            # print('rpn_cls_score.shape :',rpn_cls_score.shape)

            # print('self.rpn_loss_cls :',self.rpn_loss_cls)

            fg_cnt = torch.sum(rpn_label.data.ne(0))

            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]
            # print('rpn_bbox_targets.shape :',rpn_bbox_targets.shape)
            # print('rpn_bbox_inside_weights.shape :',rpn_bbox_inside_weights.shape)
            # print('rpn_bbox_outside_weights.shape :',rpn_bbox_outside_weights.shape)
            
            
            rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
            rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
            rpn_bbox_targets = Variable(rpn_bbox_targets)

            self.rpn_loss_box =  _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets.to(rpn_bbox_pred.device), \
                                                 rpn_bbox_inside_weights.to(rpn_bbox_pred.device), \
                                                 rpn_bbox_outside_weights.to(rpn_bbox_pred.device), \
                                                 sigma=3, dim=[1,2,3,4])
            # print('self.rpn_loss_box :',self.rpn_loss_box)
            # print('self.rpn_loss_box :',self.rpn_loss_box)


        return rois, self.rpn_loss_cls, self.rpn_loss_box


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # a good example is v_TrampolineJumping_g17_c01
    # feats = torch.rand(1,512,16,4,4).cuda()
    # feats = torch.rand(1,512,8,4,4).cuda().float()
    feats = torch.rand(2,256,16,7,7).float().to(device)
    gt_bboxes = torch.Tensor([[[42., 44.,  0., 68., 98., 15., 11.]],
                              [[34., 52.,  0., 67., 98., 15., 11.]]]).to(device)
    im_info = torch.Tensor([[112,112,16],[112,112,16]]).to(device)
    n_actions = torch.Tensor([1,1]).to(device)
    model = _RPN(256).to(device)
    out = model(feats,im_info, gt_bboxes, None, n_actions)

