from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from config import cfg
from proposal_layer import _ProposalLayer
from proposal_layer_xy import _ProposalLayer_xy
from anchor_target_layer_mine import _AnchorTargetLayer
from anchor_target_layer_xy import _AnchorTargetLayer_xy
from net_utils import _smooth_l1_loss

import numpy as np
import math
import pdb
import time

class _RPN(nn.Module):
    """ region proposal network """
    def __init__(self, din, sample_duration):
        super(_RPN, self).__init__()
        

        self.din = din  # get depth of input feature map, e.g., 512
        self.sample_duration =sample_duration # get sample_duration
        self.anchor_scales = [4, 8, 16 ]
        self.anchor_ratios = [0.5, 1, 2]
        self.feat_stride = [16, ]
        # self.anchor_duration = [16,8,4,3] # add
        self.anchor_duration = [sample_duration,int(sample_duration/2),int(sample_duration/4)] # add 

        # # define the convrelu layers processing input feature map

        self.RPN_Conv = nn.Conv3d(self.din, 512, 3, stride=1, padding=1, bias=True)

        # define bg/fg classifcation score layer for each kernel 
        # 2(bg/fg) * 9  (anchors) * 4 (duration : 16,8,4,3)
        self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios) * len(self.anchor_duration) * 2
        # 2(bg/fg) * 9  (anchors) 
        self.nc_score_16 = len(self.anchor_scales) * len(self.anchor_ratios) * 2 
        self.RPN_cls_score = nn.Conv3d(512, self.nc_score_out, 1, 1, 0)
        self.RPN_cls_16 = nn.Conv2d(512, self.nc_score_16, 1, 1, 0)

        # define anchor box offset prediction layer
        # 6(coords:x1,y1,t1) * 9 (anchors)  * 2 (duration)
        self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * len(self.anchor_duration) * 6
        # 4(coords:x1,y1) * 9 (anchors)  
        self.bbox_16 = len(self.anchor_scales) * len(self.anchor_ratios) * 4
        self.RPN_bbox_pred = nn.Conv3d(512, self.nc_bbox_out, 1, 1, 0) # for regression
        self.RPN_bbox_only16 = nn.Conv2d(512, self.bbox_16, 1, 1, 0) # for regression
        ## temporal regression
        # define proposal layer
        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios, self.anchor_duration)
        self.RPN_proposal_16 = _ProposalLayer_xy(self.feat_stride,  self.anchor_scales, self.anchor_ratios, [sample_duration])
        # define anchor target layer
        self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride,  self.anchor_scales, self.anchor_ratios, self.anchor_duration)
        self.RPN_anchor_16 = _AnchorTargetLayer_xy(self.feat_stride, self.anchor_scales, self.anchor_ratios, [sample_duration])

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0
        self.rpn_loss_cls_16 = 0
        self.rpn_loss_box_16 = 0
        # self.keep=self.RPN_cls_score.weight.data.clone() # modify here
        # self.init_rpn()
        
    # def init_rpn(self):

    #     def normal_init(m, mean, stddev, truncated=False):
    #         """
    #         weight initalizer: truncated normal and random normal.
    #         """
    #         # x is a parameter
    #         if truncated:
    #             m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
    #         else:
    #             m.weight.data.normal_(mean, stddev)
    #             m.bias.data.zero_()

    #     truncated = False
    #     normal_init(self.RPN_Conv, 0, 0.01, truncated)
    #     normal_init(self.RPN_cls_score, 0, 0.01, truncated)
    #     normal_init(self.RPN_bbox_pred, 0, 0.001, truncated)


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

    @staticmethod
    def reshape2d(x, d):
        input_shape = x.size()
        # print('input_shape.shape :',input_shape)
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3],
        )
        return x


    def forward(self, base_feat, im_info, gt_boxes, gt_rois):

        batch_size = base_feat.size(0)
        # print('Inside region net')
        rpn_conv1 = F.relu(self.RPN_Conv(base_feat), inplace=True) # 3d convolution
        # rpn_conv1 = rpn_conv1.permute(0,1,3,4,2) # move time dim as last dim
        # print('rpn_conv1.shape :',rpn_conv1.shape)

        # ## get classification score for all anchors
        rpn_cls_score = self.RPN_cls_score(rpn_conv1)  # classification layer
        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv1)  # regression layer

        rpn_conv1 = rpn_conv1.permute(0,1,3,4,2).mean(4)
        rpn_cls_16    = self.RPN_cls_16(rpn_conv1)  # classification layer
        rpn_bbox_16   = self.RPN_bbox_only16(rpn_conv1)
        # print('rpn_cls_score shape : ', rpn_cls_score.shape)
        # print('rpn_cls_16    shape : ', rpn_cls_16.shape)
        # print('rpn_bbox_pred shape : ', rpn_bbox_pred.shape)
        # print('rpn_bbox_16   shape : ', rpn_bbox_16.shape)

        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)

        rpn_cls_16_reshape = self.reshape2d(rpn_cls_16, 2)
        rpn_16_prob_reshape = F.softmax(rpn_cls_16_reshape, 1)
        rpn_16_prob = self.reshape2d(rpn_16_prob_reshape, self.nc_score_16)


        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'

        # print('rpn_cls_prob.shape :',rpn_cls_prob.shape)
        rois = self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data,
                                     im_info, cfg_key,16))
        # print('rpn_bbox_16.shape :',rpn_bbox_16.shape)
        # print('rpn_bbox_pred.shape :',rpn_bbox_pred.shape)
        rois_16 = self.RPN_proposal_16((rpn_16_prob.data, rpn_bbox_16.data,
                                     im_info, cfg_key,16))

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0
        self.rpn_loss_cls_16 = 0
        self.rpn_loss_box_16 = 0


        # generating training labels a# nd build the rpn loss
        if self.training:

            assert gt_boxes is not None

            rpn_data = self.RPN_anchor_target((rpn_cls_score.data, gt_boxes, im_info, gt_rois, self.sample_duration)) # time_limit = 16
            rpn_data_16 = self.RPN_anchor_16((rpn_cls_16.data, gt_boxes, im_info, gt_rois, self.sample_duration)) # time_limit = 16

            ## Regular data
            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3,4, 1).contiguous()
            rpn_cls_score = rpn_cls_score.view(batch_size, -1, 2) ## exw [1, 441, 2]
            # print('rpn_cls_score.shape :',rpn_cls_score.shape) 

            rpn_label = rpn_data[0].view(batch_size, -1)
            # print('rpn_label :',rpn_label.shape)
            rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))
            # print('rpn_label :',rpn_label.view(-1).shape)
            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1,2), 0, rpn_keep)
            # print('rpn_cls_score.shape :',rpn_cls_score.shape)
            # print('rpn_cls_score :',rpn_cls_score)
            # print('rpn_labels :',rpn_label)
            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
            rpn_label = Variable(rpn_label.long())

            # print('rpn_cls_score.shape :',rpn_cls_score.shape)
            # print('rpn_label.shape :',rpn_label.shape)

            self.rpn_loss_cls =  F.cross_entropy(rpn_cls_score, rpn_label)
            # print('self.rpn_loss_cls :',self.rpn_loss_cls)
            fg_cnt = torch.sum(rpn_label.data.ne(0))

            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]
            # print('rpn_bbox_targets.shape :',rpn_bbox_targets.shape)
            # print('rpn_bbox_inside_weights.shape :',rpn_bbox_inside_weights.shape)
            # print('rpn_bbox_outside_weights.shape :',rpn_bbox_outside_weights.shape)
            
            rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
            rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
            rpn_bbox_targets = Variable(rpn_bbox_targets)

            self.rpn_loss_box =  _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                               rpn_bbox_outside_weights, sigma=3, dim=[1,2,3,4])
            ## only 16 frames

            # print('self.rpn_loss_box :',self.rpn_loss_box)
            # print('self.rpn_loss_box :',self.rpn_loss_box)
            rpn_cls_16 = rpn_cls_16_reshape.permute(0, 2, 3, 1).contiguous()
            rpn_cls_16 = rpn_cls_16.view(batch_size, -1, 2) ## exw [1, 441, 2]
            # print('rpn_cls_score.shape :',rpn_cls_score.shape) 

            rpn_label_16 = rpn_data_16[0].view(batch_size, -1)
            # print('rpn_label :',rpn_label.shape)
            rpn_keep_16 = Variable(rpn_label_16.view(-1).ne(-1).nonzero().view(-1))
            # print('rpn_label :',rpn_label.view(-1).shape)
            rpn_cls_score_16 = torch.index_select(rpn_cls_16.view(-1,2), 0, rpn_keep_16)
            # print('rpn_cls_score.shape :',rpn_cls_score.shape)
            # print('rpn_cls_score :',rpn_cls_score)
            # print('rpn_labels :',rpn_label)
            rpn_label_16 = torch.index_select(rpn_label_16.view(-1), 0, rpn_keep_16.data)
            rpn_label_16 = Variable(rpn_label_16.long())

            # print('rpn_cls_score.shape :',rpn_cls_score.shape)
            # print('rpn_label.shape :',rpn_label.shape)

            self.rpn_loss_cls_16 =  F.cross_entropy(rpn_cls_score_16, rpn_label_16)
            # print('self.rpn_loss_cls :',self.rpn_loss_cls)
            fg_cnt_16 = torch.sum(rpn_label_16.data.ne(0))

            rpn_bbox_targets_16, rpn_bbox_inside_weights_16, rpn_bbox_outside_weights_16 = rpn_data_16[1:]
            # print('rpn_bbox_targets.shape :',rpn_bbox_targets.shape)
            # print('rpn_bbox_inside_weights.shape :',rpn_bbox_inside_weights.shape)
            # print('rpn_bbox_outside_weights.shape :',rpn_bbox_outside_weights.shape)
            
            
            rpn_bbox_inside_weights_16 = Variable(rpn_bbox_inside_weights_16)
            rpn_bbox_outside_weights_16 = Variable(rpn_bbox_outside_weights_16)
            rpn_bbox_targets_16 = Variable(rpn_bbox_targets_16)

            # print('rpn_bbox_16.shape :',rpn_bbox_16.shape)
            # print('rpn_bbox_targets_16.shape :',rpn_bbox_targets_16.shape)
            self.rpn_loss_box_16 =  _smooth_l1_loss(rpn_bbox_16, rpn_bbox_targets_16, rpn_bbox_inside_weights_16,
                                                               rpn_bbox_outside_weights_16, sigma=3, dim=[1,2,3])
        return rois, rois_16, self.rpn_loss_cls, self.rpn_loss_box, self.rpn_loss_cls_16, self.rpn_loss_box_16


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

