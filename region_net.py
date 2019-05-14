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
from net_utils import _smooth_l1_loss, get_number_of_combinations

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
        # self.anchor_scales = [0.5,1, 2, 4, 8, 16 ]
        self.anchor_scales = [1, 2, 4, 8, 16, 32 ]

        self.anchor_ratios = [0.5, 1, 2]
        self.feat_stride = [16, ]
        # self.anchor_duration = [16,8,4,3] # add
        self.anchor_duration = [sample_duration,int(sample_duration*3/4), int(sample_duration/2)] #,int(sample_duration/4)] # add 

        # # define the convrelu layers processing input feature map
        self.anchor_num = get_number_of_combinations(sample_duration, self.anchor_duration)
        self.RPN_Conv = nn.Conv3d(self.din, self.din * 2, 3, stride=1, padding=1, bias=True)

        # define bg/fg classifcation score layer for each kernel 
        # 2(bg/fg) * 9  (anchors) * 3 (duration : 16,12,8)
        self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios) * self.anchor_num * 2
        # 2(bg/fg) * 9  (anchors) 
        self.nc_score_16 = len(self.anchor_scales) * len(self.anchor_ratios) * 2 

        self.RPN_cls_score = nn.Conv3d(self.din*2, self.nc_score_out, (self.sample_duration,1,1), 1, 0)
        self.RPN_cls_16 = nn.Conv3d(self.din*2, self.nc_score_16, (sample_duration,1,1), 1, 0)

        # define anchor box offset prediction layer
        # 6(coords:x1,y1,t1) * 9 (anchors)  * 2 (duration)
        self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * self.anchor_num * self.sample_duration * 4
        # 4(coords:x1,y1) * 9 (anchors)  
        self.bbox_16 = len(self.anchor_scales) * len(self.anchor_ratios) * 4
        self.RPN_bbox_pred = nn.Conv3d(self.din*2, self.nc_bbox_out, (sample_duration,1,1), 1, 0) # for regression
        self.RPN_bbox_only16 = nn.Conv3d(self.din*2, self.bbox_16, (sample_duration,1,1), 1, 0) # for regression
        ## temporal regression
        # define proposal layer
        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios, self.anchor_duration,  len(self.anchor_scales) * len(self.anchor_ratios) * self.anchor_num)
        self.RPN_proposal_16 = _ProposalLayer_xy(self.feat_stride,  self.anchor_scales, self.anchor_ratios, [sample_duration])

        # define anchor target layer
        self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride,  self.anchor_scales, self.anchor_ratios, self.anchor_duration, self.anchor_num)
        self.RPN_anchor_16 = _AnchorTargetLayer_xy(self.feat_stride, self.anchor_scales, self.anchor_ratios, [sample_duration])

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0
        self.rpn_loss_cls_16 = 0
        self.rpn_loss_box_16 = 0

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
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
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3],
        )
        return x


    def forward(self, base_feat, im_info, gt_boxes, gt_rois):

        batch_size = base_feat.size(0)
        rpn_conv1 = F.relu(self.RPN_Conv(base_feat), inplace=True) # 3d convolution

        # # ## get classification score for all anchors
        rpn_cls_score = self.RPN_cls_score(rpn_conv1).squeeze(2)  # classification layer
        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv1).squeeze(2)  # regression layer

        # 16 frames 

        # rpn_cls_16    = self.RPN_cls_16(rpn_conv1).squeeze(-3)  # classification layer
        # rpn_bbox_16   = self.RPN_bbox_only16(rpn_conv1).squeeze(-3)
        # print('rpn_bbox_pred.shape ',rpn_bbox_pred.shape)
        # exit(-1)
        # reshape
        rpn_cls_score_reshape = self.reshape2d(rpn_cls_score, 2)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)
        rpn_cls_prob = self.reshape2d(rpn_cls_prob_reshape, self.nc_score_out)

        # rpn_cls_16_reshape = self.reshape2d(rpn_cls_16, 2)
        # rpn_16_prob_reshape = F.softmax(rpn_cls_16_reshape, 1)
        # rpn_16_prob = self.reshape2d(rpn_16_prob_reshape, self.nc_score_16)

        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'


        rois = self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data,
                                     im_info, cfg_key,16))
        # rois_16 = self.RPN_proposal_16((rpn_16_prob.data, rpn_bbox_16.data,
        #                              im_info, cfg_key,16))

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0
        # self.rpn_loss_cls_16 = 0
        # self.rpn_loss_box_16 = 0


        # generating training labels a# nd build the rpn loss
        if self.training:

            assert gt_boxes is not None

            # get tubes lasting sample duration
            gt_boxes_16 = gt_boxes.new(gt_boxes.shape).zero_()

            dur = gt_boxes[:,:,5] - gt_boxes[:,:,2] + 1
            dur_16 = dur.eq(self.sample_duration).nonzero()
            if dur_16.nelement() != 0:
                for b in range(dur_16.size(0)):
                    gt_boxes_16[dur_16[b,0],dur_16[b,1]] =  gt_boxes[dur_16[b,0],dur_16[b,1]]

            ## Regular data
            rpn_data = self.RPN_anchor_target((rpn_cls_score.data, gt_boxes, im_info, gt_rois, self.sample_duration)) # time_limit = 16

            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3,4, 1).contiguous()
            rpn_cls_score = rpn_cls_score.view(batch_size, -1, 2) ## exw [1, 441, 2]

            rpn_label = rpn_data[0].view(batch_size, -1)
            rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))

            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1,2), 0, rpn_keep)

            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
            rpn_label = Variable(rpn_label.long())

            self.rpn_loss_cls =  F.cross_entropy(rpn_cls_score, rpn_label)
            fg_cnt = torch.sum(rpn_label.data.ne(0))

            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]
            
            rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
            rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
            rpn_bbox_targets = Variable(rpn_bbox_targets)

            self.rpn_loss_box =  _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                               rpn_bbox_outside_weights, sigma=3, dim=[1,2,3,4])

            ## only 16 frames

            rpn_data_16 = self.RPN_anchor_16((rpn_cls_16.data, gt_boxes_16, im_info, gt_rois, self.sample_duration)) # time_limit = 16

            rpn_cls_16 = rpn_cls_16_reshape.permute(0, 2, 3, 1).contiguous()
            rpn_cls_16 = rpn_cls_16.view(batch_size, -1, 2) ## exw [1, 441, 2]

            rpn_label_16 = rpn_data_16[0].view(batch_size, -1)
            rpn_keep_16 = Variable(rpn_label_16.view(-1).ne(-1).nonzero().view(-1))
            rpn_cls_score_16 = torch.index_select(rpn_cls_16.view(-1,2), 0, rpn_keep_16)

            rpn_label_16 = torch.index_select(rpn_label_16.view(-1), 0, rpn_keep_16.data)
            rpn_label_16 = Variable(rpn_label_16.long())

            self.rpn_loss_cls_16 =  F.cross_entropy(rpn_cls_score_16, rpn_label_16)
            fg_cnt_16 = torch.sum(rpn_label_16.data.ne(0))

            rpn_bbox_targets_16, rpn_bbox_inside_weights_16, rpn_bbox_outside_weights_16 = rpn_data_16[1:]
            
            rpn_bbox_inside_weights_16 = Variable(rpn_bbox_inside_weights_16)
            rpn_bbox_outside_weights_16 = Variable(rpn_bbox_outside_weights_16)
            rpn_bbox_targets_16 = Variable(rpn_bbox_targets_16)

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

