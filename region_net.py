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

        self.anchor_scales = [1, 2, 4, 8, 16]
        self.anchor_ratios = [0.5, 1, 2]
        self.feat_stride = [16, ]

        self.anchor_duration = [sample_duration,int(sample_duration*3/4), int(sample_duration/2), int(sample_duration/4)] # add 

        # # define the convrelu layers processing input feature map
        self.RPN_Conv = nn.Conv3d(self.din, self.din, 3, stride=1, padding=1, bias=True)
        # self.RPN_Conv_3_4 = nn.Conv3d(self.din, self.din, 3, stride=1, padding=1, bias=True)
        # self.RPN_Conv_2 = nn.Conv3d(self.din, self.din, 3, stride=1, padding=1, bias=True)
        # self.RPN_Conv_4 = nn.Conv3d(self.din, self.din, 3, stride=1, padding=1, bias=True)        

        self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios) *  2

        self.RPN_cls_score = nn.Conv3d(self.din, self.nc_score_out, 1, 1, 0)
        self.RPN_cls_score_3_4 = nn.Conv3d(self.din, self.nc_score_out, 1, 1, 0)
        self.RPN_cls_score_2 = nn.Conv3d(self.din, self.nc_score_out, 1, 1, 0)
        self.RPN_cls_score_4 = nn.Conv3d(self.din, self.nc_score_out, 1, 1, 0)


        # define anchor box offset prediction layer

        self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * self.sample_duration * 4
        self.nc_bbox_out_3_4 = len(self.anchor_scales) * len(self.anchor_ratios) * int(self.sample_duration * 3 / 4) * 4
        self.nc_bbox_out_2 = len(self.anchor_scales) * len(self.anchor_ratios) * int(self.sample_duration / 2 )* 4
        self.nc_bbox_out_4 = len(self.anchor_scales) * len(self.anchor_ratios) * int(self.sample_duration / 4 )* 4
        
        self.RPN_bbox_pred = nn.Conv3d(self.din, self.nc_bbox_out, 1, 1, 0) # for regression
        self.RPN_bbox_pred_3_4 = nn.Conv3d(self.din, self.nc_bbox_out_3_4, 1, 1, 0) # for regression
        self.RPN_bbox_pred_2 = nn.Conv3d(self.din, self.nc_bbox_out_2, 1, 1, 0) # for regression
        self.RPN_bbox_pred_4 = nn.Conv3d(self.din, self.nc_bbox_out_4, 1, 1, 0) # for regression

        # self.RPN_bbox_only16 = nn.Conv3d(self.din*2, self.bbox_16, (sample_duration,1,1), 1, 0) # for regression
        # define proposal layer
        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios, self.anchor_duration,  len(self.anchor_scales) * len(self.anchor_ratios) )

        # define anchor target layer
        self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride,  self.anchor_scales, self.anchor_ratios, self.anchor_duration, None)

        self.avg_pool = nn.MaxPool3d((self.sample_duration,1,1), stride=1)
        self.avg_pool_3_4 = nn.MaxPool3d((int(sample_duration*3/4),1,1), stride=1)
        self.avg_pool_2 = nn.MaxPool3d((int(sample_duration/2),1,1), stride=1)
        self.avg_pool_4 = nn.MaxPool3d((int(sample_duration/4),1,1), stride=1)

        # self.avg_pool = nn.AvgPool3d((self.sample_duration,1,1), stride=1)
        # self.avg_pool_3_4 = nn.AvgPool3d((int(sample_duration*3/4),1,1), stride=1)
        # self.avg_pool_2 = nn.AvgPool3d((int(sample_duration/2),1,1), stride=1)
        # self.avg_pool_4 = nn.AvgPool3d((int(sample_duration/4),1,1), stride=1)

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

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

        rpn_conv_avg = self.avg_pool(rpn_conv1)
        rpn_conv_avg_3_4 = self.avg_pool_3_4(rpn_conv1)
        rpn_conv_avg_2 = self.avg_pool_2(rpn_conv1)
        rpn_conv_avg_4 = self.avg_pool_4(rpn_conv1)

        rpn_cls_score = self.RPN_cls_score(rpn_conv_avg)  # classification layer
        rpn_cls_score_3_4 = self.RPN_cls_score(rpn_conv_avg_3_4)  # classification layer
        rpn_cls_score_2 = self.RPN_cls_score(rpn_conv_avg_2)  # classification layer
        rpn_cls_score_4 = self.RPN_cls_score(rpn_conv_avg_4)  # classification layer

        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv_avg)  # regression layer
        rpn_bbox_pred_3_4 = self.RPN_bbox_pred_3_4(rpn_conv_avg_3_4)  # regression layer
        rpn_bbox_pred_2 = self.RPN_bbox_pred_2(rpn_conv_avg_2)  # regression layer
        rpn_bbox_pred_4 = self.RPN_bbox_pred_4(rpn_conv_avg_4)  # regression layer

        # batch_size = 2
        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)

        rpn_cls_score_reshape_3_4 = self.reshape(rpn_cls_score_3_4, 2)
        rpn_cls_prob_reshape_3_4 = F.softmax(rpn_cls_score_reshape_3_4, 1)
        rpn_cls_prob_3_4 = self.reshape(rpn_cls_prob_reshape_3_4, self.nc_score_out)

        rpn_cls_score_reshape_2 = self.reshape(rpn_cls_score_2, 2)
        rpn_cls_prob_reshape_2 = F.softmax(rpn_cls_score_reshape_2, 1)
        rpn_cls_prob_2 = self.reshape(rpn_cls_prob_reshape_2, self.nc_score_out)

        rpn_cls_score_reshape_4 = self.reshape(rpn_cls_score_4, 2)
        rpn_cls_prob_reshape_4 = F.softmax(rpn_cls_score_reshape_4, 1)
        rpn_cls_prob_4 = self.reshape(rpn_cls_prob_reshape_4, self.nc_score_out)


        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'

        rois = self.RPN_proposal((rpn_cls_prob.data, rpn_cls_prob_3_4.data, rpn_cls_prob_2.data, rpn_cls_prob_4.data,
                                  rpn_bbox_pred.data, rpn_bbox_pred_3_4.data, rpn_bbox_pred_2.data, rpn_bbox_pred_4.data,
                                     im_info, cfg_key))

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

        # generating training labels a# nd build the rpn loss
        if self.training:

            assert gt_rois is not None

            ## Regular data
            rpn_data = self.RPN_anchor_target((rpn_cls_score.data, rpn_cls_score_3_4.data, \
                                               rpn_cls_score_2.data, rpn_cls_score_4.data, \
                                               gt_boxes, im_info,    \
                                               gt_rois)) # time_limit = 16

            ## 16 frames
            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 4,1).contiguous()
            rpn_cls_score = rpn_cls_score.view(batch_size, -1, 2) ## exw [1, 441, 2]

            rpn_label_ = rpn_data[0].view(batch_size, -1)
            rpn_keep_ = Variable(rpn_label_.view(-1).ne(-1).nonzero().view(-1))

            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1,2), 0, rpn_keep_)

            rpn_label_ = torch.index_select(rpn_label_.view(-1), 0, rpn_keep_.data)
            rpn_label_ = Variable(rpn_label_.long())

            rpn_loss_cls =  F.cross_entropy(rpn_cls_score, rpn_label_)

            ## 12 frames
            rpn_cls_score_3_4 = rpn_cls_score_reshape_3_4.permute(0, 2, 3, 4,1).contiguous()
            rpn_cls_score_3_4 = rpn_cls_score_3_4.view(batch_size, -1, 2) ## exw [1, 441, 2]

            rpn_label_3_4 = rpn_data[1].view(batch_size, -1)

            rpn_keep_3_4 = Variable(rpn_label_3_4.view(-1).ne(-1).nonzero().view(-1))
            
            rpn_cls_score_3_4 = torch.index_select(rpn_cls_score_3_4.view(-1,2), 0, rpn_keep_3_4)

            rpn_label_3_4 = torch.index_select(rpn_label_3_4.view(-1), 0, rpn_keep_3_4.data)
            rpn_label_3_4 = Variable(rpn_label_3_4.long())

            rpn_loss_cls_3_4 =  F.cross_entropy(rpn_cls_score_3_4, rpn_label_3_4)

            ## 8 frames
            rpn_cls_score_2 = rpn_cls_score_reshape_2.permute(0, 2, 3, 4,1).contiguous()
            rpn_cls_score_2 = rpn_cls_score_2.view(batch_size, -1, 2) ## exw [1, 441, 2]

            rpn_label_2 = rpn_data[2].view(batch_size, -1)
            rpn_keep_2 = Variable(rpn_label_2.view(-1).ne(-1).nonzero().view(-1))

            rpn_cls_score_2 = torch.index_select(rpn_cls_score_2.view(-1,2), 0, rpn_keep_2)

            rpn_label_2 = torch.index_select(rpn_label_2.view(-1), 0, rpn_keep_2.data)
            rpn_label_2 = Variable(rpn_label_2.long())

            rpn_loss_cls_2 =  F.cross_entropy(rpn_cls_score_2, rpn_label_2)

            ## 4 frames
            rpn_cls_score_4 = rpn_cls_score_reshape_4.permute(0, 2, 3, 4,1).contiguous()
            rpn_cls_score_4 = rpn_cls_score_4.view(batch_size, -1, 2) ## exw [1, 441, 2]

            rpn_label_4 = rpn_data[3].view(batch_size, -1)
            rpn_keep_4 = Variable(rpn_label_4.view(-1).ne(-1).nonzero().view(-1))

            rpn_cls_score_4 = torch.index_select(rpn_cls_score_4.view(-1,2), 0, rpn_keep_4)

            rpn_label_4 = torch.index_select(rpn_label_4.view(-1), 0, rpn_keep_4.data)
            rpn_label_4 = Variable(rpn_label_4.long())

            rpn_loss_cls_4 =  F.cross_entropy(rpn_cls_score_4, rpn_label_4)

            # Rest stuff
            rpn_bbox_targets_, rpn_bbox_targets_3_4, rpn_bbox_targets_2, rpn_bbox_targets_4, \
            rpn_bbox_inside_weights_, rpn_bbox_inside_weights_3_4, rpn_bbox_inside_weights_2,\
            rpn_bbox_inside_weights_4, \
            rpn_bbox_outside_weights_, rpn_bbox_outside_weights_3_4, rpn_bbox_outside_weights_2,\
            rpn_bbox_outside_weights_4 = rpn_data[4:]

            rpn_bbox_inside_weights_ = Variable(rpn_bbox_inside_weights_)
            rpn_bbox_outside_weights_ = Variable(rpn_bbox_outside_weights_)
            rpn_bbox_targets_ = Variable(rpn_bbox_targets_)

            rpn_bbox_inside_weights_3_4 = Variable(rpn_bbox_inside_weights_3_4)
            rpn_bbox_outside_weights_3_4 = Variable(rpn_bbox_outside_weights_3_4)
            rpn_bbox_targets_3_4 = Variable(rpn_bbox_targets_3_4)

            rpn_bbox_inside_weights_2 = Variable(rpn_bbox_inside_weights_2)
            rpn_bbox_outside_weights_2 = Variable(rpn_bbox_outside_weights_2)
            rpn_bbox_targets_2 = Variable(rpn_bbox_targets_2)

            rpn_bbox_inside_weights_4 = Variable(rpn_bbox_inside_weights_4)
            rpn_bbox_outside_weights_4 = Variable(rpn_bbox_outside_weights_4)
            rpn_bbox_targets_4 = Variable(rpn_bbox_targets_4)

            rpn_loss_box =  _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets_, rpn_bbox_inside_weights_,
                                                               rpn_bbox_outside_weights_, sigma=3,dim=[1,2,3,4])
            rpn_loss_box_3_4 =  _smooth_l1_loss(rpn_bbox_pred_3_4, rpn_bbox_targets_3_4, rpn_bbox_inside_weights_3_4,
                                                               rpn_bbox_outside_weights_3_4, sigma=3, dim=[1,2,3,4])
            rpn_loss_box_2 =  _smooth_l1_loss(rpn_bbox_pred_2, rpn_bbox_targets_2, rpn_bbox_inside_weights_2,
                                                               rpn_bbox_outside_weights_2, sigma=3, dim=[1,2,3,4])
            rpn_loss_box_4 =  _smooth_l1_loss(rpn_bbox_pred_4, rpn_bbox_targets_4, rpn_bbox_inside_weights_4,
                                                               rpn_bbox_outside_weights_4, sigma=3, dim=[1,2,3,4])


            self.rpn_loss_cls = rpn_loss_cls + rpn_loss_cls_3_4 + rpn_loss_cls_2 + rpn_loss_cls_4
            self.rpn_loss_box = rpn_loss_box + rpn_loss_box_3_4 + rpn_loss_box_2 + rpn_loss_box_4

            
        return rois, None, self.rpn_loss_cls, self.rpn_loss_box, None, None


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

