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
        self.RPN_cls_score = nn.Conv3d(512, self.nc_score_out, 1, 1, 0).cuda()

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
        self.keep=self.RPN_cls_score.weight.data.clone() # modify here
        self.init_rpn()
        
    def init_rpn(self):

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
        normal_init(self.RPN_Conv, 0, 0.01, truncated)
        normal_init(self.RPN_cls_score, 0, 0.01, truncated)
        normal_init(self.RPN_bbox_pred, 0, 0.001, truncated)


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

    def forward(self, base_feat, im_info, gt_boxes, gt_rois, num_boxes):

        batch_size = base_feat.size(0)

        if (self.RPN_cls_score.weight.data==self.keep).all(): print('same!')
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

            rpn_data = self.RPN_anchor_target((rpn_cls_score.data, gt_boxes, im_info, gt_rois, num_boxes, 16)) # time_limit = 16

            # print('rpn_cls_score.shape :',rpn_cls_score.shape) 
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
            # print('self.rpn_loss_box :',self.rpn_loss_box)
            # print('self.rpn_loss_box :',self.rpn_loss_box)


        return rois, self.rpn_loss_cls, self.rpn_loss_box


if __name__ == '__main__':


    # a good example is v_TrampolineJumping_g17_c01
    # feats = torch.rand(1,512,16,4,4).cuda()
    # feats = torch.rand(1,512,8,4,4).cuda().float()
    feats = torch.rand(1,256,16,7,7).cuda().float()

    h = torch.Tensor([240]).cuda()
    w = torch.Tensor([320]).cuda()
    gt_bboxes = torch.Tensor([[[160.7641,  70.0822, 242.5207, 175.3398,   1.]],
                             [[161.1543,  70.5410, 242.4840, 175.2963,    1.]],
                             [[161.1610,  70.5489, 242.4820, 175.2937,    1.]],
                             [[161.0852,  70.4499, 243.3874, 176.4665,    1.]],
                             [[161.0921,  70.4580, 243.3863, 176.4650,    1.]],
                             [[161.5888,  73.5920, 242.9024, 176.6015,    1.]],
                             [[161.5971,  73.5839, 242.9018, 177.6026,    1.]],
                             [[161.6053,  73.5757, 242.9014, 177.6040,    1.]],
                             [[160.7641,  70.0822, 242.5207, 175.3398,    1.]],
                             [[161.1543,  70.5410, 242.4840, 175.2963,    1.]],
                             [[161.1610,  70.5489, 242.4820, 175.2937,    1.]],
                             [[161.0852,  70.4499, 243.3874, 176.4665,    1.]],
                             [[161.0921,  70.4580, 243.3863, 176.4650,    1.]],
                             [[161.5888,  73.5920, 242.9024, 176.6015,    1.]],
                             [[161.5971,  73.5839, 242.9018, 177.6026,    1.]],
                             [[161.6053,  73.5757, 242.9014, 177.6040,    1.]]]).cuda().float()

    # gt_bboxes = torch.Tensor([[[160.7641,  70.0822, 242.5207, 175.3398,   1.]],
    #                          [[161.1543,  70.5410, 242.4840, 175.2963,    1.]],
    #                          [[161.1610,  70.5489, 242.4820, 175.2937,    1.]],
    #                          [[161.0852,  70.4499, 243.3874, 176.4665,    1.]],
    #                          [[161.0921,  70.4580, 243.3863, 176.4650,    1.]],
    #                          [[161.5888,  73.5920, 242.9024, 176.6015,    1.]],
    #                          [[161.5971,  73.5839, 242.9018, 177.6026,    1.]],
    #                          [[161.6053,  73.5757, 242.9014, 177.6040,    1.]]]).cuda().float()


    print(gt_bboxes.shape)
    print('h {}, w {}, gt_bboxes.shape {}'.format(h,w,gt_bboxes.shape))
    model = _RPN(256)
    out = model(feats,torch.Tensor([[h,w]]*gt_bboxes.size(1)).cuda(), gt_bboxes, len(gt_bboxes))

