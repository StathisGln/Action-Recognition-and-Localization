from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from config import cfg
from proposal_layer import _ProposalLayer
from anchor_target_layer import _AnchorTargetLayer
from net_utils import _smooth_l1_loss

import numpy as np
import math
import pdb
import time

class _RPN(nn.Module):
    """ region proposal network """
    def __init__(self, din):
        super(_RPN, self).__init__()
        
        # self.din = din  # get depth of input feature map, e.g., 512
        # self.anchor_scales = cfg.ANCHOR_SCALES
        # self.anchor_ratios = cfg.ANCHOR_RATIOS
        # self.feat_stride = [16]

        self.din = din  # get depth of input feature map, e.g., 512
        self.anchor_scales = [4, 8, 16 ]
        # self.anchor_scales = [64, 128,256, 512 ] # according to DetectAndTrack
        self.anchor_ratios = [0.5, 1, 2]
        self.feat_stride = [16, ]

        # # define the convrelu layers processing input feature map

        self.RPN_Conv = nn.Conv3d(self.din, 512, 3, stride=1, padding=1, bias=True).cuda()
        # self.RPN_Conv_2 = nn.Conv3d(512, 512, 3, stride=(2,1,1), padding=1, bias=True).cuda()
        # self.RPN_Conv_3 = nn.Conv3d(512, 512, 3, stride=(2,1,1), padding=1, bias=True).cuda()
        # self.RPN_Conv_4 = nn.Conv3d(512, 512, 3, stride=(2,1,1), padding=1, bias=True).cuda()
        # self.RPN_Conv_5 = nn.Conv3d(512, 512, 3, stride=(2,1,1), padding=1, bias=True).cuda()

        ## check for tubes:
        self.RPN_time_16 = nn.Conv3d(512, 512, (16,3,3), stride=1, padding=(0,1,1), bias=True).cuda()
        self.RPN_time_8  = nn.Conv3d(512, 512, (8,3,3),  stride=1, padding=(0,1,1), bias=True).cuda()
        self.RPN_time_4  = nn.Conv3d(512, 512, (4,3,3),  stride=1, padding=(0,1,1), bias=True).cuda()

        # define bg/fg classifcation score layer
        self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios) * 2 # 2(bg/fg) * 9 (anchors)
        self.RPN_cls_score_16 = nn.Conv2d(512, self.nc_score_out, 1, 1, 0).cuda()
        self.RPN_cls_score_8 = nn.Conv2d(512, self.nc_score_out, 1, 1, 0).cuda()
        self.RPN_cls_score_4 = nn.Conv2d(512, self.nc_score_out, 1, 1, 0).cuda()

        # define anchor box offset prediction layer
        self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * 4 # 4(coords) * 9 (anchors)
        self.RPN_bbox_pred_16 = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0).cuda()
        self.RPN_bbox_pred_8 = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0).cuda()
        self.RPN_bbox_pred_4 = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0).cuda()

        # define bg/fg classifcation score layer
        self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios) * 2 # 2(bg/fg) * 9 (anchors)
        self.RPN_cls_score_16 = nn.Conv2d(512, self.nc_score_out, 1, 1, 0).cuda()
        self.RPN_cls_score_8 = nn.Conv2d(512, self.nc_score_out, 1, 1, 0).cuda()
        self.RPN_cls_score_4 = nn.Conv2d(512, self.nc_score_out, 1, 1, 0).cuda()

        # define anchor box offset prediction layer
        self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * 4 # 4(coords) * 9 (anchors)
        self.RPN_bbox_pred_16 = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0).cuda()
        self.RPN_bbox_pred_8 = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0).cuda()
        self.RPN_bbox_pred_4 = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0).cuda()

        # define proposal layer
        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        # define anchor target layer
        self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x

    def forward(self, base_feat, im_info, gt_boxes, num_boxes):

        batch_size = base_feat.size(0)
        print('batch_size :', batch_size)
        # # return feature map after convrelu layer
        # rpn_conv1 = F.relu(self.RPN_Conv(base_feat), inplace=True) # 3d convolution
        # rpn_conv2 = F.relu(self.RPN_Conv_2(rpn_conv1), inplace=True)
        # rpn_conv3 = F.relu(self.RPN_Conv_3(rpn_conv2), inplace=True)
        # rpn_conv4 = F.relu(self.RPN_Conv_4(rpn_conv3), inplace=True)
        # rpn_conv5 = F.relu(self.RPN_Conv_5(rpn_conv4), inplace=True)


        # print('base_feat shape :', base_feat.shape)

        # print('rpn_conv2 shape :', rpn_conv2.shape)
        # print('rpn_conv3 shape :', rpn_conv3.shape)
        # print('rpn_conv4 shape :', rpn_conv4.shape)
        # print('rpn_conv5 shape :', rpn_conv5.shape)

        rpn_conv1 = F.relu(self.RPN_Conv(base_feat), inplace=True) # 3d convolution
        print('rpn_conv1 shape :', rpn_conv1.shape)

        feat_time_16 = F.relu(self.RPN_time_16(rpn_conv1), inplace=True)
        print('feat_time_16.shape :',feat_time_16.shape)
        feat_time_8 = F.relu(self.RPN_time_8(rpn_conv1), inplace=True)
        feat_time_4 = F.relu(self.RPN_time_4(rpn_conv1), inplace=True)

        print('feat_time_16.shape :',feat_time_16.shape)
        print('feat_time_8.shape :',feat_time_8.shape)
        print('feat_time_4.shape :',feat_time_4.shape)
        # rpn_conv5 = rpn_conv5.squeeze(2)
        # print('AFTER SQUEEZE :rpn_conv5 shape :', rpn_conv5.shape)

        # # get rpn classification score
        # rpn_cls_score = self.RPN_cls_score(rpn_conv5)

        # rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        # rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)
        # rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)

        # # get rpn offsets to the anchor boxes
        # rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv5)

        # # proposal layer
        # cfg_key = 'TRAIN' if self.training else 'TEST'

        # rois = self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data,
        #                          im_info, cfg_key))

        # self.rpn_loss_cls = 0
        # self.rpn_loss_box = 0

        # # generating training labels and build the rpn loss
        # if self.training:
        #     assert gt_boxes is not None

        #     rpn_data = self.RPN_anchor_target((rpn_cls_score.data, gt_boxes, im_info, num_boxes))
        #     print('rpn_data[0] :',rpn_data[0] )
        #     print('rpn_data[0].shape :',rpn_data[0].shape )
        #     # compute classification loss
        #     rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
        #     rpn_label = rpn_data[0].view(batch_size, -1)
        #     rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))
        #     rpn_cls_score = torch.index_select(rpn_cls_score.view(-1,2), 0, rpn_keep)
        #     rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
        #     rpn_label = Variable(rpn_label.long())
        #     self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)
        #     fg_cnt = torch.sum(rpn_label.data.ne(0))

        #     rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]

        #     # compute bbox regression loss
        #     rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
        #     rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
        #     rpn_bbox_targets = Variable(rpn_bbox_targets)

        #     self.rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
        #                                                     rpn_bbox_outside_weights, sigma=3, dim=[1,2,3])

        # return rois, self.rpn_loss_cls, self.rpn_loss_box

if __name__ == '__main__':

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

