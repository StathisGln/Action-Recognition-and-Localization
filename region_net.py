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
        self.anchor_duration = [16,8,4,3] # add 

        # # define the convrelu layers processing input feature map

        self.RPN_Conv = nn.Conv3d(self.din, 512, 3, stride=1, padding=1, bias=True).cuda()

        # define bg/fg classifcation score layer for each kernel 
        # 2(bg/fg) * 9  (anchors) * 4 (duration : 16,8,4,3)
        self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios) * len(self.anchor_duration) * 2 
        self.RPN_cls_score = nn.Conv3d(512, self.nc_score_out, 1, 1, 0).cuda()

        # define anchor box offset prediction layer
        # 6(coords:x1,y1,t1) * 9 (anchors)  * 4 (duration)
        self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * len(self.anchor_duration) * 6
        self.RPN_bbox_pred = nn.Conv3d(512, self.nc_bbox_out, 1, 1, 0) # for regression

        ## temporal regression
        # self.RPN_temporal_pred = nn.Conv3d(
        # define proposal layer
        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        # define anchor target layer
        self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        print('input_shape.shape :',input_shape)
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3],
            input_shape[4]
        )
        return x

    def forward(self, base_feat, im_info, gt_boxes, rois, num_boxes):

        batch_size = base_feat.size(0)

        rpn_conv1 = F.relu(self.RPN_Conv(base_feat), inplace=True) # 3d convolution

        print('rpn_conv1.shape :',rpn_conv1.shape)

        # ## get classification score for all anchors
        rpn_cls_score = self.RPN_cls_score(rpn_conv1)  # classification layer
        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv1) # regression layer

        print('rpn_cls_score shape : ', rpn_cls_score.shape)
        print('rpn_bbox_pred shape : ', rpn_bbox_pred.shape)

        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)

        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'

        rois = self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data,
                                     im_info, cfg_key,16))

        self.rpn_loss_cls_16 = 0
        self.rpn_loss_box_16 = 0

        # generating training labels a# nd build the rpn loss
        if self.training:

            assert gt_boxes is not None

            duration = (gt_boxes[:,:,5] - gt_boxes[:,:,2] + 1).tolist()

            print('duration :',duration)


            for tube_idx in range(1): # only for 1 video until now

                gt_tube =  gt_boxes[0,tube_idx,:].unsqueeze(0).unsqueeze(0)
                tube_rois = rois[tube_idx,:,:].unsqueeze(1)
                # print('tube_rois.shape :,', tube_rois.shape)

                # ### for 16 frames tube

                rpn_data_16 = self.RPN_anchor_target((rpn_cls_score_16.data, gt_tube, im_info, tube_rois, num_boxes, 16)) # time_limit = 16

                rpn_cls_score_16 = rpn_cls_score_reshape_16.permute(0, 2, 3, 1).contiguous()
                rpn_cls_score_16 = rpn_cls_score_16.view(batch_size, -1, 2) ## exw [1, 441, 2]

                rpn_label_16 = rpn_data_16[0].view(batch_size, -1)
                rpn_keep_16 = Variable(rpn_label_16.view(-1).ne(-1).nonzero().view(-1))

                rpn_cls_score_16 = torch.index_select(rpn_cls_score_16.view(-1,2), 0, rpn_keep_16)

                rpn_label_16 = torch.index_select(rpn_label_16.view(-1), 0, rpn_keep_16.data)
                rpn_label_16 = Variable(rpn_label_16.long())

                # print('rpn_cls_score_16.shape :',rpn_cls_score_16.shape)
                # print('rpn_label_16.shape :',rpn_label_16.shape)

                self.rpn_loss_cls_16 =  F.cross_entropy(rpn_cls_score_16, rpn_label_16)
                print('self.rpn_loss_cls_16 :',self.rpn_loss_cls_16)
                fg_cnt_16 = torch.sum(rpn_label_16.data.ne(0))

                rpn_bbox_frame_targets_16, rpn_bbox_frame_inside_weights_16, rpn_bbox_frame_outside_weights_16 = rpn_data_16[1:]

                rpn_bbox_inside_weights_16 = Variable(rpn_bbox_frame_inside_weights_16)
                rpn_bbox_outside_weights_16 = Variable(rpn_bbox_frame_outside_weights_16)
                rpn_bbox_targets_16 = Variable(rpn_bbox_frame_targets_16)
                # print('rpn_bbox_frame_16.shape ',rpn_bbox_frame_16.shape)
                # print('rpn_bbox_frame_targets_16.shape ',rpn_bbox_frame_targets_16.shape)
                # print('rpn_bbox_frame_inside_weights_16.shape :',rpn_bbox_frame_inside_weights_16.shape)
                # print('rpn_bbox_frame_outside_weights_16.shape :',rpn_bbox_frame_outside_weights_16.shape)
                self.rpn_loss_box_16 =  _smooth_l1_loss(rpn_bbox_frame_16, rpn_bbox_frame_targets_16, rpn_bbox_inside_weights_16,
                                                                   rpn_bbox_outside_weights_16, sigma=3, dim=[1,2,3])
                # print('self.rpn_loss_box_16 :',self.rpn_loss_box_16)
                # print('self.rpn_loss_box_16 :',self.rpn_loss_box_16)

                # print('----------\nEKSWWWW 16\n----------')
                # # #### for 8 frames tube

                # rpn_data_8  = self.RPN_anchor_target((rpn_cls_score_8.data , gt_tube, im_info, tube_rois, num_boxes,  8))

                # rpn_cls_score_8 = rpn_cls_score_reshape_8.permute(0, 2, 3, 1).contiguous()
                # rpn_cls_score_8 = rpn_cls_score_8.view(batch_size, -1, 2) ## exw [1, 441, 2]

                # rpn_label_8 = rpn_data_8[0].view(batch_size, -1)
                # rpn_keep_8 = Variable(rpn_label_8.view(-1).ne(-1).nonzero().view(-1))

                # rpn_cls_score_8 = torch.index_select(rpn_cls_score_8.view(-1,2), 0, rpn_keep_8)
                # rpn_label_8 = torch.index_select(rpn_label_8.view(-1), 0, rpn_keep_8.data)
                # rpn_label_8 = Variable(rpn_label_8.long())

                # self.rpn_loss_cls_8  =  F.cross_entropy(rpn_cls_score_8, rpn_label_8)

                # fg_cnt_8  = torch.sum(rpn_label_8.data.ne(0))

                # rpn_bbox_frame_targets_8 , rpn_bbox_frame_inside_weights_8 , rpn_bbox_frame_outside_weights_8  = rpn_data_8[1:]              

                # rpn_bbox_inside_weights_8 = Variable(rpn_bbox_frame_inside_weights_8)
                # rpn_bbox_outside_weights_8 = Variable(rpn_bbox_frame_outside_weights_8)
                # rpn_bbox_targets_8 = Variable(rpn_bbox_frame_targets_8)

                # self.rpn_loss_box_8  =  _smooth_l1_loss(rpn_bbox_frame_8, rpn_bbox_frame_targets_8, rpn_bbox_inside_weights_8,
                #                                     rpn_bbox_outside_weights_8, sigma=3, dim=[1,2,3])

                # print('self.rpn_loss_box_8 :',self.rpn_loss_box_8)

                # print('----------\nEKSWWWW 8\n----------')
                # # #### for 4 frames tube

                # rpn_data_4  = self.RPN_anchor_target((rpn_cls_score_4.data , gt_tube, im_info, tube_rois, num_boxes,  4))

                # rpn_cls_score_4 = rpn_cls_score_reshape_4.permute(0, 2, 3, 1).contiguous()
                # rpn_cls_score_4 = rpn_cls_score_4.view(batch_size, -1, 2) ## exw [1, 441, 2]

                # rpn_label_4 = rpn_data_4[0].view(batch_size, -1)
                # rpn_keep_4 = Variable(rpn_label_4.view(-1).ne(-1).nonzero().view(-1))

                # rpn_cls_score_4 = torch.index_select(rpn_cls_score_4.view(-1,2), 0, rpn_keep_4)
                # rpn_label_4 = torch.index_select(rpn_label_4.view(-1), 0, rpn_keep_4.data)
                # rpn_label_4 = Variable(rpn_label_4.long())

                # self.rpn_loss_cls_4  =  F.cross_entropy(rpn_cls_score_4, rpn_label_4)

                # fg_cnt_4  = torch.sum(rpn_label_4.data.ne(0))

                # rpn_bbox_frame_targets_4 , rpn_bbox_frame_inside_weights_4 , rpn_bbox_frame_outside_weights_4  = rpn_data_4[1:]

                # rpn_bbox_inside_weights_4 = Variable(rpn_bbox_frame_inside_weights_4)
                # rpn_bbox_outside_weights_4 = Variable(rpn_bbox_frame_outside_weights_4)
                # rpn_bbox_targets_4 = Variable(rpn_bbox_frame_targets_4)

                # self.rpn_loss_box_4  =  _smooth_l1_loss(rpn_bbox_frame_4, rpn_bbox_frame_targets_4, rpn_bbox_inside_weights_4,
                #                                     rpn_bbox_outside_weights_4, sigma=3, dim=[1,2,3])

                # print('self.rpn_loss_box_4 :',self.rpn_loss_box_4)

                # print('----------\nEKSWWWW 4\n----------')

                # self.rpn_loss_cls = self.rpn_loss_cls_16 + self.rpn_loss_cls_8 + self.rpn_loss_cls_4
                # self.rpn_loss_box = self.rpn_loss_box_16 + self.rpn_loss_box_8 + self.rpn_loss_box_4

                # # compute bbox regression loss
                # print('self.rpn_loss_cls :',self.rpn_loss_cls)
                # print('self.rpn_loss_box :',self.rpn_loss_box)

        return rois_16, self.rpn_loss_cls_16, self.rpn_loss_box_16


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

