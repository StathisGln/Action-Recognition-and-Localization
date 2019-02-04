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

        ## convolutions with kernels 16,8,4
        self.RPN_time_16 = nn.Conv3d(512, 512, (16,3,3), stride=1, padding=(0,1,1), bias=True).cuda()
        self.RPN_time_8  = nn.Conv3d(512, 512, (8,3,3),  stride=1, padding=(0,1,1), bias=True).cuda()
        self.RPN_time_4  = nn.Conv3d(512, 512, (4,3,3),  stride=1, padding=(0,1,1), bias=True).cuda()

        # define bg/fg classifcation score layer for each kernel 
        self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios) * 2 # 2(bg/fg) * 9 (anchors)

        self.RPN_cls_score_16 = nn.Conv2d(512, self.nc_score_out, 1, 1, 0).cuda()
        self.RPN_cls_score_8 = nn.Conv2d(512, self.nc_score_out, 1, 1, 0).cuda()
        self.RPN_cls_score_4 = nn.Conv2d(512, self.nc_score_out, 1, 1, 0).cuda()

        # define anchor box offset prediction layer
        self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * 4 # 4(coords) * 9 (anchors)

        self.RPN_bbox_pred_16 = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0).cuda()
        self.RPN_bbox_pred_8 = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0).cuda()
        self.RPN_bbox_pred_4 = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0).cuda()

        # # define bg/fg classifcation score layer
        # self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios) * 2 # 2(bg/fg) * 9 (anchors)
        # self.RPN_cls_score_16 = nn.Conv2d(512, self.nc_score_out, 1, 1, 0).cuda()
        # self.RPN_cls_score_8 = nn.Conv2d(512, self.nc_score_out, 1, 1, 0).cuda()
        # self.RPN_cls_score_4 = nn.Conv2d(512, self.nc_score_out, 1, 1, 0).cuda()

        # # define anchor box offset prediction layer
        # self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * 4 # 4(coords) * 9 (anchors)
        # self.RPN_bbox_pred_16 = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0).cuda()
        # self.RPN_bbox_pred_8 = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0).cuda()
        # self.RPN_bbox_pred_4 = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0).cuda()

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

        rpn_conv1 = F.relu(self.RPN_Conv(base_feat), inplace=True) # 3d convolution

        feat_time_16 = F.relu(self.RPN_time_16(rpn_conv1), inplace=True)
        feat_time_8 = F.relu(self.RPN_time_8(rpn_conv1), inplace=True)
        feat_time_4 = F.relu(self.RPN_time_4(rpn_conv1), inplace=True)


        ## permute features to the batch_size

        feat_time_16 = feat_time_16.permute(0,2,1,3,4).view(-1,512,7,7)
        feat_time_8 = feat_time_8.permute(0,2,1,3,4).view(-1,512,7,7)
        feat_time_4 = feat_time_4.permute(0,2,1,3,4).view(-1,512,7,7)

        # get rpn classification score
        rpn_cls_score_16 = self.RPN_cls_score_16(feat_time_16)
        rpn_cls_score_8 = self.RPN_cls_score_16(feat_time_8)
        rpn_cls_score_4 = self.RPN_cls_score_16(feat_time_4)

        rpn_cls_score_reshape_16 = self.reshape(rpn_cls_score_16, 2)
        rpn_cls_prob_reshape_16 = F.softmax(rpn_cls_score_reshape_16, 1)
        rpn_cls_prob_16 = self.reshape(rpn_cls_prob_reshape_16, self.nc_score_out)

        rpn_cls_score_reshape_8 = self.reshape(rpn_cls_score_8, 2)
        rpn_cls_prob_reshape_8 = F.softmax(rpn_cls_score_reshape_8, 1)
        rpn_cls_prob_8 = self.reshape(rpn_cls_prob_reshape_8, self.nc_score_out)

        rpn_cls_score_reshape_4 = self.reshape(rpn_cls_score_4, 2)
        rpn_cls_prob_reshape_4 = F.softmax(rpn_cls_score_reshape_4, 1)
        rpn_cls_prob_4 = self.reshape(rpn_cls_prob_reshape_4, self.nc_score_out)


        # get rpn offsets to the anchor boxes
        rpn_bbox_pred_16 = self.RPN_bbox_pred_16(feat_time_16)
        rpn_bbox_pred_8 = self.RPN_bbox_pred_8(feat_time_8)
        rpn_bbox_pred_4 = self.RPN_bbox_pred_4(feat_time_4)

        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'

        rois_16 = self.RPN_proposal((rpn_cls_prob_16.data, rpn_bbox_pred_16.data,
                                 im_info, cfg_key))
        rois_8 = self.RPN_proposal((rpn_cls_prob_8.data, rpn_bbox_pred_8.data,
                                 im_info, cfg_key))
        rois_4 = self.RPN_proposal((rpn_cls_prob_4.data, rpn_bbox_pred_4.data,
                                 im_info, cfg_key))
        
        print('rois_16.shape {}, rois_8.shape {}, rois_4.shape {} '.format(rois_16.shape,rois_8.shape,rois_4.shape))
        self.rpn_loss_cls_16 = 0
        self.rpn_loss_box_16 = 0

        # generating training labels a# nd build the rpn loss
        if self.training:

            assert gt_boxes is not None
            print('gt_boxes :',gt_boxes)
            print('gt_boxes.shape :', gt_boxes.shape)

            duration = (gt_boxes[:,:,5] - gt_boxes[:,:,2] + 1).tolist()
            print('duration :',duration)
            print('len(duration) :', len(duration) ,' and gt_bboxes.size(1) :',gt_boxes.size(1))
            for tube_idx in range(len(duration)): # only for 1 video until now
                ## to change 
                # ### for 16 frames tube
                gt_tube =  gt_boxes[0,tube_idx,:].unsqueeze(0).unsqueeze(0)
                # print('gt_tube.shape :',gt_tube.shape )
                rpn_data_16 = self.RPN_anchor_target((rpn_cls_score_16.data, gt_tube, im_info, num_boxes, 16)) # time_limit = 16
                # rpn_data_8  = self.RPN_anchor_target((rpn_cls_score_8.data , gt_boxes, im_info, num_boxes))
                # rpn_data_4  = self.RPN_anchor_target((rpn_cls_score_4.data , gt_boxes, im_info, num_boxes))
                # print('rpn_data[0]_16 :',rpn_data_16[0] )
                # print('rpn_data[0]_16.shape :',rpn_data_16[0].shape )
                
                # # compute classification loss
                # print('rpn_cls_score_16.shape :', rpn_cls_score_16.shape )
                # print('rpn_cls_score_16 :', rpn_cls_score_16.shape )
                # print(' rpn_cls_score_reshape_16.shape : ',  rpn_cls_score_reshape_16.shape)
                # print(' rpn_cls_score_reshape_16 : ',  rpn_cls_score_reshape_16)
                rpn_cls_score_16 = rpn_cls_score_reshape_16.permute(0, 2, 3, 1).contiguous()
                # print('rpn_cls_score_16 :', rpn_cls_score_16 )
                # print('rpn_cls_score_16.shape :', rpn_cls_score_16.shape )  ## exw [1, 63,7,2]
                rpn_cls_score_16 = rpn_cls_score_16.view(batch_size, -1, 2) ## exw [1, 441, 2]
                # print('rpn_cls_score_16.shape :', rpn_cls_score_16.shape )

                rpn_label_16 = rpn_data_16[0].view(batch_size, -1)
                # print('rpn_label_16.shape :',rpn_label_16.shape)
                # print('labels: ',rpn_label_16)
                rpn_keep_16 = Variable(rpn_label_16.view(-1).ne(-1).nonzero().view(-1))
                # print('rpn_keep_16 :',rpn_keep_16)

                # print('rpn_cls_score_16.shape before index_select:', rpn_cls_score_16.shape)
                # print('rpn_cls_score_16.shape before index_select:', rpn_cls_score_16.view(-1,2).shape)
                rpn_cls_score_16 = torch.index_select(rpn_cls_score_16.view(-1,2), 0, rpn_keep_16)
                # print('rpn_cls_score_16.shape after index_select:', rpn_cls_score_16.shape)

                print('rpn_label_16.shape before index_select: ', rpn_label_16.shape)
                print('rpn_label_16.shape before index_select: ', rpn_label_16.view(-1).shape)
                rpn_label_16 = torch.index_select(rpn_label_16.view(-1), 0, rpn_keep_16.data)
                print('rpn_label_16.shape after  index_select: ', rpn_label_16.view(-1).shape)
                
                rpn_label_16 = Variable(rpn_label_16.long())
                self.rpn_loss_cls_16 =  F.cross_entropy(rpn_cls_score_16, rpn_label_16)
                # print('rpn_cls_score_16 :', rpn_cls_score_16)
                # print('rpn_cls_label_16 :', rpn_label_16)
                # print('rpn_loss_cls_16 :',self.rpn_loss_cls_16)
                fg_cnt_16 = torch.sum(rpn_label_16.data.ne(0))

                rpn_bbox_targets_16, rpn_bbox_inside_weights_16, rpn_bbox_outside_weights_16 = rpn_data_16[1:]

                # compute bbox regression loss
                rpn_bbox_inside_weights_16 = Variable(rpn_bbox_inside_weights_16)
                rpn_bbox_outside_weights_16 = Variable(rpn_bbox_outside_weights_16)
                rpn_bbox_targets_16 = Variable(rpn_bbox_targets_16)
                # print('--------\n compute bbox regression loss \n--------')
                # print('rpn_bbox_pred_16 :', rpn_bbox_pred_16)
                # print('rpn_bbox_targets_16 :', rpn_bbox_targets_16)
                # print('rpn_bbox_inside_weights_16 :', rpn_bbox_inside_weights_16)
                # print('rpn_bbox_outside_weights_16 :', rpn_bbox_outside_weights_16)
                
                self.rpn_loss_box_16 = ( _smooth_l1_loss(rpn_bbox_pred_16, rpn_bbox_targets_16, rpn_bbox_inside_weights_16,
                                                     rpn_bbox_outside_weights_16, sigma=3, dim=[1,2,3]))

        print('self.rpn_loss_box_16 :',self.rpn_loss_box_16)
        print('self.rpn_loss_cls_16 :',self.rpn_loss_cls_16)
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

