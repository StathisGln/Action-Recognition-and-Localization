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
        self.anchor_scales = [ 8, 16, 32, 64]
        self.anchor_ratios = [0.5, 1, 2]
        self.feat_stride = [16, ]
        # self.anchor_duration = [16,8,4,3] # add
        self.anchor_duration = [sample_duration,int(sample_duration*3/4), int(sample_duration/2)] #,int(sample_duration/4)] # add 

        # # define the convrelu layers processing input feature map

        self.RPN_Conv = nn.Conv3d(self.din, self.din * 2, 3, stride=1, padding=1, bias=True)

        # define bg/fg classifcation score layer for each kernel 

        self.nc_score_out = 1 * len(self.anchor_ratios) * len(self.anchor_duration) * 2

        self.RPN_cls_score = nn.Conv3d(self.din*2, self.nc_score_out, 1, 1, 0)

        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios, self.anchor_duration)

        self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride,  self.anchor_scales, self.anchor_ratios, self.anchor_duration)

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


    def forward(self, rpn_feature_maps, im_info, gt_boxes, gt_rois):

        n_feat_maps = len(rpn_feature_maps)

        rpn_cls_scores = []
        rpn_cls_probs = []
        rpn_bbox_preds = []
        rpn_shapes = []

        for i in range(n_feat_maps):

            feat_map = rpn_feature_maps[i]
            batch_size = feat_map.size(0)

            rpn_conv1 = F.relu(self.RPN_Conv(feat_map), inplace=True) # 3d convolution

            # # ## get classification score for all anchors
            rpn_cls_score = self.RPN_cls_score(rpn_conv1)  # classification layer

            rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
            rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape,dim=1)
            rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)

            rpn_shapes.append([rpn_cls_score.size()[2], rpn_cls_score.size()[3],rpn_cls_score.size()[4]])
            rpn_cls_scores.append(rpn_cls_score.permute(0, 2, 3, 4, 1).contiguous().view(batch_size, -1, 2))
            rpn_cls_probs.append(rpn_cls_prob.permute(0, 2, 3, 4, 1).contiguous().view(batch_size, -1, 2))

        rpn_cls_score_alls = torch.cat(rpn_cls_scores, 1)
        rpn_cls_prob_alls = torch.cat(rpn_cls_probs, 1)
        rpn_bbox_pred_alls = torch.zeros((batch_size, 150480, 6)).cuda()

        n_rpn_pred = rpn_cls_score_alls.size(1)

        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'

        rois = self.RPN_proposal((rpn_cls_prob_alls.data, rpn_bbox_pred_alls.data,
                                     im_info, cfg_key,rpn_shapes))


        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

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
            rpn_data = self.RPN_anchor_target((rpn_cls_score_alls.data, gt_boxes, im_info, rpn_shapes)) # time_limit = 16
            
            rpn_label = rpn_data[0].view(batch_size, -1)

            rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))

            rpn_cls_score_alls = torch.index_select(rpn_cls_score_alls.view(-1,2), 0, rpn_keep)

            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
            rpn_label = Variable(rpn_label.long())
            self.rpn_loss_cls =  F.cross_entropy(rpn_cls_score_alls, rpn_label)

 
        return rois,0, self.rpn_loss_cls, self.rpn_loss_box,0,0
        # return rois, rois_16, self.rpn_loss_cls, self.rpn_loss_box, self.rpn_loss_cls_16, self.rpn_loss_box_16


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

