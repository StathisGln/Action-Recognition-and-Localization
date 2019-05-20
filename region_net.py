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
        # self.anchor_scales = [1, 2, 4, 8, 16, 32 ]
        self.anchor_scales = [ 8, 16, 32, 64]
        self.anchor_ratios = [0.5, 1, 2]
        self.feat_stride = [16, ]
        self.anchor_duration = [sample_duration,int(sample_duration*3/4), int(sample_duration/2)] #,int(sample_duration/4)] # add 

        # # define the convrelu layers processing input feature map
        self.anchor_num = get_number_of_combinations(sample_duration, self.anchor_duration)
        self.RPN_Conv = nn.Conv3d(self.din, self.din, 3, stride=1, padding=1, bias=True)

        # define bg/fg classifcation score layer for each kernel 
        # 2(bg/fg) * 9  (anchors) * 3 (duration : 16,12,8)
        self.nc_score_out = 1 * len(self.anchor_ratios) * self.anchor_num * 2
        self.RPN_cls_score = nn.Conv2d(self.din, self.nc_score_out, 1, 1, 0)

        # define anchor box offset prediction layer
        # 6(coords:x1,y1,t1) * 9 (anchors)  * 2 (duration)
        self.nc_bbox_out = 1 * len(self.anchor_ratios) * self.anchor_num * self.sample_duration * 4
        self.RPN_bbox_pred = nn.Conv2d(self.din, self.nc_bbox_out, 1, 1, 0) # for regression

        # define proposal layer
        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios, self.anchor_duration,  len(self.anchor_scales) * len(self.anchor_ratios) * self.anchor_num)

        # define anchor target layer
        self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride,  self.anchor_scales, self.anchor_ratios, self.anchor_duration, self.anchor_num)

        self.avg_pool = nn.AvgPool3d((self.sample_duration,1,1), stride=1)
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

        n_map_feats = len(base_feat)

        rpn_shapes = []
        rpn_cls_score_all = []
        rpn_bbox_pred_all = []
        rpn_cls_prob_all  = []

        for i in range(n_map_feats):

            feat = base_feat[i]
            batch_size = feat.size(0)
            rpn_conv1 = F.relu(self.RPN_Conv(feat), inplace=True) # 3d convolution
            rpn_conv1 = self.avg_pool(rpn_conv1).squeeze(2)

            # # ## get classification score for all anchors
            rpn_cls_score = self.RPN_cls_score(rpn_conv1)  # classification layer
            rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv1)  # regression layer



            rpn_cls_score_reshape = self.reshape2d(rpn_cls_score, 2)
            rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)
            rpn_cls_prob = self.reshape2d(rpn_cls_prob_reshape, self.nc_score_out)
            
            rpn_shapes.append([rpn_cls_score.size()[2], rpn_cls_score.size()[3]])
            rpn_cls_score_all.append(rpn_cls_score.permute(0,2,3,1).contiguous().view(batch_size,-1,2))
            rpn_bbox_pred_all.append(rpn_bbox_pred.permute(0,2,3,1).contiguous().view(batch_size,-1,self.sample_duration*4))
            rpn_cls_prob_all.append(rpn_cls_prob.permute(0,2,3,1).contiguous().view(batch_size,-1,2))

        rpn_cls_score_all = torch.cat(rpn_cls_score_all, 1)
        rpn_cls_prob_all = torch.cat(rpn_cls_prob_all, 1)
        rpn_bbox_pred_all = torch.cat(rpn_bbox_pred_all, 1)

        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'

        rois = self.RPN_proposal((rpn_cls_prob_all.data, rpn_bbox_pred_all.data,
                                     im_info, cfg_key, rpn_shapes))

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

        # generating training labels a# nd build the rpn loss
        if self.training:

            assert gt_boxes is not None

            ## Regular data
            rpn_data = self.RPN_anchor_target((rpn_cls_score.data, gt_boxes, im_info, gt_rois, rpn_shapes)) # time_limit = 16

            rpn_label = rpn_data[0].view(batch_size, -1)
            rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))

            rpn_cls_score = torch.index_select(rpn_cls_score_all.view(-1,2), 0, rpn_keep)

            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
            rpn_label = Variable(rpn_label.long())

            self.rpn_loss_cls =  F.cross_entropy(rpn_cls_score, rpn_label)

            fg_cnt = torch.sum(rpn_label.data.ne(0))

            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]
            
            rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights.unsqueeze(2) \
                                               .expand(batch_size, rpn_bbox_inside_weights.size(1), self.sample_duration * 4))

            rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights.unsqueeze(2) \
                                                .expand(batch_size, rpn_bbox_outside_weights.size(1), self.sample_duration * 4))
            rpn_bbox_targets = Variable(rpn_bbox_targets)
            self.rpn_loss_box =  _smooth_l1_loss(rpn_bbox_pred_all, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                               rpn_bbox_outside_weights, sigma=3)

        return rois, None, self.rpn_loss_cls, self.rpn_loss_box, self.rpn_loss_cls_16, self.rpn_loss_box_16


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

