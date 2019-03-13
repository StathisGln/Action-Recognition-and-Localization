from __future__ import absolute_import, print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from config import cfg
from proposal_layer import _ProposalLayer
from anchor_target_layer_mine import _AnchorTargetLayer
from reg_target import _Regression_TargetLayer

from net_utils import _smooth_l1_loss

import numpy as np

class _Regression_Layer(nn.Module):
    """
    This class gets as input, tubes and feats and finds the regression
    """
    def __init__(self, din, sample_duration):

        super(_Regression_Layer, self).__init__()
        self.din = din
        self.sample_duration = sample_duration
        
        self.Conv = nn.Conv2d(self.din, din*2, 3, stride=1, padding=1, bias=True)

        self.avg_pool = nn.AvgPool2d((7, 7), stride=1)
        self.bbox_pred = nn.Linear(din*2,4)

        self.reg_target = _Regression_TargetLayer( self.sample_duration)

    def forward(self, base_feat, tubes, gt_rois):

        batch_size = base_feat.size(0)
        
        offset = torch.arange(0,self.sample_duration)

        ## modify tubes and rois_label
        tubes = tubes.unsqueeze(-2).expand(tubes_.size(0),tubes_.size(1),self.sample_duration,7).contiguous()
        tubes[..., 0] = offset
        tubes = tubes.permute(0,2,1,3).contiguous().view(batch_size, -1,7)

        if self.training:
            rois, labels, \
            bbox_targets, bbox_inside_weights,\
            bbox_outside_weights = self.reg_target(tubes, gt_rois, torch.Tensor(3))

            
        ## modify feat
        base_feat = base_feat.permute(0,2,1,3,4).contiguous().view(-1,base_feat.size(1),base_feat.size(3),base_feat.size(4))
        conv1_feats = self.Conv(base_feat)

        feat = self.avg_pool(conv1_feats).squeeze()
        bbox_pred = self.bbox_pred(feat) # regression layer

        if self.training:
            # bounding box regression L1 loss
            rois_loss_bbox = _smooth_l1_loss(bbox_pred, bbox_target, bbox_inside_ws, bbox_outside_ws)

        print(rois_loss_bbox)
        # print('bbox_pred :',bbox_pred.shape)
    
if __name__ == '__main__':

    sample_dur = 2


    t = torch.arange(0,280).view(4,5,2,7).unsqueeze(-1).expand(4,5,2,7,7).float()
    # print('t.shape :',t.shape)
    # rois_label = torch.Tensor([[ 2.,  7., 20., 15.]])
    tubes_ = torch.Tensor([[[ 0.,  4.,  3.,  6.,  8.,  3., 10.],
                            [ 0.,  3.,  6.,  5.,  6.,  5.,  9.],
                            [ 0.,  3.,  8., 10.,  4.,  9., 15.],
                            [ 0., 10.,  7.,  5.,  13.,  9.,  8.]]]).expand(4,4,7)

    print('tubes.shape :',tubes_.shape)

    gt_rois = torch.Tensor([[[[ 10, 11, 22, 23, 10], [ 0,  0,  0,  0, -1]],
                        [[ 22, 25, 32, 55, 11], [32, 12, 78, 32, 10]],
                         [[  0,  0,  0,  0, -1], [53, 42, 98, 60, 10]]]]).expand(4,3,2,5)

    reg = _Regression_Layer(5,2)
    reg(t,tubes_, gt_rois)

    # # print(feats.shape)
    # # print('first feat, first frame   :',t[0,0,0])
    # # print('first feat, second frames :',t[0,0,1])
    # # print('second feat, first frame  :',t[0,1,0])
    # # print('after')
    # # print('first frame, first feat :',feats[0,0])
    # # print('first frame, sencd feat :',feats[0,1])
    # # print('sencd frame, first feat :',feats[1,0])
