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

class _Regression_Layer(nn.Module):
    """
    This class gets as input, tubes and feats and finds the regression
    """
    def __init__(self, din):

        self.din = din

        self.Conv = nn.Conv2d(self.din, 512, 3, stride=1, padding=1, bias=True).cuda()

        self.n_bbox_out = 4
        self.RPN_bbox_pred = nn.Conv2d(512, self.n_bbox_out, 1, 1, 0).cuda() # for regression

        # # define proposal layer
        # self.RPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios, self.anchor_duration).cuda()

        # # define anchor target layer
        # self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios, self.anchor_duration).cuda()

    def forward(self, base_feat, tubes, gt_rois):

        batch_size = base_feat.size(0)

        conv1_feats = F.relu(self.Conv(base_feat), inplace=True)

        bbox_pred = self.RPN_bbox_pred(covn1_feats) # regression layer

        proposals = bbox_transform_inv_3d(tubes, bbox_frame, batch_size) # proposals have 441 * time_dim shape

    
