from __future__ import absolute_import
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
import yaml
# from config import cfg
from conf import conf
from generate_anchors import generate_anchors_all_pyramids
# from generate_3d_anchors import generate_anchors, generate_anchors_all_pyramids

from bbox_transform import bbox_transform_inv, clip_boxes_3d, clip_boxes_batch, bbox_transform_inv_3d

import pdb

DEBUG = False

class _ProposalLayer(nn.Module):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def __init__(self, feat_stride, scales, ratios, time_dim):
        super(_ProposalLayer, self).__init__()

        # self.sample_duration = time_dim
        self._anchor_ratios = ratios
        self._feat_stride = feat_stride
        self._fpn_scales = np.array(scales)
        self._fpn_feature_strides = np.array([4, 8, 16, 32, 64])
        self._fpn_anchor_stride  = 1
        self._time_dim = time_dim

    def forward(self, input):

        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)


        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs

        scores = input[0][:, :, 1]
        bbox_frame = input[1]
        im_info = input[2]
        cfg_key = input[3]
        feat_shapes = input[4]        

        pre_nms_topN  = conf[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = conf[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh    = conf[cfg_key].RPN_NMS_THRESH
        min_size      = conf[cfg_key].RPN_MIN_SIZE

        batch_size = bbox_frame.size(0)

        ##################
        # Create anchors #
        ##################

        anchors = torch.from_numpy(generate_anchors_all_pyramids(self._fpn_scales, self._anchor_ratios, self._time_dim,
                feat_shapes, self._fpn_feature_strides, self._fpn_anchor_stride)).type_as(scores)

        num_anchors = anchors.size(0)
        anchors = anchors.view(1, num_anchors, 6).expand(batch_size, num_anchors, 6)

        # Convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv_3d(anchors, bbox_frame, batch_size)

        # 2. clip predicted boxes to image
        ## if any dimension exceeds the dims of the original image, clamp_ them

        proposals = clip_boxes_3d(proposals, im_info, batch_size)

        scores_keep = scores
        proposals_keep = proposals

        _, order = torch.sort(scores, 1, True)
        
        output = scores.new(batch_size, post_nms_topN, 8).zero_()

        for i in range(batch_size):
            # # 3. remove predicted boxes with either height or width < threshold
            # # (NOTE: convert min_size to input image scale stored in im_info[2])
            proposals_single = proposals_keep[i]
            scores_single = scores_keep[i]
            order_single = order[i]

            proposals_single = proposals_single[order_single, :]
            scores_single = scores_single[order_single].view(-1,1)
            proposals_single = proposals_single[:post_nms_topN, :]
            scores_single = scores_single[:post_nms_topN]
            
            # adding score at the end.
            num_proposal = proposals_single.size(0)
            output[i,:num_proposal,0] = i
            output[i,:num_proposal,1:7] = proposals_single
            output[i,:num_proposal,7] = scores_single.squeeze()

        return output

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _filter_boxes(self, boxes, min_size):
        """Remove all boxes with any side smaller than min_size."""
        ws = boxes[:, :, 2] - boxes[:, :, 0] + 1
        hs = boxes[:, :, 3] - boxes[:, :, 1] + 1
        keep = ((ws >= min_size.view(-1,1).expand_as(ws)) & (hs >= min_size.view(-1,1).expand_as(hs)))
        return keep
