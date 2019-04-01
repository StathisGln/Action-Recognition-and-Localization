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
from conf import conf
from generate_3d_anchors import generate_anchors
from bbox_transform import bbox_transform_inv, clip_boxes, clip_boxes_batch

import pdb

DEBUG = False

class _ProposalLayer_xy(nn.Module):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def __init__(self, feat_stride,  scales, ratios, time_dim):
        super(_ProposalLayer_xy, self).__init__()

        self._feat_stride = feat_stride
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(scales), 
                                                          ratios=np.array(ratios),
                                                          time_dim=np.array(time_dim))).float()
        self._num_anchors = self._anchors.size(0)

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        # top[0].reshape(1, 5)
        #
        # # scores blob: holds scores for R regions of interest
        # if len(top) > 1:
        #     top[1].reshape(1, 1, 1, 1)

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

        scores = input[0][:, self._num_anchors:, :, :]
        bbox_frame = input[1]
        im_info = input[2]
        cfg_key = input[3]
        time_dim = input[4]

        batch_size = bbox_frame.size(0)

        pre_nms_topN  = conf[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = conf[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh    = conf[cfg_key].RPN_NMS_THRESH
        min_size      = conf[cfg_key].RPN_MIN_SIZE

        ##################
        # Create anchors #
        ##################

        feat_height,  feat_width= scores.size(2), scores.size(3) # (batch_size, 512/256, 7, 7)
        shift_x = np.arange(0, feat_width) * self._feat_stride
        shift_y = np.arange(0, feat_height) * self._feat_stride
        shift_z = np.arange(0, 1 )
        shift_x, shift_y, shift_z = np.meshgrid(shift_x, shift_y, shift_z)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(), shift_z.ravel(),
                                             shift_x.ravel(), shift_y.ravel(), shift_z.ravel())).transpose())
        shifts = shifts.contiguous().type_as(scores).float()

        A = self._num_anchors
        K = shifts.size(0)

        self._anchors = self._anchors.type_as(scores)

        anchors = self._anchors.view(1, A, 6) + shifts.view(K, 1, 6)
        anchors = anchors.view(1, K * A, 6)
        anchors = anchors.expand(batch_size, K * A, 6)

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        bbox_frame = bbox_frame.permute(0, 2, 3, 1).contiguous()
        bbox_frame = bbox_frame.view(batch_size, -1, 4)

        # Same story for the scores:
        scores = scores.permute(0, 2, 3, 1).contiguous()
        scores = scores.view(batch_size, -1)

        """
        we have 16 frames, and 28224 3d anchors for each 16 frames
        """
        # Convert anchors into proposals via bbox transformations
        # proposals = bbox_frames_transform_inv(anchors, bbox_deltas, batch_size)
        anchors_xy = anchors[:,:,[0,1,3,4]]
        proposals_xy = bbox_transform_inv(anchors_xy, bbox_frame, batch_size) # proposals have 441 * time_dim shape

        ## if any dimension exceeds the dims of the original image, clamp_ them
        proposals_xy = clip_boxes(proposals_xy, im_info, batch_size)
        proposals = torch.cat(( proposals_xy[:,:,[0,1]],anchors[:,:,2].unsqueeze(2), proposals_xy[:,:,[2,3]], anchors[:,:,5].unsqueeze(2)), dim=2)

        scores_keep = scores
        proposals_keep = proposals

        _, order = torch.sort(scores, 1, True)
        
        output = scores.new(batch_size, post_nms_topN, 8).zero_()
        # print('output.shape :',output.shape)
        for i in range(batch_size):
            # # 3. remove predicted boxes with either height or width < threshold
            # # (NOTE: convert min_size to input image scale stored in im_info[2])
            proposals_single = proposals_keep[i]
            scores_single = scores_keep[i]
            # print('scores_single.shape :',scores_single.shape)
            # # 4. sort all (proposal, score) pairs by score from highest to lowest
            # # 5. take top pre_nms_topN (e.g. 6000)
            order_single = order[i]

            proposals_single = proposals_single[order_single, :]
            scores_single = scores_single[order_single].view(-1,1)

            
            proposals_single = proposals_single[:post_nms_topN, :]
            scores_single = scores_single[:post_nms_topN]
            # print('scores_single.shape :',scores_single.shape)
            # padding 0 at the end.
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
