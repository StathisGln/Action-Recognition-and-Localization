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
import numpy.random as npr

from config import cfg
from generate_anchors import generate_anchors_all_pyramids
from box_functions import bbox_transform, bbox_transform_inv, clip_boxes, tube_overlaps
import pdb

DEBUG = False


try:
    long        # Python 2
except NameError:
    long = int  # Python 3


class _AnchorTargetLayer(nn.Module):
    """
        Assign anchors to ground-truth targets. Produces anchor classification
        labels and bounding-box regression targets.
    """
    def __init__(self, feat_stride, scales, ratios, anchor_duration,num_anchors):
        super(_AnchorTargetLayer, self).__init__()

        self._feat_stride = feat_stride

        self.sample_duration = anchor_duration[0]
        self.time_dim = anchor_duration
        self._fpn_scales = scales
        self._anchor_ratios = ratios
        self._fpn_feature_strides = np.array([ 4, 8, 16, 32])
        self._fpn_anchor_stride  = 1

        self._allowed_border = 0  # default is 0

    def forward(self, input):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the 9 anchors
        # filter out-of-image anchors

        rpn_cls_score = input[0] ## rpn classification score
        gt_tubes = input[1]      ## gt tube
        im_info = input[2]       ## im_info
        gt_rois = input[3][:,:,:,:4].contiguous()       ## gt rois for each frame
        feat_shapes = input[4]

        # map of shape (..., H, W)
        height, width = 0, 0
        batch_size = gt_tubes.size(0)
        anchors = torch.from_numpy(generate_anchors_all_pyramids(self._fpn_scales, self._anchor_ratios, 
                feat_shapes, self._fpn_feature_strides, self._fpn_anchor_stride)).type_as(rpn_cls_score)    


        # # get time anchors
        anchors_all = []

        for i in range(len(self.time_dim)):
            for j in range(0,self.sample_duration-self.time_dim[i]+1):
                anc = torch.zeros((self.sample_duration,anchors.size(0),4))
                
                anc[ j:j+self.time_dim[i]] = anchors
                anc = anc.permute(1,0,2)

                anchors_all.append(anc)

        anchors_all = torch.cat(anchors_all,0).cuda()
        anchors_all = anchors_all.view(anchors_all.size(0), self.sample_duration * 4)

        total_anchors =  anchors_all.size(0)
        anchors_all = anchors_all.view(total_anchors,self.sample_duration,4)

        keep = ((anchors_all[:, :, 0].ge(-self._allowed_border).all(dim=1)) &
                (anchors_all[:, :, 1].ge(-self._allowed_border).all(dim=1)) &
                (anchors_all[:, :, 2].lt(long(im_info[0][1]) + self._allowed_border).all(dim=1)) &
                (anchors_all[:, :, 3].lt(long(im_info[0][0]) + self._allowed_border).all(dim=1)) )

        inds_inside = torch.nonzero(keep).view(-1)
        # keep only inside anchors
        anchors = anchors_all[inds_inside,].type_as(gt_tubes)
        anchors = anchors.view(anchors.size(0),-1)

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = gt_tubes.new(batch_size, inds_inside.size(0)).fill_(-1)

        bbox_inside_weights = gt_tubes.new(batch_size, inds_inside.size(0)).zero_()
        bbox_outside_weights = gt_tubes.new(batch_size, inds_inside.size(0)).zero_()

        overlaps = tube_overlaps(anchors, gt_rois.view(-1,self.sample_duration*4)).view(batch_size,anchors.size(0),gt_rois.size(1))

        max_overlaps, argmax_overlaps = torch.max(overlaps, 2)
        gt_max_overlaps, _ = torch.max(overlaps, 1)

        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        gt_max_overlaps[gt_max_overlaps==0] = 1e-5
        keep = torch.sum(overlaps.eq(gt_max_overlaps.view(batch_size,1,-1).expand_as(overlaps)), 2)

        if torch.sum(keep) > 0:
            labels[keep>0] = 1

        # fg label: above threshold IOU
        labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)

        sum_fg = torch.sum((labels == 1).int(), 1)
        sum_bg = torch.sum((labels == 0).int(), 1)

        for i in range(batch_size):
            # subsample positive labels if we have too many
            if sum_fg[i] > num_fg:

                fg_inds = torch.nonzero(labels[i] == 1).view(-1)
                # torch.randperm seems has a bug on multi-gpu setting that cause the segfault.
                # See https://github.com/pytorch/pytorch/issues/1868 for more details.
                # use numpy instead.
                #rand_num = torch.randperm(fg_inds.size(0)).type_as(gt_boxes).long()
                rand_num = torch.from_numpy(np.random.permutation(fg_inds.size(0))).type_as(gt_tubes).long()#.to(dev)
                disable_inds = fg_inds[rand_num[:fg_inds.size(0)-num_fg]]
                labels[i][disable_inds] = -1

            num_bg = cfg.TRAIN.RPN_BATCHSIZE - torch.sum((labels == 1).int(), 1)[i]

            # subsample negative labels if we have too many
            if sum_bg[i] > num_bg:

                bg_inds = torch.nonzero(labels[i] == 0).view(-1)
                #rand_num = torch.randperm(bg_inds.size(0)).type_as(gt_boxes).long()

                rand_num = torch.from_numpy(np.random.permutation(bg_inds.size(0))).type_as(gt_tubes).long()#.to(dev)
                disable_inds = bg_inds[rand_num[:bg_inds.size(0)-num_bg]]
                labels[i][disable_inds] = -1

        offset = torch.arange(0, batch_size)*gt_tubes.size(1)

        argmax_overlaps = argmax_overlaps + offset.view(batch_size, 1).type_as(argmax_overlaps)#.to(dev)
        
        bbox_targets = _compute_targets_batch(anchors.unsqueeze(0).expand(batch_size,anchors.size(0),anchors.size(1)).\
                                              contiguous().view(-1,self.sample_duration*4),\
                                              gt_rois.view(-1,self.sample_duration*4)[argmax_overlaps.view(-1), :]). \
                                              view(batch_size,-1,self.sample_duration*4)

        bbox_inside_weights[labels==1] = cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS[0]

        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
            num_examples = torch.sum(labels[i] >= 0)
            positive_weights = 1.0 / num_examples.item()
            negative_weights = 1.0 / num_examples.item()
        else:

            assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                    (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))

        bbox_outside_weights[labels == 1] = positive_weights
        bbox_outside_weights[labels == 0] = negative_weights

        labels = _unmap(labels, total_anchors, inds_inside, batch_size, fill=-1)

        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, batch_size, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, batch_size, fill=0)
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, batch_size, fill=0)

        outputs = []

        outputs.append(labels)
        outputs.append(bbox_targets)
        outputs.append(bbox_inside_weights)
        outputs.append(bbox_outside_weights)

        return outputs

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

def _unmap(data, count, inds, batch_size, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """

    if data.dim() == 2:
        ret = torch.Tensor(batch_size, count).fill_(fill).type_as(data).to(data.device)
        ret[:, inds] = data
    else:
        ret = torch.Tensor(batch_size, count, data.size(2)).fill_(fill).type_as(data).to(data.device)
        ret[:, inds,:] = data
    return ret


def _compute_targets_batch(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    return bbox_transform(ex_rois, gt_rois.to(ex_rois.device),(1.0,1.0,1.0,1.0))
