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
from generate_anchors import generate_anchors  
# from bbox_transform import clip_boxes, bbox_overlaps_batch, bbox_overlaps_time, bbox_transform_batch
from bbox_transform import clip_boxes, bbox_overlaps_time, bbox_transform_batch, bbox_overlaps_rois
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
    def __init__(self, feat_stride, scales, ratios):
        super(_AnchorTargetLayer, self).__init__()

        self._feat_stride = feat_stride
        self._scales = scales
        anchor_scales = scales
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(anchor_scales), ratios=np.array(ratios))).float()
        self._num_anchors = self._anchors.size(0)

        # allow boxes to sit over the edge by a small amount
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
        gt_rois = input[3]          ## rois for each frame
        num_boxes = input[4]     ## number of gt_boxes 
        time_limit = input[5]    ## time limit

        # map of shape (..., H, W)

        ### Not sure about that
        gt_rois = gt_rois.permute(1,0,2)
        
        height, width = rpn_cls_score.size(2), rpn_cls_score.size(3)
        gt_tube_batch_size = gt_tubes.size(0)
        gt_rois_batch_size = gt_rois.size(0)

        print('gt_tubes.shape :', gt_tubes.shape)
        print('gt_tubes :', gt_tubes)

        print('gt_rois.shape :', gt_rois.shape)
        print('gt_rois :', gt_rois.squeeze())


        feat_height, feat_width = rpn_cls_score.size(2), rpn_cls_score.size(3)
        shift_x = np.arange(0, feat_width) * self._feat_stride
        shift_y = np.arange(0, feat_height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                  shift_x.ravel(), shift_y.ravel())).transpose())
        shifts = shifts.contiguous().type_as(rpn_cls_score).float()

        A = self._num_anchors
        K = shifts.size(0)
        print("A {}, K {}".format(A,K))

        self._anchors = self._anchors.type_as(gt_tubes) # move to specific gpu.
        all_anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        all_anchors = all_anchors.view(K * A, 4)
        print('all_anchors.shape :', all_anchors.shape)
        # print('all_anchors[0] :', all_anchors[0])

        total_anchors = int(K * A)
        # print('all_anchors.shape :', all_anchors.shape)
        
        keep = ((all_anchors[:, 0] >= -self._allowed_border) &
                (all_anchors[:, 1] >= -self._allowed_border) &
                (all_anchors[:, 2] < long(im_info[0][1]) + self._allowed_border) &
                (all_anchors[:, 3] < long(im_info[0][0]) + self._allowed_border))

        inds_inside = torch.nonzero(keep).view(-1)

        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]

        # label: 1 is positive, 0 is negative, -1 is dont care
        tube_labels = gt_tubes.new(gt_tube_batch_size, inds_inside.size(0)).fill_(-1)
        print('tube_labels :', tube_labels.shape)
        rois_labels = gt_tubes.new(gt_rois_batch_size, inds_inside.size(0)).fill_(-1)
        print('rois_labels :', rois_labels.shape)
        
        tube_bbox_inside_weights = gt_tubes.new(gt_tube_batch_size, inds_inside.size(0)).zero_()
        tube_bbox_outside_weights = gt_tubes.new(gt_tube_batch_size, inds_inside.size(0)).zero_()
 
        rois_bbox_inside_weights = gt_tubes.new(gt_rois_batch_size, inds_inside.size(0)).zero_()
        rois_bbox_outside_weights = gt_tubes.new(gt_rois_batch_size, inds_inside.size(0)).zero_()

        print('rois_bbox_inside_weights.shape :',rois_bbox_inside_weights.shape)
        tube_overlaps = bbox_overlaps_time(anchors, gt_tubes, time_limit)
        rois_overlaps = bbox_overlaps_rois(anchors, gt_rois, time_limit)
        print('rois_overlaps.shape :',rois_overlaps.shape)
        ##################################################################
        # Until now, we have calculate overlaps for gt_tubes and anchors #
        ##################################################################

        tube_max_overlaps, tube_argmax_overlaps = torch.max(tube_overlaps, 2)
        rois_max_overlaps, rois_argmax_overlaps = torch.max(rois_overlaps, 2)
        
        gt_tube_max_overlaps, _ = torch.max(tube_overlaps, 1)
        gt_rois_max_overlaps, _ = torch.max(rois_overlaps, 1)
        
        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            tube_labels[tube_max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
            rois_labels[rois_max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
            
        gt_tube_max_overlaps[gt_tube_max_overlaps==0] = 1e-5
        gt_rois_max_overlaps[gt_rois_max_overlaps==0] = 1e-5

        tube_keep = torch.sum(tube_overlaps.eq(gt_tube_max_overlaps.view(gt_tube_batch_size,1,-1).expand_as(tube_overlaps)), 2)
        rois_keep = torch.sum(rois_overlaps.eq(gt_rois_max_overlaps.view(gt_rois_batch_size,1,-1).expand_as(rois_overlaps)), 2)

        if torch.sum(tube_keep) > 0:
            tube_labels[tube_keep>0] = 1

        if torch.sum(rois_keep) > 0:
            rois_labels[rois_keep>0] = 1


        # fg label: above threshold IOU
        tube_labels[tube_max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1
        rois_labels[rois_max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            tube_labels[tube_max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
            rois_labels[rois_max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        tube_num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
        rois_num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)

        tube_sum_fg = torch.sum((tube_labels == 1).int(), 1)
        tube_sum_bg = torch.sum((tube_labels == 0).int(), 1)

        rois_sum_fg = torch.sum((rois_labels == 1).int(), 1)
        rois_sum_bg = torch.sum((rois_labels == 0).int(), 1)

        # print('num of foreground : {}, of found sum fg : {}, and bg : {}'.format(
        #     num_fg, sum_fg, sum_bg))

        ## this loop in only for subsampling if we have too many background and foreground samples
        for i in range(gt_tube_batch_size):
            # subsample positive labels if we have too many
            if tube_sum_fg[i] > tube_num_fg:
                tube_fg_inds = torch.nonzero(tube_labels[i] == 1).view(-1)
                # torch.randperm seems has a bug on multi-gpu setting that cause the segfault.
                # See https://github.com/pytorch/pytorch/issues/1868 for more details.
                # use numpy instead.
                #rand_num = torch.randperm(fg_inds.size(0)).type_as(gt_tubes).long()
                tube_rand_num = torch.from_numpy(np.random.permutation(tube_fg_inds.size(0))).type_as(gt_tubes).long()
                tube_disable_inds = tube_fg_inds[tube_rand_num[:tube_fg_inds.size(0)-tube_num_fg]]
                tube_labels[i][tube_disable_inds] = -1

            tube_num_bg = cfg.TRAIN.RPN_BATCHSIZE - torch.sum((tube_labels == 1).int(), 1)[i]

            # subsample negative labels if we have too many
            if tube_sum_bg[i] > tube_num_bg:
                tube_bg_inds = torch.nonzero(tube_labels[i] == 0).view(-1)

                tube_rand_num = torch.from_numpy(np.random.permutation(tube_bg_inds.size(0))).type_as(gt_rois).long()
                tube_disable_inds = tube_bg_inds[tube_rand_num[:tube_bg_inds.size(0)-tube_num_bg]]
                tube_labels[i][tube_disable_inds] = -1

        for i in range(gt_rois_batch_size):
            # subsample positive labels if we have too many
            if rois_sum_fg[i] > rois_num_fg:
                rois_fg_inds = torch.nonzero(rois_labels[i] == 1).view(-1)
                # torch.randperm seems has a bug on multi-gpu setting that cause the segfault.
                # See https://github.com/pytorch/pytorch/issues/1868 for more details.
                # use numpy instead.
                #rand_num = torch.randperm(fg_inds.size(0)).type_as(gt_tubes).long()
                rois_rand_num = torch.from_numpy(np.random.permutation(rois_fg_inds.size(0))).type_as(gt_rois).long()
                rois_disable_inds = rois_fg_inds[rois_rand_num[:rois_fg_inds.size(0)-rois_num_fg]]
                rois_labels[i][rois_disable_inds] = -1

            rois_num_bg = cfg.TRAIN.RPN_BATCHSIZE - torch.sum((rois_labels == 1).int(), 1)[i]

            # subsample negative labels if we have too many
            if rois_sum_bg[i] > rois_num_bg:
                rois_bg_inds = torch.nonzero(rois_labels[i] == 0).view(-1)

                rois_rand_num = torch.from_numpy(np.random.permutation(rois_bg_inds.size(0))).type_as(gt_rois).long()
                rois_disable_inds = rois_bg_inds[rois_rand_num[:rois_bg_inds.size(0)-rois_num_bg]]
                rois_labels[i][rois_disable_inds] = -1


        tube_offset = torch.arange(0, gt_tube_batch_size)*gt_tubes.size(1)
        rois_offset = torch.arange(0, gt_rois_batch_size)*gt_rois.size(1)

        # print('offset :', offset)
        # print('agrmax_overlaps before :', argmax_overlaps)
        tube_argmax_overlaps = tube_argmax_overlaps + tube_offset.view(gt_tube_batch_size, 1).type_as(tube_argmax_overlaps)
        rois_argmax_overlaps = rois_argmax_overlaps + rois_offset.view(gt_rois_batch_size, 1).type_as(rois_argmax_overlaps)
        # print('agrmax_overlaps after :', argmax_overlaps)

        ##################################################################################################################
        # MEXRI EDW EXW ALLAKSEI TON KWDIKA, NA ton dw vima vima
        ##################################################################################################################

        # print('gt_tubes.shape :',gt_tubes.shape)
        # print(gt_tubes.view(-1,7)[argmax_overlaps.view(-1), :].view(batch_size, -1, 7)) 
        # print(gt_tubes.view(-1,7)[argmax_overlaps.view(-1), :].view(batch_size, -1, 7).shape)

        # print('gt_rois.shape :', gt_rois.shape)
        # print('gt_rois.view(-1,5)[argmax_overlaps.view(-1), :].view(batch_size, -1, 5) :',gt_rois.view(-1,5)[argmax_overlaps.view(-1), :].view(batch_size, -1, 5))
        # print('gt_rois.view(-1,5)[argmax_overlaps.view(-1), :].view(batch_size, -1, 5).shape :',gt_rois.view(-1,5)[argmax_overlaps.view(-1), :].view(batch_size, -1, 5).shape)
        # the bbox in every anchor 
        # bbox_targets = _compute_targets_batch(anchors, gt_tubes.view(-1,7)[argmax_overlaps.view(-1), :].view(batch_size, -1, 7))
        tube_bbox_targets = _compute_targets_batch(anchors, gt_tubes.view(-1,7)[tube_argmax_overlaps.view(-1), :].view(gt_tube_batch_size, -1, 7))
        rois_bbox_targets = _compute_targets_batch(anchors, gt_rois.view(-1,5)[rois_argmax_overlaps.view(-1), :].view(gt_rois_batch_size, -1, 5))
        # # use a single value instead of 4 values for easy index.
        # print('bbox_inside_weights :',bbox_inside_weights)

        tube_bbox_inside_weights[tube_labels==1] = cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS[0]
        # print('bbox_inside_weights after :',bbox_inside_weights)
        

        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
            print('i  :,',i)
            print('tube_labels.shape :', tube_labels.shape)
            print('tube_labels :', tube_labels)
            tube_num_examples = torch.sum(tube_labels[0] >= 0)
            print('tube_num_examples :',tube_num_examples)
            tube_positive_weights = 1.0 / tube_num_examples.item()
            tube_negative_weights = 1.0 / tube_num_examples.item()
        else:
            assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                    (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))

        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
            print('i  :,',i)
            print('rois_labels.shape :', rois_labels.shape)
            print('rois_labels :', rois_labels)
            rois_num_examples = torch.sum(rois_labels[i] >= 0)
            rois_positive_weights = 1.0 / rois_num_examples.item()
            rois_negative_weights = 1.0 / rois_num_examples.item()
        else:
            assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                    (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))


        # print('num_examples : {}, positive_weights {}, negative_weights {}'.format(num_examples,positive_weights, negative_weights))
        # print('bbox_outside_weights :', bbox_outside_weights)
        tube_bbox_outside_weights[tube_labels == 1] = tube_positive_weights
        tube_bbox_outside_weights[tube_labels == 0] = tube_negative_weights

        rois_bbox_outside_weights[rois_labels == 1] = rois_positive_weights
        rois_bbox_outside_weights[rois_labels == 0] = rois_negative_weights

        # print('bbox_outside_weights :', bbox_outside_weights)

        # print('labels :',labels)
        # print('labels.shape :',labels.shape)

        tube_labels = _unmap(tube_labels, total_anchors, inds_inside, gt_tube_batch_size, fill=-1)
        rois_labels = _unmap(rois_labels, total_anchors, inds_inside, gt_rois_batch_size, fill=-1)

        # print('labels :',labels)
        # print('labels.shape :',labels.shape)
        # print('total_anchors :', total_anchors)

        # print('bbox_targets :',bbox_targets)
        # print('bbox_inside_weights :',bbox_inside_weights)
        # print('bbox_outside_weights :',bbox_outside_weights)

        tube_bbox_targets = _unmap(tube_bbox_targets, total_anchors, inds_inside, gt_tube_batch_size, fill=0)  
        tube_bbox_inside_weights = _unmap(tube_bbox_inside_weights, total_anchors, inds_inside, gt_tube_batch_size, fill=0)
        tube_bbox_outside_weights = _unmap(tube_bbox_outside_weights, total_anchors, inds_inside, gt_tube_batch_size, fill=0)

        rois_bbox_targets = _unmap(rois_bbox_targets, total_anchors, inds_inside, gt_rois_batch_size, fill=0)  
        rois_bbox_inside_weights = _unmap(rois_bbox_inside_weights, total_anchors, inds_inside, gt_rois_batch_size, fill=0)
        rois_bbox_outside_weights = _unmap(rois_bbox_outside_weights, total_anchors, inds_inside, gt_rois_batch_size, fill=0)

        outputs = []

        ### tube

        tube_labels = tube_labels.view(gt_tube_batch_size, height, width, A).permute(0,3,1,2).contiguous()
        tube_labels = tube_labels.view(gt_tube_batch_size, 1, A * height, width)

        print('tube_labels.shape :',tube_labels.shape)
        outputs.append(tube_labels)

        tube_bbox_targets = tube_bbox_targets.view(gt_tube_batch_size, height, width, A*4).permute(0,3,1,2).contiguous()
        outputs.append(tube_bbox_targets)

        tube_anchors_count = tube_bbox_inside_weights.size(1)
        tube_bbox_inside_weights = tube_bbox_inside_weights.view(
            gt_tube_batch_size,tube_anchors_count,1).expand(gt_tube_batch_size, tube_anchors_count, 4)
        tube_bbox_inside_weights = tube_bbox_inside_weights.contiguous().view(gt_tube_batch_size, height, width, 4*A)\
                            .permute(0,3,1,2).contiguous()
        outputs.append(tube_bbox_inside_weights)

        tube_bbox_outside_weights = tube_bbox_outside_weights.view(
            gt_tube_batch_size,tube_anchors_count,1).expand(gt_tube_batch_size, tube_anchors_count, 4)
        tube_bbox_outside_weights = tube_bbox_outside_weights.contiguous().view(gt_tube_batch_size, height, width, 4*A)\
                            .permute(0,3,1,2).contiguous()
        outputs.append(tube_bbox_outside_weights)

        #### rois

        rois_labels = rois_labels.view(gt_rois_batch_size, height, width, A).permute(0,3,1,2).contiguous()
        rois_labels = rois_labels.view(gt_rois_batch_size, 1, A * height, width)
        outputs.append(rois_labels)

        rois_bbox_targets = rois_bbox_targets.view(gt_rois_batch_size, height, width, A*4).permute(0,3,1,2).contiguous()
        rois_bbox_targets = rois_bbox_targets.view(-1,7,7)
        outputs.append(rois_bbox_targets)

        rois_anchors_count = rois_bbox_inside_weights.size(1)
        rois_bbox_inside_weights = rois_bbox_inside_weights.view(
            gt_rois_batch_size,rois_anchors_count,1).expand(gt_rois_batch_size, rois_anchors_count, 4)
        rois_bbox_inside_weights = rois_bbox_inside_weights.contiguous().view(gt_rois_batch_size, height, width, 4*A)\
                            .permute(0,3,1,2).contiguous()
        rois_bbox_inside_weights = rois_bbox_inside_weights.view(-1,7,7)
        outputs.append(rois_bbox_inside_weights)

        rois_bbox_outside_weights = rois_bbox_outside_weights.view(
            gt_rois_batch_size,rois_anchors_count,1).expand(gt_rois_batch_size, rois_anchors_count, 4)
        rois_bbox_outside_weights = rois_bbox_outside_weights.contiguous().view(gt_rois_batch_size, height, width, 4*A)\
                            .permute(0,3,1,2).contiguous()
        rois_bbox_outside_weights = rois_bbox_outside_weights.view(-1,7,7)
        outputs.append(rois_bbox_outside_weights)

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
        ret = torch.Tensor(batch_size, count).fill_(fill).type_as(data)
        ret[:, inds] = data
    else:
        ret = torch.Tensor(batch_size, count, data.size(2)).fill_(fill).type_as(data)
        ret[:, inds,:] = data
    return ret


def _compute_targets_batch(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    # return bbox_transform_time(ex_rois, gt_rois[:,:, :6])
    # print('gt_rois[:,:, [0,1,3,4] :',gt_rois[:,:, [0,1,3,4]])
    # print('ex_rois.shape :',ex_rois.shape)
    return bbox_transform_batch(ex_rois, gt_rois[:,:, [0,1,3,4]])
