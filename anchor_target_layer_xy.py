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
from generate_3d_anchors import generate_anchors
# from bbox_transform import clip_boxes, bbox_overlaps_batch, bbox_overlaps_time, bbox_transform_batch
from bbox_transform import clip_boxes, bbox_transform_batch_3d, bbox_overlaps_batch_3d
import pdb

DEBUG = False


try:
    long        # Python 2
except NameError:
    long = int  # Python 3


class _AnchorTargetLayer_xy(nn.Module):
    """
        Assign anchors to ground-truth targets. Produces anchor classification
        labels and bounding-box regression targets.
    """
    def __init__(self, feat_stride, scales, ratios, anchor_duration):
        super(_AnchorTargetLayer_xy, self).__init__()

        self._feat_stride = feat_stride

        self._scales = scales
        anchor_scales = scales
        self.anchor_duration = anchor_duration
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(anchor_scales), ratios=np.array(ratios), time_dim=anchor_duration)).float()
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
        # gt_rois = input[3]       ## gt rois for each frame

        # map of shape (..., H, W)

        ### Not sure about that
        # print('$$$$$$$$$$')
        batch_size = gt_tubes.size(0)

        # print('time_limit :',time_limit)
        height, width  = rpn_cls_score.size(2), rpn_cls_score.size(3)
        # print('time :', time)
        feat_height, feat_width,  = rpn_cls_score.size(2), rpn_cls_score.size(3)

        shift_x = np.arange(0, feat_width) * self._feat_stride
        shift_y = np.arange(0, feat_height) * self._feat_stride
        shift_z = np.arange(0, 1) 
        shift_x, shift_y, shift_z = np.meshgrid(shift_x, shift_y, shift_z)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(), shift_z.ravel(),
                                             shift_x.ravel(), shift_y.ravel(), shift_z.ravel())).transpose())
        shifts = shifts.contiguous().type_as(rpn_cls_score).float()

        A = self._num_anchors
        K = shifts.size(0)
        # print("A {}, K {}".format(A,K))

        self._anchors = self._anchors.type_as(gt_tubes) # move to specific gpu.
        # print('self._anchors :',self._anchors)
        # print('self._anchors.shape :',self._anchors.shape)
        all_anchors = self._anchors.view(1, A, 6) + shifts.view(K, 1, 6)
        all_anchors = all_anchors.view(K * A, 6)

        # print('all_anchors :', all_anchors.shape)
        # print('all_anchors :',all_anchors)
        total_anchors = int(K * A)
        # print('total_anchor :',total_anchors)
        keep = ((all_anchors[:, 0] >= -self._allowed_border) &
                (all_anchors[:, 1] >= -self._allowed_border) &
                (all_anchors[:, 2] >= -self._allowed_border) &
                (all_anchors[:, 3] < long(im_info[0][1]) + self._allowed_border) &
                (all_anchors[:, 4] < long(im_info[0][0]) + self._allowed_border) &
                (all_anchors[:, 5] < long(im_info[0][2]) + self._allowed_border))

        inds_inside = torch.nonzero(keep).view(-1)
        
        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]

        # for i in anchors.cpu().tolist():
        #     if i[2] ==0 and i[5] ==15:
        #         print('epaeee i:',i)

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = gt_tubes.new(batch_size, inds_inside.size(0)).fill_(-1)

        bbox_inside_weights = gt_tubes.new(batch_size, inds_inside.size(0)).zero_()
        bbox_outside_weights = gt_tubes.new(batch_size, inds_inside.size(0)).zero_()
        # print('anchors.shape :',anchors.shape)
        overlaps = bbox_overlaps_batch_3d(anchors, gt_tubes)
        # print('overlaps.shape :',overlaps.shape)
        indx = np.where(overlaps.cpu().numpy() > 0.3)
        # print(indx)

        # print('rois_overlaps.shape :',rois_overlaps.shape)

        ##################################################################
        # Until now, we have calculate overlaps for gt_tubes and anchors #
        ##################################################################
        # print('max(overlaps) :',np.max(overlaps.cpu().numpy(),axis=2))
        # print('max(overlaps) :',np.max(overlaps.cpu().numpy(),axis=2).shape)
        # print('max(overlaps) :',np.max(np.max(overlaps.cpu().numpy(),axis=2),axis=1))
        # print('max(overlaps) :',np.max(np.max(overlaps.cpu().numpy(),axis=2),axis=1))


        max_overlaps, argmax_overlaps = torch.max(overlaps, 2)
        gt_max_overlaps, _ = torch.max(overlaps, 1)

        # print('max_overlaps.shape :', max_overlaps.shape)
        # print('max_overlaps :', max_overlaps.cpu().tolist())
        # print('labels.shape :',labels.shape)
        # print('labels :',labels)

        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        gt_max_overlaps[gt_max_overlaps==0] = 1e-5
        # print(gt_max_overlaps.view(batch_size,1,-1))
        # print('gt_max_overlaps.view(batch_size,1,-1) :',gt_max_overlaps.view(batch_size,1,-1).shape)
        # print('gt_max_overlaps.view(batch_size,1,-1).expand_as(overlaps).shape :',gt_max_overlaps.view(batch_size,1,-1).expand_as(overlaps).shape)
        keep = torch.sum(overlaps.eq(gt_max_overlaps.view(batch_size,1,-1).expand_as(overlaps)), 2)

        if torch.sum(keep) > 0:
            labels[keep>0] = 1
        # print('cfg.TRAIN.RPN_POSITIVE_OVERLAP :',cfg.TRAIN.RPN_POSITIVE_OVERLAP)
        # fg label: above threshold IOU
        labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)

        sum_fg = torch.sum((labels == 1).int(), 1)
        sum_bg = torch.sum((labels == 0).int(), 1)

        # print('fg :',sum_fg)
        # print('bg :',sum_bg)
        for i in range(batch_size):
            # subsample positive labels if we have too many
            if sum_fg[i] > num_fg:
                print('mpikeeeee')
                fg_inds = torch.nonzero(labels[i] == 1).view(-1)
                # torch.randperm seems has a bug on multi-gpu setting that cause the segfault.
                # See https://github.com/pytorch/pytorch/issues/1868 for more details.
                # use numpy instead.
                #rand_num = torch.randperm(fg_inds.size(0)).type_as(gt_boxes).long()
                rand_num = torch.from_numpy(np.random.permutation(fg_inds.size(0))).type_as(gt_tubes).long()
                disable_inds = fg_inds[rand_num[:fg_inds.size(0)-num_fg]]
                labels[i][disable_inds] = -1

#           num_bg = cfg.TRAIN.RPN_BATCHSIZE - sum_fg[i]
            num_bg = cfg.TRAIN.RPN_BATCHSIZE - torch.sum((labels == 1).int(), 1)[i]

            # subsample negative labels if we have too many
            if sum_bg[i] > num_bg:

                bg_inds = torch.nonzero(labels[i] == 0).view(-1)
                #rand_num = torch.randperm(bg_inds.size(0)).type_as(gt_boxes).long()

                rand_num = torch.from_numpy(np.random.permutation(bg_inds.size(0))).type_as(gt_tubes).long()
                disable_inds = bg_inds[rand_num[:bg_inds.size(0)-num_bg]]
                labels[i][disable_inds] = -1

        offset = torch.arange(0, batch_size)*gt_tubes.size(1)

        argmax_overlaps = argmax_overlaps + offset.view(batch_size, 1).type_as(argmax_overlaps)
        bbox_targets = _compute_targets_batch(anchors, gt_tubes.view(-1,7)[argmax_overlaps.view(-1), :].view(batch_size, -1, 7))
        # print('bbox_targets :',bbox_targets.shape)
        # use a single value instead of 4 values for easy index.
        # print('cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS[0] :',cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS[0])
        bbox_inside_weights[labels==1] = cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS[0]

        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
            # print('edwww323r', cfg.TRAIN.RPN_POSITIVE_WEIGHT)
            num_examples = torch.sum(labels[i] >= 0)
            # print('num_examples :',num_examples)
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
        labels = labels.view(batch_size, height, width, A).permute(0,3,1,2).contiguous()
        labels = labels.view(batch_size, 1, A *  height, width)
        # print('labels.shape :',labels.shape)
        outputs.append(labels)

        # print('bbox_targets.shape :',bbox_targets.shape)
        bbox_targets = bbox_targets[:,:,[0,1,3,4]]
        bbox_targets = bbox_targets.view(batch_size, height, width,  A*4).permute(0,3,1,2).contiguous()
        outputs.append(bbox_targets)

        anchors_count = bbox_inside_weights.size(1)
        bbox_inside_weights = bbox_inside_weights.view(batch_size,anchors_count,1).expand(batch_size, anchors_count, 6)
        bbox_inside_weights = bbox_inside_weights[:,:,[0,1,3,4]]
        bbox_inside_weights = bbox_inside_weights.contiguous().view(batch_size, height, width,  A * 4)\
                            .permute(0,3,1,2).contiguous()
        outputs.append(bbox_inside_weights)

        bbox_outside_weights = bbox_outside_weights.view(batch_size,anchors_count,1).expand(batch_size, anchors_count, 6)
        bbox_outside_weights = bbox_outside_weights[:,:,[0,1,3,4]]
        bbox_outside_weights = bbox_outside_weights.contiguous().view(batch_size, height, width,  A * 4)\
                            .permute(0,3,1,2).contiguous()

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
    # print('gt_rois.shape :',gt_rois.shape)
    return bbox_transform_batch_3d(ex_rois, gt_rois[:,:, :7])
