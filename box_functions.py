##############################################################
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################

# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch


# def split_tube_into_boxes(tube, T=None):
#     """ N tubes are represented as a (N, 4 * T,) dimensional matrix,
#     or sometimes (N, 4 * T * num_classes) matrix. In the latter case, T should
#     be passed as input. In some cases, there can be a score value at the end of
#     the tube vector, which will be copied to each split box. """
#     N = tube.shape[0]

#     if tube.shape[1] % 4 == 0:
#         scores = np.zeros((N, 0))
#     elif (tube.shape[1] - 1) % 4 == 0:
#         scores = tube[:, (-1,)]
#         tube = tube[:, :-1]
#     else:
#         raise ValueError('Invalid tube dimensions {}'.format(tube.shape))
#     T = T or tube.shape[-1] // 4
#     boxes = []
#     if 4 * T != tube.shape[-1]:
#         # Means it is the box-per-class kinda representation
#         # Take ith box from each class
#         assert(tube.shape[-1] % (4 * T) == 0)
#         num_classes = tube.shape[-1] // (4 * T)
#         for t in range(T):
#             box_rep = np.zeros((tube.shape[0], 4 * num_classes))
#             for cid in range(num_classes):
#                 box_rep[:, cid * 4: (cid + 1) * 4] = \
#                     tube[:, cid * 4 * T: (cid + 1) * 4 * T][:, t * 4: (t + 1) * 4]
#             boxes.append(box_rep)
#     else:
#         for t in range(T):
#             boxes.append(tube[..., t * 4: (t + 1) * 4])
#     # Add the scores back, if there were any
#     boxes = [np.hstack((box, scores)) for box in boxes]
#     return boxes, T

def split_tube_into_boxes(tube, T=None):
    """ N tubes are represented as a (N, 4 * T,) dimensional matrix,
    or sometimes (N, 4 * T * num_classes) matrix. In the latter case, T should
    be passed as input. In some cases, there can be a score value at the end of
    the tube vector, which will be copied to each split box. """
    N = tube.shape[0]

    T = T or tube.shape[-1] // 4
    boxes = []
    if 4 * T != tube.shape[-1]:
        # Means it is the box-per-class kinda representation
        # Take ith box from each class
        assert(tube.shape[-1] % (4 * T) == 0)
        num_classes = tube.shape[-1] // (4 * T)
        for t in range(T):
            box_rep = np.zeros((tube.shape[0], 4 * num_classes))
            for cid in range(num_classes):
                box_rep[:, cid * 4: (cid + 1) * 4] = \
                    tube[:, cid * 4 * T: (cid + 1) * 4 * T][:, t * 4: (t + 1) * 4]
            boxes.append(box_rep)
    else:
        for t in range(T):
            boxes.append(tube[..., t * 4: (t + 1) * 4])

    return boxes, T



def bbox_transform(ex_rois, gt_rois, weights):
    """Forward transform that maps proposal boxes to ground-truth boxes using
    bounding-box regression deltas. See bbox_transform_inv for a description of
    the weights argument.
    """

    if ex_rois.shape[1] > 4:
        return tube_transform(ex_rois, gt_rois, weights)
    if ex_rois.shape[0] == 0:
        return torch.zeros((0, gt_rois.shape[1])).type_as(gt_rois)

    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = torch.log(gt_widths / ex_widths)
    targets_dh = torch.log(gt_heights / ex_heights)

    targets = torch.stack(
        (targets_dx, targets_dy, targets_dw, targets_dh), 1)

    return targets

# def bbox_transform(boxes, deltas, weights):
#     """Forward transform that maps proposal boxes to ground-truth boxes using
#     bounding-box regression deltas. See bbox_transform_inv for a description of
#     the weights argument.
#     """

#     if boxes.shape[1] > 4:
#         return tube_transform(boxes, deltas, weights)
#     if boxes.shape[0] == 0:
#         return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

#     boxes = boxes.type_as(deltas)
#     print('boxes  :',boxes)
#     print('deltas :',deltas)
    
#     widths = boxes[:, 2] - boxes[:, 0] + 1.0
#     heights = boxes[:, 3] - boxes[:, 1] + 1.0
#     ctr_x = boxes[:, 0] + 0.5 * widths
#     ctr_y = boxes[:, 1] + 0.5 * heights

#     wx, wy, ww, wh = weights
#     dx = deltas[:, 0::4] / wx
#     dy = deltas[:, 1::4] / wy
#     dw = deltas[:, 2::4] / ww
#     dh = deltas[:, 3::4] / wh

#     print('dw :',dw)
#     print('dh :',dh)
#     # Prevent sending too large values into np.exp()
#     dw = torch.min(dw, torch.log(torch.Tensor([1000. / 16.]).type_as(dw)))
#     dh = torch.min(dh, torch.log(torch.Tensor([1000. / 16.]).type_as(dw)))


#     exit(-1)
#     pred_ctr_x = dx * widths.unsqueeze(1) + ctr_x.unsqueeze(1)
#     pred_ctr_y = dy * heights.unsqueeze(1) + ctr_y.unsqueeze(1)
#     pred_w = torch.exp(dw) * widths.unsqueeze(1)
#     pred_h = torch.exp(dh) * heights.unsqueeze(1)

#     pred_boxes = torch.zeros(deltas.shape).type_as(deltas)
#     # x1
#     pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
#     # y1
#     pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
#     # x2
#     pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
#     # y2
#     pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

#     return pred_boxes



def tube_transform(boxes, deltas, weights):
    """ Use the bbox_transform to compute the transform,
    and apply to all the boxes. """
    boxes_parts, time_dim = split_tube_into_boxes(boxes)
    deltas_parts, _ = split_tube_into_boxes(deltas, time_dim)
    all_tx = [bbox_transform(boxes_part, deltas_part, weights) for
              boxes_part, deltas_part in zip(boxes_parts, deltas_parts)]
    # Now need to interleave the boxes in the same way they were un-interleaved
    # in the split_tube_into_boxes function
    nclasses = all_tx[0].shape[-1] // 4
    time_dim = len(all_tx)
    res = torch.zeros(deltas.shape).type_as(deltas)
    for cid in range(nclasses):
        for t in range(time_dim):
            res[:, cid * 4 * time_dim: (cid + 1) * 4 * time_dim][
                :, t * 4: (t + 1) * 4] = all_tx[t][:, cid * 4: (cid + 1) * 4]
    return res

def bbox_transform_inv(ex_rois, gt_rois, weights):
    """Backward transform that computes bounding-box regression deltas given
    proposal boxes and ground-truth boxes. The weights argument should be a
    4-tuple of multiplicative weights that are applied to the regression target.
    """

    if ex_rois.shape[1] > 4:
        return tube_transform_inv(ex_rois, gt_rois, weights)
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    wx, wy, ww, wh = weights
    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * torch.log(gt_widths / ex_widths)
    targets_dh = wh * torch.log(gt_heights / ex_heights)

    targets = torch.stack((targets_dx, targets_dy, \
                           targets_dw, targets_dh), dim=0).t()


    return targets


def tube_transform_inv(ex_rois, gt_rois, weights):
    """ Tube extension of the box rois. """
    # print('ex_rois.shape :',ex_rois.shape)
    # print('gt_rois.shape :',gt_rois.shape)
    # exit(-1)
    assert(ex_rois.shape[1] == gt_rois.shape[1])
    ex_rois_parts, _ = split_tube_into_boxes(ex_rois)
    gt_rois_parts, _ = split_tube_into_boxes(gt_rois)
    all_tx = [bbox_transform_inv(ex_rois_part, gt_rois_part, weights) for
              ex_rois_part, gt_rois_part in zip(ex_rois_parts, gt_rois_parts)]
    return torch.cat(all_tx, dim=1)


def clip_boxes(boxes, im_shape, batch_size):

    for i in range(batch_size):
        boxes[i, :, 0::4].clamp_(0, im_shape[i, 1]-1)
        boxes[i, :, 1::4].clamp_(0, im_shape[i, 0]-1)
        boxes[i, :, 2::4].clamp_(0, im_shape[i, 1]-1)
        boxes[i, :, 3::4].clamp_(0, im_shape[i, 0]-1)

    return boxes

def clip_tiled_boxes(boxes, im_shape):
    """Clip boxes to image boundaries. im_shape is [height, width]."""
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)

    return boxes

def tube_overlaps(boxes, query_boxes):
    """ Use the cython overlap implementation to compute the overlap,
    and average the overlaps over time for tubes. """
    parts, _ = split_tube_into_boxes(boxes)
    query_parts, _ = split_tube_into_boxes(query_boxes)

    return torch.mean(torch.stack([
        bbox_overlaps(
            part, query_part)
        for part, query_part in zip(parts, query_parts)]), dim=0)

# def tube_overlaps(boxes, query_boxes):
    
def bbox_overlaps(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 4) ndarray of float

    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.size(0)
    K = gt_boxes.size(0)

    gt_boxes_x = (gt_boxes[ :, 2] - gt_boxes[ :, 0] + 1)
    gt_boxes_y = (gt_boxes[ :, 3] - gt_boxes[ :, 1] + 1)

    gt_boxes_area = (gt_boxes_x *
                     gt_boxes_y).view(1,K)

    anchors_boxes_x = (anchors[ :, 2] - anchors[ :, 0] + 1)
    anchors_boxes_y = (anchors[ :, 3] - anchors[ :, 1] + 1)

    anchors_area = (anchors_boxes_x *
                    anchors_boxes_y).view(N, 1)

    gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
    anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

    boxes = anchors.view(N, 1, 4).expand(N, K, 4)
    query_boxes = gt_boxes.view(1, K, 4).expand(N, K, 4)

    iw = (torch.min(boxes[:, :, 2], query_boxes[:, :, 2]) -
          torch.max(boxes[:, :, 0], query_boxes[:, :, 0]) + 1)
    iw[iw < 0] = 0

    ih = (torch.min(boxes[:, :, 3], query_boxes[:, :, 3]) -
          torch.max(boxes[:, :, 1], query_boxes[:, :, 1]) + 1)
    ih[ih < 0] = 0

    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    overlaps.masked_fill_(gt_area_zero.view(
        1, K).expand(N, K), 0)
    overlaps.masked_fill_(anchors_area_zero.view(
        N, 1).expand( N, K), -1)

    return overlaps

