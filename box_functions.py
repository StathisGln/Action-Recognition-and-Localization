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
    """Backward transform that computes bounding-box regression deltas given
    proposal boxes and ground-truth boxes. The weights argument should be a
    4-tuple of multiplicative weights that are applied to the regression target.
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

def bbox_transform_inv(boxes, deltas, weights):
    """Forward transform that maps proposal boxes to ground-truth boxes using
    bounding-box regression deltas. See bbox_transform_inv for a description of
    the weights argument.
    """

    if boxes.shape[1] > 4:
        return tube_transform_inv(boxes, deltas, weights)
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.type_as(deltas)
    
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0

    zero_area = (widths == 1) & (heights == 1)

    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    wx, wy, ww, wh = weights
    dx = deltas[:, 0::4] / wx
    dy = deltas[:, 1::4] / wy
    dw = deltas[:, 2::4] / ww
    dh = deltas[:, 3::4] / wh

    # Prevent sending too large values into np.exp()
    dw = torch.min(dw, torch.log(torch.Tensor([1000. / 16.]).type_as(dw)))
    dh = torch.min(dh, torch.log(torch.Tensor([1000. / 16.]).type_as(dw)))

    pred_ctr_x = dx * widths.unsqueeze(1) + ctr_x.unsqueeze(1)
    pred_ctr_y = dy * heights.unsqueeze(1) + ctr_y.unsqueeze(1)
    pred_w = torch.exp(dw) * widths.unsqueeze(1)
    pred_h = torch.exp(dh) * heights.unsqueeze(1)

    pred_boxes = torch.zeros(deltas.shape).type_as(deltas)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    pred_boxes.masked_fill_(zero_area.view(zero_area.size(0),1).expand(zero_area.size(0),4),0)

    return pred_boxes



def tube_transform_inv(boxes, deltas, weights):
    """ Use the bbox_transform_inv to compute the transform,
    and apply to all the boxes. """
    boxes_parts, time_dim = split_tube_into_boxes(boxes)
    deltas_parts, _ = split_tube_into_boxes(deltas, time_dim)
    all_tx = [bbox_transform_inv(boxes_part, deltas_part, weights) for
              boxes_part, deltas_part in zip(boxes_parts, deltas_parts)]
    return torch.cat(all_tx, dim=1)

def tube_transform(ex_rois, gt_rois, weights):
    """ Tube extension of the box rois. """
    assert(ex_rois.shape[1] == gt_rois.shape[1])
    ex_rois_parts, _ = split_tube_into_boxes(ex_rois)
    gt_rois_parts, _ = split_tube_into_boxes(gt_rois)
    all_tx = [bbox_transform(ex_rois_part, gt_rois_part, weights) for
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

# def tube_overlaps(boxes, query_boxes):
#     """ Use the cython overlap implementation to compute the overlap,
#     and average the overlaps over time for tubes. """
#     parts, _ = split_tube_into_boxes(boxes)
#     query_parts, _ = split_tube_into_boxes(query_boxes)
#     print('parts :',parts[0])
#     print('query_parts :',query_parts[0])

#     a = torch.stack([
#         bbox_overlaps(
#             part, query_part)
#         for part, query_part in zip(parts, query_parts)], dim=1)
#     print('a :',a)
#     means = torch.zeros(boxes.size(0), query_boxes.size(0))
#     print('means.shape :',means.shape)
#     for i in range(a.size(0)):
#         indx= a[i,:,0].ne(-1).nonzero().view(-1)
#         if indx.sum() == 0:
#             continue
#         print('indx :',indx)
#         print('a[i,indx] :',a[i,indx])
#         print('a[i,indx] :',a[i,indx].shape)
#         means[i] = torch.mean(a[i,indx], dim=0)

#     # return torch.mean(torch.stack([
#     #     bbox_overlaps(
#     #         part, query_part)
#     #     for part, query_part in zip(parts, query_parts)]), dim=0)
#     return means

# def tube_overlaps(boxes, query_boxes):
#     """ Use the cython overlap implementation to compute the overlap,
#     and average the overlaps over time for tubes. """
#     parts, _ = split_tube_into_boxes(boxes)
#     query_parts, _ = split_tube_into_boxes(query_boxes)
#     a =  torch.stack([
#         bbox_overlaps(
#             part, query_part)
#         for part, query_part in zip(parts, query_parts)])
#     print('a :',a)
#     print('torch.mean(a, dim=0) :',torch.mean(a, dim=0))
#     exit(-1)

def tube_overlaps(boxes, query_boxes):
    """ Use the cython overlap implementation to compute the overlap,
    and average the overlaps over time for tubes. """
    parts, _ = split_tube_into_boxes(boxes)
    query_parts, _ = split_tube_into_boxes(query_boxes)
    overlaps= torch.mean(torch.stack([
        bbox_overlaps(
            part, query_part)
        for part, query_part in zip(parts, query_parts)]), dim=0)

    # check for empty boxes
    idx = query_boxes.eq(0).all(dim=1)
    if idx.nelement() != 0:
        overlaps.masked_fill_(idx.view(
             1,idx.size(0),).expand( overlaps.size(0),idx.size(0)), -1)
    return overlaps

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

    # overlaps.masked_fill_(gt_area_zero.view(
    #     1, K).expand(N, K), -1)
    # overlaps.masked_fill_(anchors_area_zero.view(
    #     N, 1).expand( N, K), 1)

    return overlaps

if __name__ == '__main__':

   t = torch.Tensor([[43., 59., 55., 80., 43., 59., 55., 80., 44., 60., 56., 81., 44., 60.,
                      56., 81.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                      0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                      0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                      0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 22.],
                     [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
                        0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
                        0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
                        0.,   0.,   0.,   0.,  70.,  15., 112.,  83.,  70.,  15., 112.,  83.,
                        70.,  15., 112.,  83.,  70.,  15., 112.,  83.,  70.,  16., 112.,  84.,
                        70.,  16., 112.,  84., 22]])

   # t2 = torch.Tensor([[43., 59., 55., 80., 43., 59., 55., 80., 44., 60., 56., 81., 44., 60.,
   #                    56., 81., 32.,  32.,  21.,  21.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
   #                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
   #                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
   #                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 22.]])
   # t2= torch.Tensor([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
   #                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
   #                    31., 29., 54., 88., 31., 29., 54., 88., 31., 29., 54., 88., 31., 29.,
   #                    54., 89., 31., 29., 54., 91., 31., 29., 54., 91., 31., 29., 54., 91.,
   #                     31., 29., 54., 91., 31., 29., 54., 91.],
   #                   [43., 59., 55., 80., 43., 59., 55., 80., 44., 60., 56., 81., 44., 60.,
   #                    56., 81., 32.,  32.,  21.,  21.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
   #                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
   #                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
   #                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
   #                   [43., 59., 55., 80., 43., 59., 55., 80., 44., 60., 56., 81., 44., 60.,
   #                    56., 81., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
   #                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
   #                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
   #                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
   # t = torch.Tensor([[ 9.0530, 38.6012, 34.7928, 70.8962,  9.0173, 38.7589, 35.0426, 70.9730,
   #                     12.0831, 38.8715, 38.1236, 71.0357,  8.8917, 38.1383, 35.1799, 70.4063,
   #                     11.9164, 41.9390, 38.2064, 74.1712, 11.8824, 41.8906, 38.1435, 74.0142,
   #                     11.8848, 38.2370, 38.0786, 70.4760, 11.8955, 41.4256, 38.1724, 73.4482,
   #                     11.7485, 38.7613, 38.1819, 70.9632, 12.3215, 40.8892, 38.8478, 73.7179,
   #                     12.5459, 38.8393, 38.9338, 70.9604, 11.8656, 38.4042, 38.3014, 70.5136,
   #                     12.4879, 38.3381, 38.9249, 70.4038, 12.4635, 38.2855, 38.8916, 70.3645,
   #                     12.4841, 38.9793, 38.8986, 70.9616, 12.0203, 38.6086, 38.3970, 71.3812]])
   
   t = torch.Tensor([[  0.,   0., 111., 111.,   0.,   0., 111., 111.,   0.,   0., 111., 111.,
          0.,   0., 111., 111.,   0.,   0., 111., 111.,   0.,   0., 111., 111.,
          0.,   0., 111., 111.,   0.,   0., 111., 111.,   0.,   0., 111., 111.,
          0.,   0., 111., 111.,   0.,   0., 111., 111.,   0.,   0., 111., 111.,
          0.,   0., 111., 111.,   0.,   0., 111., 111.,   0.,   0., 111., 111.,
          0.,   0., 111., 111.]])
   t2 = torch.Tensor([[26.4131, 34.5952, 51.8963, 82.4958, 28.3383, 34.5595, 53.7611, 82.5421,
                       26.3752, 34.3899, 51.8106, 82.6797, 28.8261, 34.2347, 54.2790, 82.3281,
                       26.5113, 34.3087, 52.0123, 82.5305, 26.4866, 34.2786, 51.8227, 82.6316,
                       26.3476, 34.3084, 51.8784, 82.6221, 26.3684, 34.1650, 51.8589, 82.3539,
                       26.3180, 34.4100, 51.9150, 82.4075, 26.4003, 35.1745, 51.9531, 83.2799,
                       26.4585, 34.5444, 52.0024, 82.7639, 26.3956, 34.6023, 52.0360, 82.6630,
                       26.3194, 34.6185, 52.0775, 82.7460, 26.1689, 34.5681, 52.0220, 82.6796,
                       26.1317, 34.7487, 51.8750, 82.7303, 26.1167, 34.3726, 51.8539, 82.5221],
                      [ 19.4274,  38.9712,  58.1258, 109.7607,  21.5536,  35.9292,  60.2362,
                        106.8209,  21.5160,  35.6418,  60.2032, 106.7502,  21.4934,  33.1030,
                        60.2165, 104.1366,  21.4343,  38.8930,  60.2570, 109.8877,  21.3701,
                        35.7653,  60.3119, 106.7561,  21.2878,  35.2686,  60.1935, 106.2343,
                        21.2896,  35.5014,  60.2816, 106.5944,  21.3376,  35.4243,  60.3645,
                        106.6022,  19.4614,  38.8632,  58.5182, 110.0001,  21.3308,  35.8074,
                        60.4249, 107.0485,  19.6492,  35.9255,  58.7028, 107.0415,  19.4882,
                        35.6573,  58.7181, 106.6235,  19.4267,  35.7150,  58.6332, 106.8955,
                        19.4469,  33.3863,  58.6122, 104.5442,  19.4958,  36.0304,  58.7064,
                        106.9198],
                      [ 30.6485,  48.1194,  52.7715,  90.4644,  27.4066,  48.0781,  49.6746,
                        90.5801,  30.8615,  48.2356,  53.0086,  90.4472,  26.5923,  48.4513,
                        48.7919,  91.1135,  30.6483,  48.2552,  52.8131,  90.7425,  30.6722,
                        48.2809,  53.0539,  90.6190,  30.7959,  48.3056,  52.9889,  90.6300,
                        30.8562,  48.6339,  53.1470,  91.0265,  30.8433,  48.4736,  53.0175,
                        91.0106,  30.6295,  47.3822,  52.9647,  89.6620,  30.5373,  48.4853,
                        52.9980,  90.5986,  30.5804,  48.4894,  52.9732,  90.7172,  30.6360,
                        48.5639,  52.9665,  90.6922,  30.7302,  48.6981,  53.0436,  90.7674,
                        30.8120,  48.5941,  53.3768,  90.6947,  30.7936,  49.3361,  53.3917,
                        90.8810],
                      [  0.0000,   0.0000,  66.3041, 111.0000,   0.0000,   0.0000, 111.0000,
                         69.9460,   0.0000,   0.0000, 111.0000,  79.2088,   0.0000,   0.0000,
                         111.0000,   0.0000,   0.0000,   0.0000, 111.0000, 111.0000,   0.0000,
                         0.0000, 111.0000,  55.1706,   0.0000,   0.0000, 111.0000,  36.9221,
                         0.0000,   0.0000, 111.0000,  59.2223,   0.0000,   0.0000, 111.0000,
                         60.8410,   0.0000,   0.0000,  61.4616, 111.0000,   0.0000,   0.0000,
                         111.0000,  81.9855,   0.0000,   0.0000,  59.4986,  70.0075,   0.0000,
                         0.0000,  67.1606,  54.0449,   0.0000,   0.0000,  63.9623,  84.8163,
                         0.0000,   0.0000,  60.2225,  13.2250,   0.0000,   0.0000,  65.6582,
                         71.1377],
                      [  8.1401,  19.4190,  43.9165,  85.4130,   3.8552,  28.2285,  39.6992,
                         94.3184,   3.8379,  28.3362,  39.7176,  94.3971,   3.8447,  28.4923,
                         39.7253,  94.7023,   3.6827,  19.0840,  39.7149,  85.3366,   3.5968,
                         28.2146,  39.6541,  94.5199,   3.5918,  28.3124,  39.7166,  94.6301,
                         3.6583,  28.2691,  39.6902,  94.5141,   3.5718,  28.3046,  39.7444,
                         94.6057,   8.1109,  19.1810,  44.2996,  85.4892,   3.5129,  28.4391,
                         39.7523,  94.6108,   8.0921,  28.4664,  44.3829,  94.6808,   8.0551,
                         28.4014,  44.4039,  94.5948,   8.0622,  28.5600,  44.3780,  94.5970,
                         8.0842,  28.8667,  44.3890,  94.9430,   7.9199,  28.4963,  44.2931,
                         94.5194],
                      [ 13.7296,  32.9239,  36.4958,  76.2267,  11.7814,  32.8821,  34.6502,
                        76.3327,  13.8754,  32.9593,  36.6517,  76.2654,  11.2956,  33.0718,
                        34.1215,  76.6763,  13.7452,  32.9475,  36.5538,  76.4508,  13.7506,
                        32.9595,  36.6890,  76.3845,  13.8083,  32.9835,  36.6438,  76.3927,
                        13.8487,  33.1891,  36.7527,  76.6221,  13.8224,  33.1305,  36.6576,
                        76.6265,  13.6795,  32.5172,  36.6317,  75.8391,  13.6169,  33.1837,
                        36.6668,  76.4140,  13.6344,  33.2045,  36.6509,  76.4807,  13.6655,
                        33.2661,  36.6588,  76.4894,  13.7038,  33.3568,  36.7039,  76.5267,
                        13.7462,  33.3321,  36.9206,  76.4929,  13.7256,  33.8043,  36.9246,
                        76.5663],
                      [  0.0000,   0.0000,  66.1397, 111.0000,  13.5254,   0.0000, 111.0000,
                         47.7170,  14.4114,   0.0000, 111.0000,  52.5117,  12.8805,   0.0000,
                         111.0000,   0.0000,  15.2523,   0.0000, 111.0000, 111.0000,  12.8649,
                         0.0000, 111.0000,  36.8033,  15.4044,   0.0000, 111.0000,  21.8489,
                         6.8189,   0.0000, 111.0000,  38.5426,  13.5470,   0.0000, 111.0000,
                         39.3569,   0.0000,   0.0000,  63.2207, 111.0000,  14.8402,   0.0000,
                         111.0000,  54.9497,   0.0000,   0.0000,  62.8204,  47.0968,   0.0000,
                         0.0000,  67.8376,  35.0210,   0.0000,   0.0000,  65.0761,  55.8705,
                         0.0000,   0.0000,  62.4928,   0.0000,   0.0000,   0.0000,  67.2185,
                         48.0391],
                      [ 38.3542,   8.0930,  66.2151,  59.8139,  14.0020,  54.4603,  42.0555,
                        106.2266,  14.0739,  55.7281,  42.2418, 106.9548,  14.1546,  64.5452,
                        42.2697, 111.0000,  13.7203,   6.9920,  42.0681,  59.0862,  13.5539,
                        54.9774,  41.6900, 107.2625,  13.8466,  56.8845,  42.2889, 109.2975,
                        14.0392,  55.8271,  41.9817, 107.7504,  13.5754,  56.2076,  41.8875,
                        108.0533,  38.3092,   7.3330,  66.6502,  59.3614,  13.3872,  55.4255,
                        41.7578, 106.7947,  37.7066,  55.2111,  66.3214, 106.9867,  37.9176,
                        55.8387,  66.3006, 107.9144,  38.2496,  56.2387,  66.6282, 107.2780,
                        38.2485,  64.8233,  66.7308, 111.0000,  37.4137,  55.0416,  65.9588,
                        106.6763],
                      [ 23.7527,  54.3444,  50.3449,  86.0039,  23.6879,  54.5519,  50.6532,
                        86.1004,  28.2326,  54.7056,  55.1643,  86.1970,  23.5645,  53.6811,
                        50.8124,  85.3027,  28.0724,  59.0988,  55.2921,  90.6953,  28.0400,
                        59.0251,  55.1999,  90.4819,  28.0545,  53.8150,  55.1075,  85.4451,
                        28.0953,  58.3322,  55.2492,  89.6794,  27.9084,  54.5482,  55.2453,
                        86.1346,  28.7186,  57.5832,  56.2071,  90.0363,  29.0526,  54.6325,
                        56.2960,  86.1163,  28.0366,  53.9777,  55.3818,  85.4525,  28.9548,
                        53.8510,  56.2928,  85.2732,  28.9130,  53.7588,  56.2326,  85.2067,
                        28.9294,  54.7158,  56.2157,  86.0445,  28.2571,  54.2012,  55.5066,
                        86.6287],
                      [  9.0530,  38.6012,  34.7928,  70.8962,   9.0173,  38.7589,  35.0426,
                         70.9730,  12.0831,  38.8715,  38.1236,  71.0357,   8.8917,  38.1383,
                         35.1799,  70.4063,  11.9164,  41.9390,  38.2064,  74.1712,  11.8824,
                         41.8906,  38.1435,  74.0142,  11.8848,  38.2370,  38.0786,  70.4760,
                         11.8955,  41.4256,  38.1724,  73.4482,  11.7485,  38.7613,  38.1819,
                         70.9632,  12.3215,  40.8892,  38.8478,  73.7179,  12.5459,  38.8393,
                         38.9338,  70.9604,  11.8656,  38.4042,  38.3014,  70.5136,  12.4879,
                         38.3381,  38.9249,  70.4038,  12.4635,  38.2855,  38.8916,  70.3645,
                         12.4841,  38.9793,  38.8986,  70.9616,  12.0203,  38.6086,  38.3970,
                         71.3812]])
   print('t.shape :',t.shape)
   print('t2.shape :',t2.shape)
   scores = torch.Tensor([[0.7495],
                          [0.7391],
                          [0.6810]])
   overlaps= tube_overlaps(t2,t)

   print('overlaps :',overlaps)

