# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------

import torch
import numpy as np
import pdb


def bbox_transform(ex_rois, gt_rois):
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


# def bbox_transform_time(ex_rois, gt_rois):
#     '''
#     ex_rois are the anchors shape [(
#     gt_rois are the tubes shape [(0->x1),(1->y1),(2->t1),(3->x2),(4->y2),(5->t2)]
#     '''
#     if ex_rois.dim() == 2:
#         print('ex_rois.shape :',ex_rois.shape)
#         print('gt_rois.shape :',gt_rois.shape)

#         ex_widths = ex_rois[:, 3] - ex_rois[:, 0] + 1.0
#         ex_heights = ex_rois[:, 4] - ex_rois[:, 1] + 1.0
#         ex_time = ex_rois[:,5] - ex_rois[:,2] + 1 # for time dim

#         ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
#         ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights
#         ex_ctr_t = ex_rois[:, 2] + 0.5 * ex_time  # center t

#         ## groundtruth rois
#         gt_widths = gt_rois[:, :, 3] - gt_rois[:, :, 0] + 1.0
#         gt_heights = gt_rois[:, :, 4] - gt_rois[:, :, 1] + 1.0
#         gt_time = gt_rois[:, :, 2] - gt_rois[:, :, 2] + 1.0

#         gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_widths
#         gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_heights
#         gt_ctr_t = gt_rois[:, :, 2] + 0.5 * gt_heights

#         targets_dx = (gt_ctr_x - ex_ctr_x.view(1, -
#                                                1).expand_as(gt_ctr_x)) / ex_widths
#         targets_dy = (gt_ctr_y - ex_ctr_y.view(1, -
#                                                1).expand_as(gt_ctr_y)) / ex_heights
#         targets_dt = (gt_ctr_t - ex_ctr_t.view(1, -
#                                                1).expand_as(gt_ctr_t)) / ex_time

#         targets_dw = torch.log(
#             gt_widths / ex_widths.view(1, -1).expand_as(gt_widths))
#         targets_dh = torch.log(
#             gt_heights / ex_heights.view(1, -1).expand_as(gt_heights))
#         targets_dh = torch.log(
#             gt_time / ex_time.view(1, -1).expand_as(gt_time))

#         print('For x1 : \n\tex_widths {}, \n\tex_ctr_x {}, \n\tgt_widths {}, \n\tgt_ctr_x {}, \n\ttargets_dx {}'.format(
#             ex_widths, ex_ctr_x, gt_widths, gt_ctr_x, targets_dx))
#         print('For x1 : \n\tex_height {}, \n\tex_ctr_y {}, \n\tgt_heights {}, \n\tgt_ctr_y {}, \n\ttargets_dy {}'.format(
#             ex_heights, ex_ctr_y, gt_heights, gt_ctr_y, targets_dy))
#         print('For x1 : \n\tex_time {}, \n\tex_ctr_t {}, \n\tgt_time {}, \n\tgt_ctr_t {}, \n\ttargets_dt {}'.format(
#             ex_time, ex_ctr_t, gt_time, gt_ctr_t, targets_dt))
        
#     elif ex_rois.dim() == 3:
#         ex_widths = ex_rois[:, :, 2] - ex_rois[:, :, 0] + 1.0
#         ex_heights = ex_rois[:, :, 3] - ex_rois[:, :, 1] + 1.0
#         ex_ctr_x = ex_rois[:, :, 0] + 0.5 * ex_widths
#         ex_ctr_y = ex_rois[:, :, 1] + 0.5 * ex_heights

#         gt_widths = gt_rois[:, :, 2] - gt_rois[:, :, 0] + 1.0
#         gt_heights = gt_rois[:, :, 3] - gt_rois[:, :, 1] + 1.0
#         gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_widths
#         gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_heights

#         targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
#         targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
#         targets_dw = torch.log(gt_widths / ex_widths)
#         targets_dh = torch.log(gt_heights / ex_heights)
#     else:
#         raise ValueError('ex_roi input dimension is not correct.')

#     targets = torch.stack(
#         (targets_dx, targets_dy, targets_dw, targets_dh), 2)

#     print('tagets :', targets)

#     return targets


def bbox_transform_batch(ex_rois, gt_rois):

    if ex_rois.dim() == 2:
        ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
        ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
        ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
        ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

        # print('ex_widths :',ex_widths)
        # print('ex_heights :',ex_heights)
        # print('ex_ctr_x :',ex_ctr_x)
        # print('ex_ctr_y :',ex_ctr_y)

        gt_widths = gt_rois[:, :, 2] - gt_rois[:, :, 0] + 1.0
        # print('gt_rois[:,:,2] :',gt_rois[:,:,2])
        # print('gt_rois[:,:,0] :',gt_rois[:,:,0])
        gt_heights = gt_rois[:, :, 3] - gt_rois[:, :, 1] + 1.0
        gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_widths
        gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_heights


        # print('gt_widths :',gt_widths)
        # print('gt_heights :',gt_heights)
        # print('gt_ctr_x :',gt_ctr_x)
        # print('gt_ctr_y :',gt_ctr_y)
        targets_dx = (gt_ctr_x - ex_ctr_x.view(1, -
                                               1).expand_as(gt_ctr_x)) / ex_widths
        targets_dy = (gt_ctr_y - ex_ctr_y.view(1, -
                                               1).expand_as(gt_ctr_y)) / ex_heights
        targets_dw = torch.log(
            gt_widths / ex_widths.view(1, -1).expand_as(gt_widths))
        targets_dh = torch.log(
            gt_heights / ex_heights.view(1, -1).expand_as(gt_heights))

        # print('For x : \n\tex_widths {}, \n\tex_ctr_x {}, \n\tgt_widths {}, \n\tgt_ctr_x {}, \n\ttargets_dx {}, \n\ttargets_dw {}'.format(
        #     ex_widths, ex_ctr_x, gt_widths, gt_ctr_x, targets_dx, targets_dw))
        # print('For y : \n\tex_height {}, \n\tex_ctr_y {}, \n\tgt_heights {}, \n\tgt_ctr_y {}, \n\ttargets_dy {}, \n\ttargets_dh {}'.format(
        #     ex_heights, ex_ctr_y, gt_heights, gt_ctr_y, targets_dy, targets_dh))

    elif ex_rois.dim() == 3:
        ex_widths = ex_rois[:, :, 2] - ex_rois[:, :, 0] + 1.0
        ex_heights = ex_rois[:, :, 3] - ex_rois[:, :, 1] + 1.0
        ex_ctr_x = ex_rois[:, :, 0] + 0.5 * ex_widths
        ex_ctr_y = ex_rois[:, :, 1] + 0.5 * ex_heights

        gt_widths = gt_rois[:, :, 2] - gt_rois[:, :, 0] + 1.0
        gt_heights = gt_rois[:, :, 3] - gt_rois[:, :, 1] + 1.0
        gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_widths
        gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_heights

        targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = torch.log(gt_widths / ex_widths)
        targets_dh = torch.log(gt_heights / ex_heights)

    else:
        raise ValueError('ex_roi input dimension is not correct.')

    targets = torch.stack(
        (targets_dx, targets_dy, targets_dw, targets_dh), 2)

    # print('targets :',targets)
    # print('targets.shape :',targets.shape)

    return targets

def bbox_transform_batch_3d(ex_rois, gt_rois):

    if ex_rois.dim() == 2:
        ex_widths = ex_rois[:, 3] - ex_rois[:, 0] + 1.0
        ex_heights = ex_rois[:, 4] - ex_rois[:, 1] + 1.0
        ex_times = ex_rois[:, 5] - ex_rois[:, 2] + 1.0

        ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
        ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights
        ex_ctr_t = ex_rois[:, 2] + 0.5 * ex_heights

        gt_widths = gt_rois[:, :, 3] - gt_rois[:, :, 0] + 1.0
        gt_heights = gt_rois[:, :, 4] - gt_rois[:, :, 1] + 1.0
        gt_times = gt_rois[:, :, 5] - gt_rois[:, :, 2] + 1.0

        gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_widths
        gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_heights
        gt_ctr_t = gt_rois[:, :, 2] + 0.5 * gt_heights

        targets_dx = (gt_ctr_x - ex_ctr_x.view(1, -
                                               1).expand_as(gt_ctr_x)) / ex_widths
        targets_dy = (gt_ctr_y - ex_ctr_y.view(1, -
                                               1).expand_as(gt_ctr_y)) / ex_heights
        targets_dt = (gt_ctr_t - ex_ctr_t.view(1, -
                                               1).expand_as(gt_ctr_t)) / ex_times

        targets_dw = torch.log(
            gt_widths / ex_widths.view(1, -1).expand_as(gt_widths))
        targets_dh = torch.log(
            gt_heights / ex_heights.view(1, -1).expand_as(gt_heights))

        targets_dt = torch.log(
            gt_times / ex_times.view(1, -1).expand_as(gt_times))

    elif ex_rois.dim() == 3:
        ex_widths = ex_rois[:, :, 2] - ex_rois[:, :, 0] + 1.0
        ex_heights = ex_rois[:, :, 3] - ex_rois[:, :, 1] + 1.0
        ex_ctr_x = ex_rois[:, :, 0] + 0.5 * ex_widths
        ex_ctr_y = ex_rois[:, :, 1] + 0.5 * ex_heights

        gt_widths = gt_rois[:, :, 2] - gt_rois[:, :, 0] + 1.0
        gt_heights = gt_rois[:, :, 3] - gt_rois[:, :, 1] + 1.0
        gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_widths
        gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_heights

        targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = torch.log(gt_widths / ex_widths)
        targets_dh = torch.log(gt_heights / ex_heights)
    else:
        raise ValueError('ex_roi input dimension is not correct.')

    targets = torch.stack(
        (targets_dx, targets_dy, targets_dt, targets_dw, targets_dh, targets_dt), 2)

    return targets



def bbox_transform_inv(boxes, deltas, batch_size):
    widths = boxes[:, :, 2] - boxes[:, :, 0] + 1.0
    heights = boxes[:, :, 3] - boxes[:, :, 1] + 1.0
    ctr_x = boxes[:, :, 0] + 0.5 * widths
    ctr_y = boxes[:, :, 1] + 0.5 * heights

    dx = deltas[:, :, 0::4]
    # print('dx[0] :', dx[0])
    # print('dx.shape :', dx.shape)
    dy = deltas[:, :, 1::4]
    dw = deltas[:, :, 2::4]
    dh = deltas[:, :, 3::4]
    # print('widths.shape :',widths.shape, 'widths.unsqueeze(2).shape :',widths.unsqueeze(2).shape)
    pred_ctr_x = dx * widths.unsqueeze(2) + ctr_x.unsqueeze(2)
    pred_ctr_y = dy * heights.unsqueeze(2) + ctr_y.unsqueeze(2)
    pred_w = torch.exp(dw) * widths.unsqueeze(2)
    pred_h = torch.exp(dh) * heights.unsqueeze(2)

    pred_boxes = deltas.clone()
    # x1
    pred_boxes[:, :, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, :, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, :, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, :, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

def bbox_frames_transform_inv(boxes, deltas, batch_size):
    # print('boxes.shape :',boxes.shape)
    # print('deltas.shape :',deltas.shape)
    
    widths = boxes[:, :, 2] - boxes[:, :, 0] + 1.0
    heights = boxes[:, :, 3] - boxes[:, :, 1] + 1.0
    ctr_x = boxes[:, :, 0] + 0.5 * widths
    ctr_y = boxes[:, :, 1] + 0.5 * heights

    # print('widths {}, heights {},  ctr_x.shape {}, ctr_y {} '.format(widths.shape, heights.shape, ctr_x.shape,ctr_y.shape))
    dx = deltas[:, :, 0::4]
    # print('dx[0] :', dx[0])
    # print('dx.shape :', dx.shape)
    dy = deltas[:, :, 1::4]
    dw = deltas[:, :, 2::4]
    dh = deltas[:, :, 3::4]
    # print('widths.shape :',widths.shape, 'widths.unsqueeze(2).shape :',widths.unsqueeze(2).shape)
    pred_ctr_x = dx * widths.unsqueeze(2) + ctr_x.unsqueeze(2)
    pred_ctr_y = dy * heights.unsqueeze(2) + ctr_y.unsqueeze(2)
    pred_w = torch.exp(dw) * widths.unsqueeze(2)
    pred_h = torch.exp(dh) * heights.unsqueeze(2)

    pred_boxes = deltas.clone()
    # x1
    pred_boxes[:, :, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, :, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, :, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, :, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes




def bbox_transform_inv_3d(boxes, deltas, batch_size):
    widths = boxes[:, :, 3] - boxes[:, :, 0] + 1.0
    heights = boxes[:, :, 4] - boxes[:, :, 1] + 1.0
    dur = boxes[:, :, 5] - boxes[:, :, 2] + 1.0  # duration
    ctr_x = boxes[:, :, 0] + 0.5 * widths
    ctr_y = boxes[:, :, 1] + 0.5 * heights
    ctr_t = boxes[:, :, 2] + 0.5 * dur  # center frame

    dx = deltas[:, :, 0::6]
    dy = deltas[:, :, 1::6]
    dt = deltas[:, :, 2::6]
    dw = deltas[:, :, 3::6]
    dh = deltas[:, :, 4::6]
    dd = deltas[:, :, 5::6]

    pred_ctr_x = dx * widths.unsqueeze(2) + ctr_x.unsqueeze(2)
    pred_ctr_y = dy * heights.unsqueeze(2) + ctr_y.unsqueeze(2)
    pred_ctr_t = dt * dur.unsqueeze(2) + ctr_t.unsqueeze(2)
    pred_w = torch.exp(dw) * widths.unsqueeze(2)
    pred_h = torch.exp(dh) * heights.unsqueeze(2)
    pred_t = torch.exp(dd) * dur.unsqueeze(2)

    pred_boxes = deltas.clone()
    # x1
    pred_boxes[:, :, 0::6] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, :, 1::6] = pred_ctr_y - 0.5 * pred_h
    # t1
    pred_boxes[:, :, 2::6] = pred_ctr_t - 0.5 * pred_t
    # x2
    pred_boxes[:, :, 3::6] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, :, 4::6] = pred_ctr_y + 0.5 * pred_h
    # t2
    pred_boxes[:, :, 5::6] = pred_ctr_t + 0.5 * pred_t

    return pred_boxes


def clip_boxes_batch(boxes, im_shape, batch_size):
    """
    Clip boxes to image boundaries.
    """
    num_rois = boxes.size(1)

    boxes[boxes < 0] = 0
    # batch_x = (im_shape[:,0]-1).view(batch_size, 1).expand(batch_size, num_rois)
    # batch_y = (im_shape[:,1]-1).view(batch_size, 1).expand(batch_size, num_rois)

    batch_x = im_shape[:, 1] - 1
    batch_y = im_shape[:, 0] - 1

    boxes[:, :, 0][boxes[:, :, 0] > batch_x] = batch_x
    boxes[:, :, 1][boxes[:, :, 1] > batch_y] = batch_y
    boxes[:, :, 2][boxes[:, :, 2] > batch_x] = batch_x
    boxes[:, :, 3][boxes[:, :, 3] > batch_y] = batch_y

    return boxes


def clip_boxes(boxes, im_shape, batch_size):

    for i in range(batch_size):
        boxes[i, :, 0::4].clamp_(0, im_shape[i, 1]-1)
        boxes[i, :, 1::4].clamp_(0, im_shape[i, 0]-1)
        boxes[i, :, 2::4].clamp_(0, im_shape[i, 1]-1)
        boxes[i, :, 3::4].clamp_(0, im_shape[i, 0]-1)

    return boxes


def clip_boxes_3d(boxes, im_shape, batch_size):

    for i in range(batch_size):
        boxes[i, :, 0::6].clamp_(0, im_shape[i, 1]-1)
        boxes[i, :, 1::6].clamp_(0, im_shape[i, 0]-1)
        boxes[i, :, 2::6].clamp_(0, im_shape[i, 2]-1)
        boxes[i, :, 3::6].clamp_(0, im_shape[i, 1]-1)
        boxes[i, :, 4::6].clamp_(0, im_shape[i, 0]-1)
        boxes[i, :, 5::6].clamp_(0, im_shape[i, 2]-1)

    return boxes


def bbox_overlaps(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 4) ndarray of float

    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.size(0)
    K = gt_boxes.size(0)
    gt_boxes_area = ((gt_boxes[:, 2] - gt_boxes[:, 0] + 1) *
                     (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)).view(1, K)

    anchors_area = ((anchors[:, 2] - anchors[:, 0] + 1) *
                    (anchors[:, 3] - anchors[:, 1] + 1)).view(N, 1)

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

    return overlaps

def bbox_overlaps_time(anchors, gt_boxes, time_limit):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (b, K, 5) ndarray of float

    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    batch_size = gt_boxes.size(0)

    # print('gt_bboxes.shape : {}'.format(gt_boxes.shape))
    # print('anchors.dim() : ', anchors.dim())
    if anchors.dim() == 2:

        N = anchors.size(0) # N is the size of anchors ==> 94 probably
        K = gt_boxes.size(1)

        # print('N : {}, K : {}'.format(N, K))
        # print('achors shape before view :', anchors.shape)
        anchors = anchors.view(1, N, 4).expand(batch_size, N, 4).contiguous()

        # print('achors shape after view :', anchors.shape)
        # print('gt_boxes.shape ', gt_boxes.shape)
        # print('gt_boxes ', gt_boxes[0])

        time_duration = gt_boxes[:,:,5] - gt_boxes[:,:,2] + 1
        time_duration = time_duration.squeeze()
        # print('time_duration :',time_duration)
        # print('time_duration.shape :',time_duration.shape)
        # print('time_limit == time_duration:',time_limit == time_duration)

        if time_limit == time_duration:
            gt_boxes = gt_boxes[:, :, [0,1,3,4]].contiguous()

            gt_boxes_x = (gt_boxes[:, :, 2] - gt_boxes[:, :, 0] + 1)
            gt_boxes_y = (gt_boxes[:, :, 3] - gt_boxes[:, :, 1] + 1)
            gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size, 1, K)
            # gt_boxes_area = (gt_boxes_x * gt_boxes_y)
            # gt_boxes_area = gt_boxes_area.view(batch_size, 1, K)
            # print('gt_boxes_x.shape : ', gt_boxes_x.shape)
            # print('gt_boxes_x : ', gt_boxes_x)
            # print('gt_boxes_area.shape :', gt_boxes_area.shape)
            # print('gt_boxes_area :', gt_boxes_area)


            anchors_boxes_x = (anchors[:, :, 2] - anchors[:, :, 0] + 1)
            anchors_boxes_y = (anchors[:, :, 3] - anchors[:, :, 1] + 1)
            anchors_area = (anchors_boxes_x * anchors_boxes_y).view(batch_size, N, 1)

            # print('anchors_boxes_x.shape :',anchors_boxes_x.shape)
            # print('anchors_boxes_x :',anchors_boxes_x)
            # print('anchors_area.shape :',anchors_area.shape)
            # print('anchors_area :',anchors_area)

            # print('gt_boxes_x == 1 : ', gt_boxes_x == 1)
            # print('gt_boxes_x :', gt_boxes_x)

            gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1) # padding boxes, remove them
            anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

            # print('gt_area_zero : ',gt_area_zero)

            boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)
            # print('gt_boxes.shape :', gt_boxes.shape)
            query_boxes = gt_boxes.view(batch_size, 1, K, 4)
            # print('query_boxes.shape :', query_boxes.shape)
            # print('query_boxes :', query_boxes)
            query_boxes = query_boxes.expand(batch_size, N, K, 4)
            # print('query_boxes.shape :', query_boxes.shape)
            # print('query_boxes :', query_boxes)
            iw = (torch.min(boxes[:, :, :, 2], query_boxes[:, :, :, 2]) -
                  torch.max(boxes[:, :, :, 0], query_boxes[:, :, :, 0]) + 1)
            iw[iw < 0] = 0

            ih = (torch.min(boxes[:, :, :, 3], query_boxes[:, :, :, 3]) -
                  torch.max(boxes[:, :, :, 1], query_boxes[:, :, :, 1]) + 1)
            ih[ih < 0] = 0
            ua = anchors_area + gt_boxes_area - (iw * ih)
            overlaps = iw * ih / ua

            # mask the overlap here.
            overlaps.masked_fill_(gt_area_zero.view(
                batch_size, 1, K).expand(batch_size, N, K), 0)
            overlaps.masked_fill_(anchors_area_zero.view(
                batch_size, N, 1).expand(batch_size, N, K), -1)
            # mask the overlap here.
        else:
            overlaps = torch.zeros(batch_size,N,K) # N -> number of anchors, K:number of gt_tubes
    else:
        raise ValueError('anchors input dimension is not correct.')


    return overlaps

def bbox_overlaps_rois(anchors, gt_boxes, time_limit):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (b, K, 5) ndarray of float

    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    batch_size = gt_boxes.size(0)
    # print('batch_size :', batch_size)
    # print('gt_bboxes.shape : {}'.format(gt_boxes.shape))
    # print('anchors.dim() : ', anchors.dim())
    if anchors.dim() == 2:

        N = anchors.size(0) # N is the size of anchors ==> 94 probably
        K = gt_boxes.size(1)

        # print('N : {}, K : {}'.format(N, K))
        # print('achors shape before view :', anchors.shape)
        anchors = anchors.view(1, N, 4).expand(batch_size, N, 4).contiguous()

        # print('achors shape after view :', anchors.shape)
        # print('gt_boxes.shape ', gt_boxes.shape)
        # print('gt_boxes ', gt_boxes[0])

        gt_boxes = gt_boxes[:, :, :4].contiguous()

        gt_boxes_x = (gt_boxes[:, :, 2] - gt_boxes[:, :, 0] + 1)
        gt_boxes_y = (gt_boxes[:, :, 3] - gt_boxes[:, :, 1] + 1)
        gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size, 1, K)
        # gt_boxes_area = (gt_boxes_x * gt_boxes_y)
        # gt_boxes_area = gt_boxes_area.view(batch_size, 1, K)
        # print('gt_boxes_x.shape : ', gt_boxes_x.shape)
        # print('gt_boxes_x : ', gt_boxes_x)
        # print('gt_boxes_area.shape :', gt_boxes_area.shape)
        # print('gt_boxes_area :', gt_boxes_area)


        anchors_boxes_x = (anchors[:, :, 2] - anchors[:, :, 0] + 1)
        anchors_boxes_y = (anchors[:, :, 3] - anchors[:, :, 1] + 1)
        anchors_area = (anchors_boxes_x * anchors_boxes_y).view(batch_size, N, 1)

        # print('anchors_boxes_x.shape :',anchors_boxes_x.shape)
        # print('anchors_boxes_x :',anchors_boxes_x)
        # print('anchors_area.shape :',anchors_area.shape)
        # print('anchors_area :',anchors_area)

        # print('gt_boxes_x == 1 : ', gt_boxes_x == 1)
        # print('gt_boxes_x :', gt_boxes_x)

        gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1) # padding boxes, remove them
        anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

        # print('gt_area_zero : ',gt_area_zero)

        boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)
        # print('gt_boxes.shape :', gt_boxes.shape)
        # print('boxes.shape :', boxes.shape)

        query_boxes = gt_boxes.view(batch_size, 1, K, 4)
        # print('query_boxes.shape :', query_boxes.shape)
        # print('query_boxes :', query_boxes)
        query_boxes = query_boxes.expand(batch_size, N, K, 4)
        # print('query_boxes.shape :', query_boxes.shape)
        # print('boxes.shape :',boxes.shape )
        iw = (torch.min(boxes[:, :, :, 2], query_boxes[:, :, :, 2]) -
              torch.max(boxes[:, :, :, 0], query_boxes[:, :, :, 0]) + 1)
        iw[iw < 0] = 0

        ih = (torch.min(boxes[:, :, :, 3], query_boxes[:, :, :, 3]) -
              torch.max(boxes[:, :, :, 1], query_boxes[:, :, :, 1]) + 1)
        ih[ih < 0] = 0
        ua = anchors_area + gt_boxes_area - (iw * ih)
        overlaps = iw * ih / ua

        # mask the overlap here.
        overlaps.masked_fill_(gt_area_zero.view(
            batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps.masked_fill_(anchors_area_zero.view(
            batch_size, N, 1).expand(batch_size, N, K), -1)
        # mask the overlap here.
    else:
        raise ValueError('anchors input dimension is not correct.')


    return overlaps



def bbox_overlaps_batch(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (b, K, 5) ndarray of float

    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    batch_size = gt_boxes.size(0)

    # print('gt_bboxes.shape : {}'.format(gt_boxes.shape))
    # print('anchors.dim() : ', anchors.dim())
    if anchors.dim() == 2:
        print('edwwww')
        N = anchors.size(0)
        K = gt_boxes.size(1)
        # print('N : {}, K : {}'.format(N, K))
        # print('achors shape before view :', anchors.shape)
        anchors = anchors.view(1, N, 4).expand(batch_size, N, 4).contiguous()
        # print('achors shape after view :', anchors.shape)
        # print('gt_boxes.shape ', gt_boxes.shape)
        # print('gt_boxes ', gt_boxes[0])
        gt_boxes = gt_boxes[:, :, [0,1,3,4]].contiguous()
        # print('gt_boxes.shape ', gt_boxes.shape)

        gt_boxes_x = (gt_boxes[:, :, 2] - gt_boxes[:, :, 0] + 1)
        gt_boxes_y = (gt_boxes[:, :, 3] - gt_boxes[:, :, 1] + 1)

        # print('gt_boxes_x.shape : ', gt_boxes_x.shape)
        gt_boxes_area = (gt_boxes_x * gt_boxes_y)
        # print('gt_boxes_area.shape :', gt_boxes_area.shape)
        gt_boxes_area = gt_boxes_area.view(batch_size, 1, K)
        # gt_boxes_area = gt_boxes_area.view(batch_size, 1, F, K)
        # print('gt_boxes_area.shape :', gt_boxes_area.shape)

        anchors_boxes_x = (anchors[:, :, 2] - anchors[:, :, 0] + 1)
        anchors_boxes_y = (anchors[:, :, 3] - anchors[:, :, 1] + 1)
        anchors_area = (anchors_boxes_x *
                        anchors_boxes_y).view(batch_size, N, 1)

        # print('gt_boxes_x == 1 : ', gt_boxes_x == 1)
        # print('gt_boxes_x :', gt_boxes_x)

        gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
        anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

        # print('gt_area_zero : ',gt_area_zero)
        boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)
        # print('gt_boxes.shape :', gt_boxes.shape)
        query_boxes = gt_boxes.view(batch_size, 1, K, 4)
        # print('query_boxes.shape :', query_boxes.shape)
        # print('query_boxes :', query_boxes)
        query_boxes = query_boxes.expand(batch_size, N, K, 4)
        # print('query_boxes.shape :', query_boxes.shape)
        # print('query_boxes :', query_boxes)
        iw = (torch.min(boxes[:, :, :, 2], query_boxes[:, :, :, 2]) -
              torch.max(boxes[:, :, :, 0], query_boxes[:, :, :, 0]) + 1)
        iw[iw < 0] = 0

        ih = (torch.min(boxes[:, :, :, 3], query_boxes[:, :, :, 3]) -
              torch.max(boxes[:, :, :, 1], query_boxes[:, :, :, 1]) + 1)
        ih[ih < 0] = 0
        ua = anchors_area + gt_boxes_area - (iw * ih)
        overlaps = iw * ih / ua

        # mask the overlap here.
        overlaps.masked_fill_(gt_area_zero.view(
            batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps.masked_fill_(anchors_area_zero.view(
            batch_size, N, 1).expand(batch_size, N, K), -1)
        print('overlaps.shape  after mask:', overlaps.shape)
        # mask the overlap here.


    elif anchors.dim() == 3:

        N = anchors.size(1)
        K = gt_boxes.size(1)

        if anchors.size(2) == 4:
            anchors = anchors[:, :, :4].contiguous()
        else:
            anchors = anchors[:, :, 1:5].contiguous()

        gt_boxes = gt_boxes[:, :, :4].contiguous()

        gt_boxes_x = (gt_boxes[:, :, 2] - gt_boxes[:, :, 0] + 1)
        gt_boxes_y = (gt_boxes[:, :, 3] - gt_boxes[:, :, 1] + 1)
        gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size, 1, K)

        anchors_boxes_x = (anchors[:, :, 2] - anchors[:, :, 0] + 1)
        anchors_boxes_y = (anchors[:, :, 3] - anchors[:, :, 1] + 1)
        anchors_area = (anchors_boxes_x *
                        anchors_boxes_y).view(batch_size, N, 1)

        gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
        anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

        boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)
        query_boxes = gt_boxes.view(
            batch_size, 1, K, 4).expand(batch_size, N, K, 4)

        iw = (torch.min(boxes[:, :, :, 2], query_boxes[:, :, :, 2]) -
              torch.max(boxes[:, :, :, 0], query_boxes[:, :, :, 0]) + 1)
        iw[iw < 0] = 0

        ih = (torch.min(boxes[:, :, :, 3], query_boxes[:, :, :, 3]) -
              torch.max(boxes[:, :, :, 1], query_boxes[:, :, :, 1]) + 1)
        ih[ih < 0] = 0
        ua = anchors_area + gt_boxes_area - (iw * ih)

        overlaps = iw * ih / ua

        # mask the overlap here.
        overlaps.masked_fill_(gt_area_zero.view(
            batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps.masked_fill_(anchors_area_zero.view(
            batch_size, N, 1).expand(batch_size, N, K), -1)
    else:
        raise ValueError('anchors input dimension is not correct.')

    return overlaps


# def bbox_overlaps_batch_3d(anchors, gt_boxes):
#     """
#     anchors: (N, 4) ndarray of float
#     gt_boxes: (b, K, 5) ndarray of float

#     overlaps: (N, K) ndarray of overlap between boxes and query_boxes
#     """
#     batch_size = gt_boxes.size(0)

#     print('gt_bboxes.shape : {}'.format(gt_boxes.shape))
#     print('anchors.dim() : ', anchors.dim())
#     if anchors.dim() == 2:

#         N = anchors.size(0)
#         K = gt_boxes.size(1)
#         print('N = {} K = {}'.format(N, K))
#         print('batch_size', batch_size)
#         anchors = anchors.view(1, N, 6)
#         print('anchors shape :', anchors.shape)
#         anchors = anchors.expand(batch_size, N, 6).contiguous()
#         print('anchors shape :', anchors.shape)
#         print('gt_boxes ', gt_boxes.shape)
#         print('gt_boxes ', gt_boxes[0])
#         gt_boxes = gt_boxes[:, :, :6].contiguous()
#         # gt_boxes = gt_boxes.squeeze(2).contiguous()
#         print('gt_boxes.shape ', gt_boxes.shape)

#         gt_boxes_x = (gt_boxes[:, :, 3] - gt_boxes[:, :, 0] + 1)
#         gt_boxes_y = (gt_boxes[:, :, 4] - gt_boxes[:, :, 1] + 1)
#         gt_boxes_t = (gt_boxes[:, :, 5] - gt_boxes[:, :, 2] + 1)
#         if batch_size == 1:  # only 1 video in batch:
#             gt_boxes_x = gt_boxes_x.unsqueeze(0)
#             gt_boxes_y = gt_boxes_y.unsqueeze(0)
#             gt_boxes_t = gt_boxes_t.unsqueeze(0)

#         print('gt_boxes_x.shape :', gt_boxes_x.shape)
#         print('gt_boxes_x :', gt_boxes_x)
#         gt_boxes_area = (gt_boxes_x * gt_boxes_y * gt_boxes_t)
#         # if batch_size == 1:  # only 1 video in batch:
#         #     print('first gt_boxes_area.shape :', gt_boxes_area.shape)
#         #     gt_boxes_area = gt_boxes_area.unsqueeze(0)
#         # print('gt_boxes_area.shape :',gt_boxes_area.shape)
#         # gt_boxes_area = gt_boxes_area.view(batch_size, 1,  K)
#         # print('gt_boxes_area.shape :',gt_boxes_area.shape)

#         anchors_boxes_x = (anchors[:, :, 3] - anchors[:, :, 0] + 1)
#         anchors_boxes_y = (anchors[:, :, 4] - anchors[:, :, 1] + 1)
#         anchors_boxes_t = (anchors[:, :, 5] - anchors[:, :, 2] + 1)
#         print('anchors_boxes_y : ', anchors_boxes_y.shape)

#         anchors_area = (anchors_boxes_x * anchors_boxes_y *
#                         anchors_boxes_t).view(batch_size, N, 1)  # for 1 frame
#         print('anchors_area.shape :', anchors_area.shape)

#         print('gt_boxes_x == 1 : ', gt_boxes_x == 1)
#         print('gt_boxes_x :', gt_boxes_x)
#         gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1) 
#         anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

#         boxes = anchors.view(batch_size, N, 1, 6)
#         # print('anchors : ', boxes)
#         boxes = boxes.expand(batch_size, N, K, 6)
#         print('gt_boxes.shape :', gt_boxes.shape)
#         # print('gt_boxes :', gt_boxes)
#         query_boxes = gt_boxes.view(batch_size, 1, K, 6)
#         print('query_boxes.shape :', query_boxes.shape)
#         query_boxes = query_boxes.expand(batch_size, N, K, 6)
#         print('query_boxes.shape :', query_boxes.shape)
#         # print('query_boxes[0] :', query_boxes[0][:])


#         iw = (torch.min(boxes[:, :, :, 3], query_boxes[:, :, :, 3]) -
#               torch.max(boxes[:, :, :, 0], query_boxes[:, :, :, 0]) + 1)
#         print(' iw.shape :', iw.shape)
#         # print(' iw :', iw)
#         iw[iw < 0] = 0

#         ih = (torch.min(boxes[:, :, :, 4], query_boxes[:, :, :, 4]) -
#               torch.max(boxes[:, :, :, 1], query_boxes[:, :, :, 1]) + 1)
#         ih[ih < 0] = 0

#         it = (torch.min(boxes[:, :, :, 5], query_boxes[:, :, :, 5]) -
#               torch.max(boxes[:, :, :, 2], query_boxes[:, :, :, 2]) + 1)
#         it[it < 0] = 0

#         print('anchors_area.shape : ', anchors_area.shape )
#         print('gt_boxes_area.shape :',gt_boxes_area.shape)
#         ua = anchors_area + gt_boxes_area - (iw * ih * it)
#         print('ua.shape :', ua.shape)
#         overlaps = iw * ih * it / ua
#         # print('overlaps :', overlaps)
#         print('overlaps.shape :', overlaps.shape)
#         # mask the overlap here.
#         print('gt_area_zero.shape :', gt_area_zero.shape, ' K {} N {}'.format(K,N))
#         overlaps.masked_fill_(gt_area_zero.view(
#             batch_size, 1, K).expand(batch_size, N, K), 0)
#         overlaps.masked_fill_(anchors_area_zero.view(
#             batch_size, N, 1).expand(batch_size, N, K), -1)
#         print('overlaps.shape after mask :', overlaps.shape)
#         # print('overlaps :', overlaps)
#     elif anchors.dim() == 3:
#         N = anchors.size(1)
#         K = gt_boxes.size(1)

#         if anchors.size(2) == 4:
#             anchors = anchors[:, :, :4].contiguous()
#         else:
#             anchors = anchors[:, :, 1:5].contiguous()

#         gt_boxes = gt_boxes[:, :, :4].contiguous()

#         gt_boxes_x = (gt_boxes[:, :, 2] - gt_boxes[:, :, 0] + 1)
#         gt_boxes_y = (gt_boxes[:, :, 3] - gt_boxes[:, :, 1] + 1)
#         gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size, 1, K)

#         anchors_boxes_x = (anchors[:, :, 2] - anchors[:, :, 0] + 1)
#         anchors_boxes_y = (anchors[:, :, 3] - anchors[:, :, 1] + 1)
#         anchors_area = (anchors_boxes_x *
#                         anchors_boxes_y).view(batch_size, N, 1)

#         gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
#         anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

#         boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)
#         query_boxes = gt_boxes.view(
#             batch_size, 1, K, 4).expand(batch_size, N, K, 4)

#         iw = (torch.min(boxes[:, :, :, 2], query_boxes[:, :, :, 2]) -
#               torch.max(boxes[:, :, :, 0], query_boxes[:, :, :, 0]) + 1)
#         iw[iw < 0] = 0

#         ih = (torch.min(boxes[:, :, :, 3], query_boxes[:, :, :, 3]) -
#               torch.max(boxes[:, :, :, 1], query_boxes[:, :, :, 1]) + 1)
#         ih[ih < 0] = 0
#         ua = anchors_area + gt_boxes_area - (iw * ih)

#         overlaps = iw * ih / ua

#         # mask the overlap here.
#         overlaps.masked_fill_(gt_area_zero.view(
#             batch_size, 1, K).expand(batch_size, N, K), 0)
#         overlaps.masked_fill_(anchors_area_zero.view(
#             batch_size, N, 1).expand(batch_size, N, K), -1)
#     else:
#         raise ValueError('anchors input dimension is not correct.')

#     return overlaps
