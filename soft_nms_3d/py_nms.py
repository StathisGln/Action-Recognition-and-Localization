##############################################################
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
def py_cpu_nms_tubes(dets, thresh):
    """Pure Python NMS baseline."""
    T = (dets.shape[1] - 1) // 4

    areas = {}
    for t in range(T):
        x1 = dets[:, 4 * t + 0]
        y1 = dets[:, 4 * t + 1]
        x2 = dets[:, 4 * t + 2]
        y2 = dets[:, 4 * t + 3]

        areas[t] = (x2 - x1 + 1) * (y2 - y1 + 1)

    scores = dets[:, -1]
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        ovT = 0.0
        for t in range(T):
            xx1 = np.maximum(dets[i, 4 * t + 0], dets[order[1:], 4 * t + 0])
            yy1 = np.maximum(dets[i, 4 * t + 1], dets[order[1:], 4 * t + 1])
            xx2 = np.minimum(dets[i, 4 * t + 2], dets[order[1:], 4 * t + 2])
            yy2 = np.minimum(dets[i, 4 * t + 3], dets[order[1:], 4 * t + 3])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h

            # print('inter.shape :',inter.shape)
            # print('areas[t].shape :',areas[t].shape)
            # print('areas[t][i].shape :',areas[t][i])
            # print('areas[t][i].shape :',areas[t][order[1:]].shape)

            ovr = inter / (areas[t][i] + areas[t][order[1:]] - inter)
            ovT += ovr
        ovT /= T

        inds = np.where(ovT <= thresh)[0]
        order = order[inds + 1]

    return keep

def py_cpu_softnms(dets, scores, Nt=0.3, sigma=0.5, thresh=0.001, method=2):
    """
    py_cpu_softnms
    :param dets:   boexs format [x1, y1, x2, y2]

    """
    # print('dets.shape :',dets.shape)
    # indexes concatenate boxes with the last column
    N = dets.shape[0]
    indexes = torch.arange(N)

    T = dets.size(1) 

    zero_frames = dets.eq(0).all(dim=2)

    # _, order = torch.sort(scores, descending=True)

    x1 = dets[:,:, 0]
    y1 = dets[:,:, 1]
    x2 = dets[:,:, 2]
    y2 = dets[:,:, 3]
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N-1):

        # intermediate parameters for later parameters exchange
        tBD = dets[i, :].clone().detach()
        tidx = indexes[i].clone().detach()
        tscore = scores[i].clone().detach()
        tarea = areas[i].clone().detach()
        pos = i + 1
        
        max_score,max_pos = torch.max(scores[pos:], dim=0)

        if tscore < max_score:

            dets[i] = dets[max_pos + i + 1]
            dets[max_pos + i + 1, :] = tBD
            tBD = dets[i, :].clone().detach()

            indexes[i] = indexes[max_pos + i + 1]
            indexes[max_pos + i + 1] = tidx
            tidx = indexes[i].clone().detach()

            scores[i] = scores[max_pos + i + 1]
            scores[max_pos + i + 1] = tscore
            tscore = scores[i].clone().detach()

            areas[i] = areas[max_pos + i + 1]
            areas[max_pos + i + 1] = tarea
            tarea = areas[i].clone().detach()

        ovT = 0.0

        xx1 = torch.max(dets[i,:, 0], dets[pos:, :, 0])
        yy1 = torch.max(dets[i,:, 1], dets[pos:, :, 1])
        xx2 = torch.min(dets[i,:, 2], dets[pos:, :, 2])
        yy2 = torch.min(dets[i,:, 3], dets[pos:, :, 3])

        w = (xx2 - xx1 + 1).clamp_(min=0)
        h = (yy2 - yy1 + 1).clamp_(min=0)
        inter = w * h
        ua  = areas[i] + areas[pos:] - inter

        ovr = inter / ua

        zero_area = zero_frames[i] & zero_frames[pos:]
        n_empty = zero_area.sum(dim=1)


        ovr.masked_fill_(zero_area,0)
        ovT = ovr.sum(dim=1) / (T- n_empty.type_as(ovr))

        # Three methods: 1.linear 2.gaussian 3.original NMS
        
        if method == 1:  # linear

            weight = torch.ones(ovT.shape).type_as(scores)
            indices = (ovT > Nt).nonzero().view(-1)
            weight[indices] = weight[indices] - ovT[indices]

        # elif method == 2:  # gaussian
        #     weight = np.exp(-(ovT * ovT) / sigma)
        else:  # original NMS
            weight = torch.ones(ovT.shape).type_as(scores)
            weight[ovT > Nt] = 0

        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes

    return indexes

def py_cpu_temp_softnms(dets, scores, Nt=0.3, sigma=0.5, thresh=0.001, method=2):
    """
    py_cpu_softnms
    :param dets:   boexs format [x1, y1, x2, y2]

    """
    print('dets.shape :',dets.shape)
    # indexes concatenate boxes with the last column
    N = dets.shape[0]
    indexes = torch.arange(N)

    zero_frames = dets.eq(0).all(dim=1)

    # _, order = torch.sort(scores, descending=True)

    x1 = dets[:, 0]
    x2 = dets[:, 1]
    
    areas = (x2 - x1 + 1) 

    for i in range(N-1):

        # intermediate parameters for later parameters exchange
        tBD = dets[i, :].clone().detach()
        tidx = indexes[i].clone().detach()
        tscore = scores[i].clone().detach()
        tarea = areas[i].clone().detach()
        pos = i + 1
        
        max_score,max_pos = torch.max(scores[pos:], dim=0)

        if tscore < max_score:

            dets[i] = dets[max_pos + i + 1]
            dets[max_pos + i + 1, :] = tBD
            tBD = dets[i, :].clone().detach()

            indexes[i] = indexes[max_pos + i + 1]
            indexes[max_pos + i + 1] = tidx
            tidx = indexes[i].clone().detach()

            scores[i] = scores[max_pos + i + 1]
            scores[max_pos + i + 1] = tscore
            tscore = scores[i].clone().detach()

            areas[i] = areas[max_pos + i + 1]
            areas[max_pos + i + 1] = tarea
            tarea = areas[i].clone().detach()

        ovT = 0.0

        xx1 = torch.max(dets[i, 0], dets[pos:, 0])
        xx2 = torch.min(dets[i, 1], dets[pos:, 1])

        inter = (xx2 - xx1 + 1).clamp_(min=0)
        ua  = areas[i] + areas[pos:] - inter

        ovr = inter / ua
        zero_area = zero_frames[i] & zero_frames[pos:]
        print('zero_area :',zero_area)
        print('zero_area :',zero_area.shape)
        print('n_empty :',zero_area.sum(dim=0))

        ovr.masked_fill_(zero_area,0)

        # Three methods: 1.linear 2.gaussian 3.original NMS
        
        if method == 1:  # linear

            weight = torch.ones(ovr.shape).type_as(scores)
            indices = (ovr > Nt).nonzero().view(-1)
            weight[indices] = weight[indices] - ovr[indices]

        elif method == 2:  # gaussian
            weight = torch.exp(-(ovr * ovr) / sigma)
        else:  # original NMS
            weight = torch.ones(ovr.shape).type_as(scores)
            weight[ovr > Nt] = 0

        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes

    return indexes
