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
        print('keep :',keep)
        ovT = 0.0
        for t in range(T):
            xx1 = np.maximum(dets[i, 4 * t + 0], dets[order[1:], 4 * t + 0])
            yy1 = np.maximum(dets[i, 4 * t + 1], dets[order[1:], 4 * t + 1])
            xx2 = np.minimum(dets[i, 4 * t + 2], dets[order[1:], 4 * t + 2])
            yy2 = np.minimum(dets[i, 4 * t + 3], dets[order[1:], 4 * t + 3])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[t][i] + areas[t][order[1:]] - inter)
            ovT += ovr
        ovT /= T

        inds = np.where(ovT <= thresh)[0]
        order = order[inds + 1]
        print('order :',order)
    return keep

def py_cpu_softnms(dets, sc, Nt=0.3, sigma=0.5, thresh=0.001, method=2):
    """
    py_cpu_softnms
    :param dets:   boexs format [x1, y1, x2, y2]

    """

    # indexes concatenate boxes with the last column
    N = dets.shape[0]
    indexes = np.array([np.arange(N)])
    dets = np.concatenate((dets, indexes.T), axis=1)

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

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tBD = dets[i, :].copy()
        tscore = scores[i].copy()
        tarea = areas[i].copy()
        pos = i + 1

        #
        if i != N-1:
            maxscore = np.max(scores[pos:], axis=0)
            maxpos = np.argmax(scores[pos:], axis=0)
        else:
            maxscore = scores[-1]
            maxpos = 0
        if tscore < maxscore:
            dets[i, :] = dets[maxpos + i + 1, :]
            dets[maxpos + i + 1, :] = tBD
            tBD = dets[i, :]

            scores[i] = scores[maxpos + i + 1]
            scores[maxpos + i + 1] = tscore
            tscore = scores[i]

            areas[i] = areas[maxpos + i + 1]
            areas[maxpos + i + 1] = tarea
            tarea = areas[i]

        ovT = 0.0
        for t in range(T):

            # IoU calculate
            xx1 = np.maximum(dets[i, 0], dets[pos:, 0])
            yy1 = np.maximum(dets[i, 1], dets[pos:, 1])
            xx2 = np.minimum(dets[i, 2], dets[pos:, 2])
            yy2 = np.minimum(dets[i, 3], dets[pos:, 3])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[pos:] - inter)
            ovT += ovr
        ovT /= T

        # Three methods: 1.linear 2.gaussian 3.original NMS
        if method == 1:  # linear
            weight = np.ones(ovT.shape)
            weight[ovT > Nt] = weight[ovT > Nt] - ovr[ovT > Nt]
        elif method == 2:  # gaussian
            weight = np.exp(-(ovT * ovT) / sigma)
        else:  # original NMS
            weight = np.ones(ovT.shape)
            weight[ovT > Nt] = 0

        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    inds = dets[:, -1][scores > thresh]
    keep = inds.astype(int)
    print(keep)

    return keep
