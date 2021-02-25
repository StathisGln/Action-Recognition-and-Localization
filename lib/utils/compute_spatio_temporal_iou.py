###########################################################################################################
# Get from                                                                                                #
# https://github.com/gurkirt/corrected-UCF101-Annots/blob/master/evaluation/compute_spatio_temporal_iou.m #
# and modify to python code                                                                               #
###########################################################################################################

# We are here talking about spatio-temporal detections, i.e. a set of ground-truth bounding boxes that
#  I will denote by g_t, with t between t_g^b and t_g^e (beginning and end time of the ground-truth)
# versus a detection which is also a set of bounding boxes, denoted by d_t, with t between t_d^e et t_d^e.
#
# a) temporal iou =  T_i / T_u
#  this is the intersection over union between the timing of the the tubes,
# ie mathematically T_i / T_u with
# the intersection T_i = max(0,   max(t_g^b,t_d^b)-min(t_d^e,t_g^e) )
# and the union T_u = min(t_g^b,t_d^b)-max(t_d^e,t_g^e)
#
# b) for each t between max(tgb,tdb)-min(tde,tge), we compute the IoU between g_t and d_t, and average them
#
# Multiplying (a) and (b) is the same as computed the average of the spatial iou over all frames in T_u of the two tubes, with a spatial iou of 0 for frames where only one box exists.
# c) as this is standard in detection problem, if there are multiple detections for the same groundtruth detection, the first one is counted as positive and the other ones as negatives

# gt_fnr = 1xn doube
# gt_bb = nx4 doubld - [x y w h]
# dt_fnr = 1xm double
# dt_bb = mx4 double - [x y w h]

import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_spatio_temporal_iou(gt_tubes, tubes):


    T =  tubes.shape[-1] // 4
    n_gt_tubes = gt_tubes.size(0)
    n_tubes = tubes.size(0)

    gt_tubes = gt_tubes.view(n_gt_tubes,T,4)
    tubes = tubes.view(n_tubes,T,4)

    print('T :',T)
    print('gt_tubes.shape :',gt_tubes.shape)
    print('tubes.shape :',tubes.shape)

    gt_fr_b = torch.zeros(n_gt_tubes)
    gt_fr_e = torch.zeros(n_gt_tubes)

    t_fr_b = torch.zeros(n_tubes)
    t_fr_e = torch.zeros(n_tubes)
    
    for i in range(n_gt_tubes):
        non_empty_gt = gt_tubes[i].ne(0).all(dim=1).nonzero()
        if non_empty_gt.nelement() == 0:
            gt_fr_b[i] = -1
            gt_fr_e[i] = -1
        else:
            gt_fr_b[i] = non_empty_gt[0]
            gt_fr_e[i] = non_empty_gt[-1]

    for i in range(n_tubes):
        non_empty_t = tubes[i].ne(0).all(dim=1).nonzero()
        if non_empty_gt.nelement() == 0:
            t_fr_b[i] = -1
            t_fr_e[i] = -1
        else:
            t_fr_b[i] = non_empty_t[0]
            t_fr_e[i] = non_empty_t[-1]
            
    gt_fr_b = gt_fr_b.view(1,n_gt_tubes).expand( n_tubes, n_gt_tubes)
    gt_fr_e = gt_fr_e.view(1,n_gt_tubes).expand( n_tubes, n_gt_tubes)

    t_fr_b = t_fr_b.view(n_tubes,1).expand(n_tubes, n_gt_tubes)
    t_fr_e = t_fr_e.view(n_tubes,1).expand(n_tubes, n_gt_tubes)


    T_i = ( torch.min(gt_fr_e, t_fr_e) - torch.max(gt_fr_b, t_fr_b)+1).clamp_(min=0)
    T_u = ( torch.max(gt_fr_e, t_fr_e) - torch.min(gt_fr_b, t_fr_b)+1).clamp_(min=0)

    T_iou = T_i / T_u

    for i in range(n_tubes):
        for j in range(n_gt_tubes):
            frames = torch.range(torch.max(gt_fr_b[i,j], t_fr_b[i,j]), torch.min(gt_fr_e[i,j], t_fr_e[i,j])).long()
            overlap = 0
            for z in range(len(frames)):

                w = bbox_overlaps(tubes[i,frames[z]].unsqueeze(0), gt_tubes[j,frames[z]].unsqueeze(0))
                overlap += w
            overlap = overlap / len(frames)
            overlap = overlap * T_iou[i].type_as(overlap)
            print('new_overlap :',overlap)

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


def tube_overlaps(boxes, query_boxes):
    """ Use the cython overlap implementation to compute the overlap,
    and average the overlaps over time for tubes. """
    parts, _ = split_tube_into_boxes(boxes)
    query_parts, _ = split_tube_into_boxes(query_boxes)
    a = torch.stack([
        bbox_overlaps(
            part, query_part)
        for part, query_part in zip(parts, query_parts)])
    print('a :',a)

    non_zero = a.ne(-1.0).sum(0).float()
    print('non_zero :',non_zero)

    sums = a.clamp_(min=0).sum(0)
    print('sums :',sums)
    overlaps = sums/non_zero
    
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

    anchors_area_zero = anchors_area_zero.view(N, 1).expand( N, K)
    gt_area_zero = gt_area_zero.view( 1, K).expand(N, K)

    zero_area =  (gt_area_zero == 1)

    overlaps.masked_fill_(zero_area, -1)

    return overlaps
if __name__ == '__main__' :

    t = torch.Tensor([[ 27.6192, 31.4147, 48.0159, 72.3430, 27.7839, 32.2596, 48.4604,
        72.9832, 27.4605, 32.2294, 47.9650, 72.6398, 27.2034, 31.8159, 47.4094,
        71.0437,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
         0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
         0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
         0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
         0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
         0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                        0.0000,  ],
    [ 27.6192, 31.4147, 48.0159, 72.3430, 27.7839, 32.2596, 48.4604,
        72.9832, 27.4605, 32.2294, 47.9650, 72.6398, 27.2034, 31.8159, 47.4094,
        71.0437,  10.0000,  20.0000,  14.0000,  40.0000,  0.0000,  0.0000,  0.0000,
         0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
         0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
         0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
         0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
         0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
         0.0000,  ],
        [ 27.6192, 31.4147, 48.0159, 72.3430, 27.7839, 32.2596, 48.4604,
        72.9832, 27.4605, 32.2294, 47.9650, 72.6398, 27.2034, 31.8159, 47.4094,
        71.0437,  10.0000,  20.0000,  14.0000,  40.0000,  40.0000,  10.0000,  50.0000,
         20.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
         0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
         0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
         0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
         0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
         0.0000,  ]]).cuda()

    gt = torch.Tensor([[[18., 30., 45., 76., 23.],
                       [18., 30., 45., 76., 23.],
                       [19., 31., 46., 77., 23.],
                       [19., 31., 46., 77., 23.],
                       [19., 31., 46., 77., 23.],
                       [19., 31., 46., 77., 23.],
                       [21., 31., 48., 77., 23.],
                       [18., 31., 45., 77., 23.],
                       [18., 31., 45., 77., 23.],
                       [11., 30., 46., 85., 23.],
                       [11., 30., 47., 85., 23.],
                       [11., 30., 47., 85., 23.],
                       [12., 30., 47., 85., 23.],
                       [12., 30., 47., 85., 23.],
                       [10., 30., 46., 85., 23.],
                       [12., 30., 47., 85., 23.]]]).cuda()

    gt = gt[:,:,:4].contiguous().view(-1,16*4).contiguous()
    t = t.view(-1,16*4).contiguous()
    print('t.shape :',t.shape)
    print('gt.shape :',gt.shape)
    iou1 = tube_overlaps(t,gt)
    print('iou1 :',iou1)
    iou = compute_spatio_temporal_iou(gt, t)
