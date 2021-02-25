import torch
from torch.autograd import Function
from _ext import calc as c
# from .._ext import calc as c


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
    # use masks
    anchors_area_zero = anchors_area_zero.view(N, 1).expand( N, K)
    gt_area_zero = gt_area_zero.view( 1, K).expand(N, K)

    zero_area = (anchors_area_zero == 1) & (gt_area_zero == 1)

    overlaps.masked_fill_(zero_area, -1)

    return overlaps

class Tube_Overlaps(Function):

    def __init__(self,):
        ...

    def forward(self, boxes, query_boxes):

        N = boxes.size(0)
        K = query_boxes.size(0)
        
        parts, T = split_tube_into_boxes(boxes)
        query_parts, _ = split_tube_into_boxes(query_boxes)


        a = torch.stack([
            bbox_overlaps(
                part, query_part)
            for part, query_part in zip(parts, query_parts)])

        a = a.permute(2,1,0).contiguous().view(-1,T).contiguous()
        print('a :',a)
        means = torch.zeros((a.size(0))).type_as(a)
        # ## cpu version
        # for i in range(a.size(0)):
        #     sum = 0
        #     k = 0
        #     for j in range(T):
        #         if a[i,j] == -1.0:
        #             break
        #         sum += a[i,j]
        #         k += 1
        #     if k != 0:
        #         means[i] = sum / k
        #     else:
        #         means[i] = -1
        ## gpu version
        c.mean_overlaps(a.size(0), a.size(1),a,means)

        means = means.view(K,N)

        return means
        
if __name__ == '__main__':

    t = torch.Tensor([[0.0000,   0.0000, 111.0000, 103.8053,   0.0000,   0.0000, 111.0000,
                       100.9938,   0.0000,   0.0000, 111.0000,  87.6325,   0.0000,   0.0000,
                       111.0000, 103.7976,   10.0000,   10.0000,   102.0000,  102.0000,   0.0000,
                       0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                       0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                       0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                       0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                       0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                       0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                       0.0000],
                      [0.0000,   0.0000, 111.0000, 103.8053,   0.0000,   0.0000, 111.0000,
                       100.9938,   0.0000,   0.0000, 111.0000,  87.6325,   0.0000,   0.0000,
                       111.0000, 103.7976,   0.0000,   0.0000,   0.0000,  0.0000,   0.0000,
                       0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                       0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                       0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                       0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                       0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                       0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
                       0.0000],
                      [0.0000,   0.0000,   0.0000,   0.0000,
                       0.0000,   0.0000,   0.0000,   0.0000,
                       0.0000,   0.0000,   0.0000,   0.0000,
                       0.0000,   0.0000,   0.0000,   0.0000,
                       0.0000,   0.0000,   0.0000,   0.0000,
                       0.0000,   0.0000,   0.0000,   0.0000,
                       0.0000,   0.0000,   0.0000,   0.0000,
                       0.0000,   0.0000, 111.0000,  98.5491,
                       0.0000,   0.0000, 111.0000, 100.8000,
                       0.0000,   0.0000, 111.0000,  98.3274,
                       0.0000,   0.0000, 111.0000,  96.4514,
                       0.0000,   0.0000, 111.0000,  89.3754,
                       0.0000,   0.0000, 111.0000, 111.0000,
                       0.0000,   0.0000, 111.0000, 108.5003,
                       0.0000,   0.0000, 111.0000,  98.3336,
                       0.0000,   0.0000,   0.0000,   0.0000]]).cuda()
    # t = torch.Tensor([[0.0000,   0.0000, 111.0000, 103.8053,   0.0000,   0.0000, 111.0000,
    #                    100.9938,   0.0000,   0.0000, 111.0000,  87.6325,   0.0000,   0.0000,
    #                    111.0000, 103.7976,   10.0000,   10.0000,   102.0000,  102.0000,   0.0000,
    #                    0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
    #                    0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
    #                    0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
    #                    0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
    #                    0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
    #                    0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
    #                    0.0000]]).cuda()
    t2 = torch.Tensor([[52., 28., 65., 71., 52., 28., 65., 68., 52., 28., 65., 68., 52., 28.,
                        65., 68.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                       [ 62.,  21., 107.,  86.,  62.,  21., 107.,  86.,  62.,  21., 107.,  86.,
                         62.,  21., 107.,  86.,  59.,  22., 104.,  88.,   0.,   0.,   0.,   0.,
                         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
                         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
                         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
                         0.,   0.,   0.,   0.]]).cuda()
    print('t.shape :',t.shape)
    print('t2.shape :',t2.shape)
    scores = torch.Tensor([[0.7495],
                          [0.7391],
                          [0.6810]])
    overlaps= Tube_Overlaps()(t2,t)
    print('overlaps :',overlaps)
