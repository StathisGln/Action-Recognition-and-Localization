import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def bbox_overlaps_batch_3d(tubes, tubes_curr):
    """
    tubes: (N, 6) ndarray of float
    tubes_curr: (b, K, 7) ndarray of float

    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    batch_size = tubes_curr.size(0)
    
    if tubes.dim() == 2:
        N = tubes.size(0)
        K = tubes_curr.size(1)

        # print('N {} K {}'.format(N,K))

        tubes = tubes[:,1:7]
        tubes_curr = tubes_curr[:,:,1:7]

        # print('tubes :',tubes)
        # print('tubes_curr :',tubes_curr)
        tubes = tubes.view(1,N,6).expand(batch_size, N, 6).contiguous()

        tubes_curr_x = (tubes_curr[:, :, 3] - tubes_curr[:, :, 0] + 1)
        tubes_curr_y = (tubes_curr[:, :, 4] - tubes_curr[:, :, 1] + 1)
        tubes_curr_t  = (tubes_curr[:, :, 5] - tubes_curr[:, :, 2] + 1)
        
        # print('tubes_curr_x.shape :',tubes_curr_x)
        # print('tubes_curr_y.shape :',tubes_curr_y)
        # print('tubes_curr_t.shape :',tubes_curr_t)

        tubes_curr_area_xy = (tubes_curr_x * tubes_curr_y ).view(batch_size, 1, K)
        # print('tubes_curr_area_xy :',tubes_curr_area_xy)

        tubes_boxes_x = (tubes[:, :, 3] - tubes[:, :, 0] + 1)
        tubes_boxes_y = (tubes[:, :, 4] - tubes[:, :, 1] + 1)
        tubes_boxes_t = (tubes[:, :, 5] - tubes[:, :, 2] + 1)

        # print('tubes_boxes_x.shape :',tubes_boxes_x)
        # print('tubes_boxes_y.shape :',tubes_boxes_y)
        # print('tubes_boxes_t.shape :',tubes_boxes_t)

        tubes_area_xy = (tubes_boxes_x * tubes_boxes_y ).view(batch_size, N, 1)  # for 1 frame
        # print('tubes_area_xy :',tubes_area_xy)
        
        gt_area_zero = (tubes_curr_x == 1) & (tubes_curr_y == 1) & (tubes_curr_t == 1)
        tubes_area_zero = (tubes_boxes_x == 1) & (tubes_boxes_y == 1) & (tubes_boxes_t == 1)

        boxes = tubes.view(batch_size, N, 1, 6)
        boxes = boxes.expand(batch_size, N, K, 6)

        # print('boxes :',boxes)

        query_boxes = tubes_curr.view(batch_size, 1, K, 6)
        query_boxes = query_boxes.expand(batch_size, N, K, 6)

        # print('query_boxes :',query_boxes )
        
        iw = (torch.min(boxes[:, :, :, 3], query_boxes[:, :, :, 3]) -
              torch.max(boxes[:, :, :, 0], query_boxes[:, :, :, 0]) + 1)
        iw[iw < 0] = 0

        ih = (torch.min(boxes[:, :, :, 4], query_boxes[:, :, :, 4]) -
              torch.max(boxes[:, :, :, 1], query_boxes[:, :, :, 1]) + 1)
        ih[ih < 0] = 0

        it = (torch.min(boxes[:, :, :, 5], query_boxes[:, :, :, 5]) -
              torch.max(boxes[:, :, :, 2], query_boxes[:, :, :, 2]) + 1)
        it[it < 0] = 0

        # print('iw :',iw)
        # print('ih :',ih)
        # print('it.shape :',it.shape)
        ua_xy = tubes_area_xy + tubes_curr_area_xy - (iw * ih )
        overlaps_xy = iw * ih / ua_xy

        # print('overlaps_xy :',overlaps_xy)
        ua_t = tubes_boxes_t.view(batch_size,N,1) + tubes_curr_t.view(batch_size,1,K) - it
        overlaps_t = it / ua_t 
        # print('overlaps_t :',overlaps_t)
        overlaps = overlaps_xy * (overlaps_t * 3) # 2.5 because time_dim reduces / 3

        overlaps.masked_fill_(gt_area_zero.view(
            batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps.masked_fill_(tubes_area_zero.view(
            batch_size, N, 1).expand(batch_size, N, K), -1)

    elif tubes.dim() == 3:
        N = tubes.size(1)
        K = tubes_curr.size(1)

        if tubes.size(2) == 6:
            tubes = tubes[:, :, :6].contiguous()
        else:
            tubes = tubes[:, :, 1:7].contiguous()


        if tubes_curr.size(2) == 6:
            tubes_curr = tubes_curr[:, :, :6].contiguous()
        else:
            tubes_curr = tubes_curr[:, :, 1:7].contiguous()
    
        tubes_curr = tubes_curr[:, :, :6].contiguous()

        tubes_curr_x = (tubes_curr[:, :, 3] - tubes_curr[:, :, 0] + 1)
        tubes_curr_y = (tubes_curr[:, :, 4] - tubes_curr[:, :, 1] + 1)
        tubes_curr_t = (tubes_curr[:, :, 5] - tubes_curr[:, :, 2] + 1)

        tubes_curr_area_xy = (tubes_curr_x * tubes_curr_y ).view(batch_size, 1, K)

        tubes_boxes_x = (tubes[:, :, 3] - tubes[:, :, 0] + 1)
        tubes_boxes_y = (tubes[:, :, 4] - tubes[:, :, 1] + 1)
        tubes_boxes_t = (tubes[:, :, 5] - tubes[:, :, 2] + 1)

        tubes_area_xy = (tubes_boxes_x *
                        tubes_boxes_y ).view(batch_size, N, 1)

        # print('tubes_area.shape :',tubes_area.shape)
        gt_area_zero = (tubes_curr_x == 1) & (tubes_curr_y == 1) & (tubes_curr_t == 1)
        tubes_area_zero = (tubes_boxes_x == 1) & (tubes_boxes_y == 1) & (tubes_boxes_t == 1)

        boxes = tubes.view(batch_size, N, 1, 6).expand(batch_size, N, K, 6)
        query_boxes = tubes_curr.view(
            batch_size, 1, K, 6).expand(batch_size, N, K, 6)

        iw = (torch.min(boxes[:, :, :, 3], query_boxes[:, :, :, 3]) -
              torch.max(boxes[:, :, :, 0], query_boxes[:, :, :, 0]) + 1)
        iw[iw < 0] = 0

        ih = (torch.min(boxes[:, :, :, 4], query_boxes[:, :, :, 4]) -
              torch.max(boxes[:, :, :, 1], query_boxes[:, :, :, 1]) + 1)
        ih[ih < 0] = 0

        it = (torch.min(boxes[:, :, :, 5], query_boxes[:, :, :, 5]) -
              torch.max(boxes[:, :, :, 2], query_boxes[:, :, :, 2]) + 1)
        it[it < 0] = 0

        ua_xy = tubes_area_xy + tubes_curr_area_xy - (iw * ih )
        overlaps_xy = iw * ih / ua_xy

        ua_t = tubes_boxes_t.view(batch_size,N,1) + tubes_curr_t.view(batch_size,1,K) - it
        overlaps_t = it / ua_t

        overlaps = overlaps_xy 

        # mask the overlap here.
        overlaps.masked_fill_(gt_area_zero.view(
            batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps.masked_fill_(tubes_area_zero.view(
            batch_size, N, 1).expand(batch_size, N, K), -1)
    else:
        raise ValueError('tubes    input dimension is not correct.')

    return overlaps

def connect_tubes(tubes, tubes_curr, p_tubes, pooled_feats, rois_label, index): # tubes are the already connected and tubes_curr are the new
    '''
    tubes : a list containing the position in p_tubes for each tube, in each sequence
    tubes_curr : proposed tubes according to last loop
    p_tubes : all the proposed tubes
    pooled_feats : 
    rois_label : each tube label
    index :
    '''
    # iou_thresh = 0.3
    iou_thresh = 0.5
    start = 0
    # print('tubes_curr.shape :',tubes_curr.shape)
    
    if len(tubes) == 0: # first batch_size
        start = 1
        tubes = [[i] for i in zip([0] * tubes_curr.size(1), range(0,tubes_curr.size(1)))]

    # print('p_tubes[0] :',p_tubes[0])
    for tb in range(start, tubes_curr.size(0)):
    # for tb in range(start, 2):

        ## firstly get last tubes
        tubes_last = np.array([i[-1] for i in tubes])
        # print('tubes_last :',tubes_last)

        last_tubes = torch.zeros(tubes_last.shape[0],8).type_as(p_tubes)
        for j in range(tubes_last.shape[0]):
            last_tubes[j] = p_tubes[tubes_last[j,0],tubes_last[j,1],:] # the last tubes

        new_tubes = tubes_curr[tb]
        overlaps = bbox_overlaps_batch_3d(last_tubes, new_tubes.unsqueeze(0))

        max_overlaps, max_index = torch.max(overlaps,2) # max_overlaps contains the most likely new tubes to be connected
        max_overlaps = torch.where(max_overlaps > iou_thresh, max_overlaps, torch.zeros_like(max_overlaps).type_as(max_overlaps))
        connect_indices = max_overlaps.nonzero() ## [:,1] # connect_indices says which pre tubes to connect

        max_index_np = max_index[0].cpu().numpy()
        poss_indices = np.arange(0,new_tubes.size(0))
        rest_indice = np.where(np.isin(poss_indices,max_index_np)==False)
        # print(rest_indice)
        # print('max_index_np  :',max_index_np )
        # print('connect_indices  :',connect_indices )
        # print('connect_indices.shape  :',connect_indices.shape )
        if (connect_indices.nelement() == 0):
            print('no connection')

        connect_indices = connect_indices[:,1]
        
        for i in connect_indices:
            # print('max_index.shape :',max_index.shape)
            # print('[tb, max_index[i.item()]] :',[tb, max_index[0,i.item()].item()])
            # print('len(tubes) :',len(tubes))
            # print('i.item() :',i.item())
            tubes[i.item()] += [(tb, max_index[0,i.item()].item())]
            
            # tubes[i] += [tubes_curr[0,max_index[0,i]].cpu().tolist()]
        if rest_indice[0].size != 0:
            for i in rest_indice[0]:
                tubes += [[(tb, i)]]

    # print('tubes :',tubes)
    # print('len(tubes) :',len(tubes))
    return tubes

if __name__ == '__main__':

    tubes_curr =torch.Tensor([[[  0., 120.,   4.,   0., 316., 236.,  15.],
                               [  0., 120.,   4.,   0., 316., 236.,  15.],
                               [  0., 120.,   4.,   0., 316., 236.,  15.],
                               [  0., 120.,   4.,   0., 316., 236.,  15.],
                               [  0., 130.,   4.,   0., 316., 236.,  15.],
                               [  0., 120.,   4.,   0., 316., 236.,  15.],
                               [  0., 120.,   4.,   0., 316., 236.,  15.],
                               [  0., 120.,   4.,   0., 316., 236.,  15.],
                               [  0., 120.,   4.,   0., 316., 236.,  15.]]]).cuda()

    # tubes_all = torch.Tensor([[[  0.0000, 120.0000,   4.0000,   0.0000, 316.0000, 236.0000,  15.0000],
    #                        [  0.0000,   0.0000,   0.0000,   0.1917, 202.0876, 208.8185,  16.2964],
    #                        [  0., 130.,   4.,   0., 316., 236.,  15.],
    #                        [  0.0000,   0.0000,   0.0000,   0.0688, 253.3957, 118.9900,   7.8497]]]).tolist()
    tubes_all = [[[ 0.0000, 120.0000,   4.0000,   0.0000, 316.0000, 236.0000,  15.0000]],
                 [[  0.0000,   0.0000,   0.0000,   0.1917, 202.0876, 208.8185,  16.2964]],
                 [[  0., 130.,   4.,   0., 316., 236.,  15.]],
                 [[  0.0000,   0.0000,   0.0000,   0.0688, 253.3957, 118.9900,   7.8497]]]


    print(connect_tubes(tubes_all, tubes_curr, 8))
    print(connect_tubes([], tubes_curr, 0))

    

