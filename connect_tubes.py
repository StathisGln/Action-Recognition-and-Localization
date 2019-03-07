import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from create_tubes_from_boxes import create_tube_from_tubes

def bbox_overlaps_batch_3d(tubes, tubes_curr):
    """
    tubes: (N, 6) ndarray of float
    tubes_curr: (b, K, 7) ndarray of float

    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    batch_size = tubes_curr.size(0)
    

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
    sb_t = tubes_boxes_t.view(batch_size,N,1) + tubes_curr_t.view(batch_size,1,K) - 2 * it
    overlaps_iou_t = it / ua_t
    overlaps_sub_t = sb_t / ua_t
    # print('overlaps_iou_t.shape :',overlaps_iou_t.shape)
    # print('overlaps_sub_t.shape :',overlaps_sub_t.shape)
    overlaps_pre_t = torch.stack((overlaps_iou_t,overlaps_sub_t))
    overlaps_t = torch.mean(overlaps_pre_t,0)
    # print('overlaps_pre_t.shape :',overlaps_pre_t.shape)
    # print('overlaps_t :',overlaps_t.shape)
    # print('overlaps_xy.shape :',overlaps_xy.shape)
    overlaps = overlaps_xy * (overlaps_t * 2) # 2.5 because time_dim reduces / 3

    overlaps.masked_fill_(gt_area_zero.view(
        batch_size, 1, K).expand(batch_size, N, K), 0)
    overlaps.masked_fill_(tubes_area_zero.view(
        batch_size, N, 1).expand(batch_size, N, K), -1)

    return overlaps

def tubes_overlaps(tubes, gt_tube):
    """
    tubes: (N, 6) ndarray of float
    gt_tube: (b, K, 7) ndarray of float

    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """

    batch_size = gt_tube.size(0)
    N = tubes.size(0)
    K = gt_tube.size(1)

    gt_tube = gt_tube[:,:,:6]
    tubes = tubes[:,1:7]
    tubes = tubes.view(1,N,6).expand(batch_size, N, 6).contiguous()

    gt_tube_x = (gt_tube[:, :, 3] - gt_tube[:, :, 0] + 1)
    gt_tube_y = (gt_tube[:, :, 4] - gt_tube[:, :, 1] + 1)
    gt_tube_t  = (gt_tube[:, :, 5] - gt_tube[:, :, 2] + 1)

    gt_tube_area = (gt_tube_x * gt_tube_y * gt_tube_t ).view(batch_size, 1, K)
    # gt_tube_area_xy = (gt_tube_x * gt_tube_y  ).view(batch_size, 1, K)

    tubes_boxes_x = (tubes[:, :, 3] - tubes[:, :, 0] + 1)
    tubes_boxes_y = (tubes[:, :, 4] - tubes[:, :, 1] + 1)
    tubes_boxes_t = (tubes[:, :, 5] - tubes[:, :, 2] + 1)

    tubes_area = (tubes_boxes_x * tubes_boxes_y * tubes_boxes_t).view(batch_size, N, 1)  # for 1 frame
    # tubes_area_xy = (tubes_boxes_x * tubes_boxes_y ).view(batch_size, N, 1)  # for 1 frame

    gt_area_zero = (gt_tube_x == 1) & (gt_tube_y == 1) & (gt_tube_t == 1)
    tubes_area_zero = (tubes_boxes_x == 1) & (tubes_boxes_y == 1) & (tubes_boxes_t == 1)

    boxes = tubes.view(batch_size, N, 1, 6)
    boxes = boxes.expand(batch_size, N, K, 6)

    query_boxes = gt_tube.view(batch_size, 1, K, 6)
    query_boxes = query_boxes.expand(batch_size, N, K, 6)

    iw = (torch.min(boxes[:, :, :, 3], query_boxes[:, :, :, 3]) -
          torch.max(boxes[:, :, :, 0], query_boxes[:, :, :, 0]) + 1)
    iw[iw < 0] = 0

    ih = (torch.min(boxes[:, :, :, 4], query_boxes[:, :, :, 4]) -
          torch.max(boxes[:, :, :, 1], query_boxes[:, :, :, 1]) + 1)
    ih[ih < 0] = 0

    it = (torch.min(boxes[:, :, :, 5], query_boxes[:, :, :, 5]) -
          torch.max(boxes[:, :, :, 2], query_boxes[:, :, :, 2]) + 1)
    it[it < 0] = 0

    ua = tubes_area + gt_tube_area - (iw * ih * it )
    overlaps = iw * ih * it / ua

    # ua_xy = tubes_area_xy + gt_tube_area_xy - (iw * ih  )
    # overlaps_xy = iw * ih  / ua_xy

    # # # print('overlaps_xy :',overlaps_xy)
    # ua_t = tubes_boxes_t.view(batch_size,N,1) + gt_tube_t.view(batch_size,1,K) - it
    # overlaps_t = it / ua_t 
    # # print('overlaps_t :',overlaps_t)
    # overlaps = overlaps_xy * overlaps_t 

    overlaps.masked_fill_(gt_area_zero.view(
        batch_size, 1, K).expand(batch_size, N, K), 0)
    overlaps.masked_fill_(tubes_area_zero.view(
        batch_size, N, 1).expand(batch_size, N, K), -1)

    return overlaps

def tubes_overlaps_batch(tubes, gt_tube):
    """
    tubes: (N, 6) ndarray of float
    gt_tube: (b, K, 7) ndarray of float

    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """

    batch_size = gt_tube.size(0)
    N = tubes.size(0)
    K = gt_tube.size(1)
    # print('N {} K {} tubes.size() {}, gt_tube.size() {}, batch_size {}'.format(N,K,tubes.size(),gt_tube.shape, batch_size))
    gt_tube = gt_tube[:,:,:6]
    tubes = tubes.view(1,N,6).expand(batch_size, N, 6).contiguous()

    gt_tube_x = (gt_tube[:, :, 3] - gt_tube[:, :, 0] + 1)
    gt_tube_y = (gt_tube[:, :, 4] - gt_tube[:, :, 1] + 1)
    gt_tube_t  = (gt_tube[:, :, 5] - gt_tube[:, :, 2] + 1)

    gt_tube_area = (gt_tube_x * gt_tube_y * gt_tube_t ).view(batch_size, 1, K)
    # gt_tube_area_xy = (gt_tube_x * gt_tube_y  ).view(batch_size, 1, K)

    tubes_boxes_x = (tubes[:, :, 3] - tubes[:, :, 0] + 1)
    tubes_boxes_y = (tubes[:, :, 4] - tubes[:, :, 1] + 1)
    tubes_boxes_t = (tubes[:, :, 5] - tubes[:, :, 2] + 1)

    tubes_area = (tubes_boxes_x * tubes_boxes_y * tubes_boxes_t).view(batch_size, N, 1)  # for 1 frame
    # tubes_area_xy = (tubes_boxes_x * tubes_boxes_y ).view(batch_size, N, 1)  # for 1 frame

    gt_area_zero = (gt_tube_x == 1) & (gt_tube_y == 1) & (gt_tube_t == 1)
    tubes_area_zero = (tubes_boxes_x == 1) & (tubes_boxes_y == 1) & (tubes_boxes_t == 1)

    boxes = tubes.view(batch_size, N, 1, 6)
    boxes = boxes.expand(batch_size, N, K, 6)

    query_boxes = gt_tube.view(batch_size, 1, K, 6)
    query_boxes = query_boxes.expand(batch_size, N, K, 6)

    iw = (torch.min(boxes[:, :, :, 3], query_boxes[:, :, :, 3]) -
          torch.max(boxes[:, :, :, 0], query_boxes[:, :, :, 0]) + 1)
    iw[iw < 0] = 0

    ih = (torch.min(boxes[:, :, :, 4], query_boxes[:, :, :, 4]) -
          torch.max(boxes[:, :, :, 1], query_boxes[:, :, :, 1]) + 1)
    ih[ih < 0] = 0

    it = (torch.min(boxes[:, :, :, 5], query_boxes[:, :, :, 5]) -
          torch.max(boxes[:, :, :, 2], query_boxes[:, :, :, 2]) + 1)
    it[it < 0] = 0

    ua = tubes_area + gt_tube_area - (iw * ih * it )
    overlaps = iw * ih * it / ua

    # ua_xy = tubes_area_xy + gt_tube_area_xy - (iw * ih  )
    # overlaps_xy = iw * ih  / ua_xy

    # # # print('overlaps_xy :',overlaps_xy)
    # ua_t = tubes_boxes_t.view(batch_size,N,1) + gt_tube_t.view(batch_size,1,K) - it
    # overlaps_t = it / ua_t 
    # # print('overlaps_t :',overlaps_t)
    # overlaps = overlaps_xy * overlaps_t 

    overlaps.masked_fill_(gt_area_zero.view(
        batch_size, 1, K).expand(batch_size, N, K), 0)
    overlaps.masked_fill_(tubes_area_zero.view(
        batch_size, N, 1).expand(batch_size, N, K), -1)

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
    iou_thresh = 0.35
    start = 0
    # print('tubes_curr.shape :',tubes_curr.shape)
    
    if len(tubes) == 0: # first batch_size
        start = 1
        tubes = [[i] for i in zip([0] * tubes_curr.size(1), range(0,tubes_curr.size(1)))]

    # print('p_tubes[0] :',p_tubes[0])
    # print('tubes :',tubes)
    for tb in range(start, tubes_curr.size(0)):
    # for tb in range(start, 2):

        # if (index == 8 and tb==2):
        #     break

        ## firstly get last tubes
        tubes_last = np.array([i[-1] for i in tubes])
        # print('tubes_last :',tubes_last)
        # print('len(tubes_last) :', tubes_last.shape)
        last_tubes = torch.zeros(tubes_last.shape[0],8).type_as(p_tubes)
        for j in range(tubes_last.shape[0]):
            last_tubes[j] = p_tubes[tubes_last[j,0],tubes_last[j,1],:] # the last tubes
        # print('last_tubes :',last_tubes)

        new_tubes = tubes_curr[tb]
        # print('new_tubes :',new_tubes)
        overlaps = bbox_overlaps_batch_3d(last_tubes, new_tubes.unsqueeze(0))

        max_overlaps, max_index = torch.max(overlaps,2) # max_overlaps contains the most likely new tubes to be connected
        # print('max_overlaps :',max_overlaps)
        max_overlaps_ = torch.where(max_overlaps > iou_thresh, max_overlaps, torch.zeros_like(max_overlaps).type_as(max_overlaps))
        connect_indices = max_overlaps_.nonzero() ## [:,1] # connect_indices says which pre tubes to connect

        max_index_np = max_index[0].cpu().numpy()
        poss_indices = np.arange(0,new_tubes.size(0))
        rest_indice = np.where(np.isin(poss_indices,max_index_np)==False)
        # print(rest_indice)
        # print('max_index_np  :',max_index_np )
        # print('connect_indices  :',connect_indices )
        # print('connect_indices.shape  :',connect_indices.shape )
        if (connect_indices.nelement() == 0):
            print('no connection')
            # print('new_tubes.shape :',new_tubes.shape)
            for i in range(new_tubes.size(0)):
                tubes += [[(tb+index, i)]]
        else:
            connect_indices = connect_indices[:,1]
        
        for i in connect_indices:
            tubes[i.item()] += [(tb+index, max_index[0,i.item()].item())]

        if rest_indice[0].size != 0:
            for i in rest_indice[0]:
                tubes += [[(tb+index, i)]]

    # print('tubes :',tubes)
    f_tubes = remove_tubes(tubes,index+tb-1)
    
    return f_tubes

def remove_tubes(tubes, index):
    '''
    Removes tubes that last only 16 frames --> Not sure I have to
    '''

    f_tubes = []
    # print('index :',index)
    # print('len(tubes) :',len(tubes))
    for i in tubes:
        if not(len(i) == 1 and i[0][0] < index):
            f_tubes.append(i)
    return f_tubes

def get_gt_tubes_feats_label(f_tubes, p_tubes, features, rois_label, video_tubes):

    '''
    for each aera search for the position that corresponds to the gt_tube
    '''

    gt_tb_feat = torch.zeros(features.size(0), video_tubes.size(1), features.size(2), features.size(3))
    gt_list = [[] for i in range( video_tubes.size(1))]

    for i in range(video_tubes.size(0)):
        tb = p_tubes[i]
        gt_tb = video_tubes[i]
        overlaps = tubes_overlaps(tb, gt_tb.unsqueeze(0))
        values, max_indices = torch.max(overlaps, 1)

        for j in range(video_tubes.size(1)):
            pos_ = (i,max_indices[0,j].item())
            if values[0,j] < 0.5:
                continue
            gt_list[j] += [pos_]
            gt_tb_feat[i,j] = features[i,max_indices[0,j]]
    return gt_tb_feat, gt_list

def get_tubes_feats_label(f_tubes, p_tubes, features, rois_label, video_tubes):
    '''
    f_tubes : contains the position of each tube in p_tubes
    p_tubes : contains the actual tube 
    
    TODO : remove tubes smaller than a duration
    '''
    tube_label = np.zeros((len(f_tubes)))
    # print('rois_label.shape :',rois_label.shape)
    # print('tube_label.shape :',tube_label.shape)
    # print('f_tubes :',f_tubes)
    for i in range(len(f_tubes)):
        tb = f_tubes[i]
        # get labels
        tube_label_ = np.zeros((len(tb)))
        for j in range(len(tb)):
            tube_label_[j] = rois_label[tb[j][0],tb[j][1]]
        tube_label[i] = tube_label_[0] if np.all(tube_label_ == tube_label_[0]) else 0

    ### calculate max_overlap and keep
    fg_tubes = np.where(tube_label != 0)[0]
    # print('fg_tubes :',fg_tubes)
    # print('fg_tubes.size :',fg_tubes.size)

    # if fg_tubes == 0 : # no fg_tubes
    #     return 
    proposed_seq = torch.zeros(len(fg_tubes),6).type_as(p_tubes)
    for i in range(fg_tubes.shape[0]):

        tube_pos = f_tubes[i]
        tmp_tube = torch.zeros((len(tube_pos),6))
        for j in range(len(tube_pos)):
            tmp_tube[j] = p_tubes[tube_pos[j][0]][tube_pos[j][1]][1:7]
        proposed_seq[i] = create_tube_from_tubes(tmp_tube)
        # if i < 10 :
        #     print(tube_pos)
        #     print(tmp_tube)
        #     print('proposed_seq[i] :',proposed_seq[i])

    # print('proposed_seq :',proposed_seq.shape)
    # print('video_tubes :',video_tubes.shape)
    if proposed_seq.size(0) < 1:
        return []
    overlaps = tubes_overlaps_batch(proposed_seq, video_tubes)
    max_overlaps,gt_assignment = torch.max(overlaps,2)
    max_overlaps = max_overlaps.squeeze(0)
    bg_inds = torch.nonzero((max_overlaps < 0.5) &
                            (max_overlaps >= 0.1)).view(-1)
    # print('bg_inds :',bg_inds)
    bg_num_rois = bg_inds.numel()

    if bg_num_rois > 0 :
        bg_rois_picked = min(16,2 * video_tubes.size(1))
        rand_num = np.floor(np.random.rand(bg_rois_picked) * bg_num_rois)
        rand_num = torch.from_numpy(rand_num).type_as(video_tubes).long()
        bg_inds = bg_inds[rand_num]

    # print('bg_num_rois :',bg_num_rois)
    # print('bg_rois_picked :',bg_rois_picked)
    # print('bg_inds :',bg_inds)

    ret = [f_tubes[int(i)] for i in bg_inds]

    return ret

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

    

