import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lib.dataloaders.ucf_dataset import Video_Dataset_small_clip

from lib.utils.spatial_transforms import (
    Compose, Normalize, Scale, ToTensor)
from lib.utils.temporal_transforms import LoopPadding
from lib.models.action_net import ACT_net

np.random.seed(42)

def bbox_transform_inv_3d(boxes, deltas, batch_size):


    if boxes.size(-1) == 7:
        boxes = boxes[:,:, 1:]
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


def bbox_overlaps_batch_3d(tubes, gt_tubes):
    """
    tubes: (N, 6) ndarray of float
    gt_tubes: (b, K, 5) ndarray of float

    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    batch_size = gt_tubes.size(0)

    if tubes.dim() == 2:

        N = tubes.size(0)
        K = gt_tubes.size(1)

        if tubes.size(-1) >= 7:
            tubes = tubes[:,1:7]
        if gt_tubes.size(-1) == 7:
            gt_tubes = gt_tubes[:,:,:6]

        tubes = tubes.view(1, N, 6)
        tubes = tubes.expand(batch_size, N, 6).contiguous()
        gt_tubes = gt_tubes[:, :, :6].contiguous()

        gt_tubes_x = (gt_tubes[:, :, 3] - gt_tubes[:, :, 0] + 1)
        gt_tubes_y = (gt_tubes[:, :, 4] - gt_tubes[:, :, 1] + 1)
        gt_tubes_t = (gt_tubes[:, :, 5] - gt_tubes[:, :, 2] + 1)

        if batch_size == 1:  # only 1 video in batch:
            gt_tubes_x = gt_tubes_x.unsqueeze(0)
            gt_tubes_y = gt_tubes_y.unsqueeze(0)
            gt_tubes_t = gt_tubes_t.unsqueeze(0)

        gt_tubes_area = (gt_tubes_x * gt_tubes_y * gt_tubes_t)
        gt_tubes_area_xy = gt_tubes_x * gt_tubes_y

        tubes_boxes_x = (tubes[:, :, 3] - tubes[:, :, 0] + 1)
        tubes_boxes_y = (tubes[:, :, 4] - tubes[:, :, 1] + 1)
        tubes_boxes_t = (tubes[:, :, 5] - tubes[:, :, 2] + 1)

        tubes_area = (tubes_boxes_x * tubes_boxes_y *
                        tubes_boxes_t).view(batch_size, N, 1)  # for 1 frame
        tubes_area_xy = (tubes_boxes_x * tubes_boxes_y).view(batch_size, N, 1)  # for 1 frame
        
        gt_area_zero = (gt_tubes_x == 1) & (gt_tubes_y == 1) & (gt_tubes_t == 1)
        tubes_area_zero = (tubes_boxes_x == 1) & (tubes_boxes_y == 1) & (tubes_boxes_t == 1)

        gt_area_zero_xy = (gt_tubes_x == 1) & (gt_tubes_y == 1) 
        tubes_area_zero_xy = (tubes_boxes_x == 1) & (tubes_boxes_y == 1) 

        gt_area_zero_t =  (gt_tubes_t == 1)
        tubes_area_zero_t =  (tubes_boxes_t == 1)

        boxes = tubes.view(batch_size, N, 1, 6)
        boxes = boxes.expand(batch_size, N, K, 6)
        query_boxes = gt_tubes.view(batch_size, 1, K, 6)
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

        ua = tubes_area + gt_tubes_area - (iw * ih * it)
        ua_xy = tubes_area_xy + gt_tubes_area_xy - (iw * ih )
        ua_t = tubes_boxes_t.unsqueeze(2) + gt_tubes_t - it

        overlaps = iw * ih * it / ua
        overlaps_xy = iw * ih  / ua_xy
        overlaps_t = it / ua_t

        overlaps.masked_fill_(gt_area_zero.view(
            batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps.masked_fill_(tubes_area_zero.view(
            batch_size, N, 1).expand(batch_size, N, K), -1)

        overlaps_xy.masked_fill_(gt_area_zero.view(
            batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps_xy.masked_fill_(tubes_area_zero.view(
            batch_size, N, 1).expand(batch_size, N, K), -1)

        overlaps_t.masked_fill_(gt_area_zero.view(
            batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps_t.masked_fill_(tubes_area_zero.view(
            batch_size, N, 1).expand(batch_size, N, K), -1)

    else:
        raise ValueError('tubes input dimension is not correct.')

    return overlaps, overlaps_xy, overlaps_t

def validation(epoch, device, model, dataset_folder, sample_duration, spatial_transform, temporal_transform, boxes_file, splt_txt_path, cls2idx, batch_size, n_threads):

    iou_thresh = 0.5 # Intersection Over Union thresh
    data = Video_Dataset_small_clip(dataset_folder, frames_dur=sample_duration, spatial_transform=spatial_transform,
                                    temporal_transform=temporal_transform, bboxes_file= boxes_file,
                                    split_txt_path=splt_txt_path, mode='test', classes_idx=cls2idx)
    data_loader = torch.utils.data.DataLoader(data, batch_size=2,
                                              shuffle=True, num_workers=0, pin_memory=True)
    model.eval()

    true_pos = 0
    false_neg = 0

    true_pos_xy = 0
    false_neg_xy = 0

    true_pos_t = 0
    false_neg_t = 0

    sgl_true_pos = 0
    sgl_false_neg = 0

    ## 2 rois : 1450
    tubes_sum = 0
    for step, data  in enumerate(data_loader):

        # if step == 2:
        #     break
        print('step :',step)

        clips, h, w, gt_tubes_r, gt_rois, n_actions, n_frames, im_info = data
        clips_ = clips.to(device)
        gt_tubes_r_ = gt_tubes_r.to(device)
        gt_rois_ = gt_rois.to(device)
        n_actions_ = n_actions.to(device)
        im_info_ = im_info.to(device)
        start_fr = torch.zeros(clips_.size(0)).to(device)
        # for i in range(2):
        #     print('gt_rois :',gt_rois[i,:n_actions[i]])
        tubes, bbox_pred, _,_,_,_,_,_,_,_,sgl_rois_bbox_pred,_   = model(clips,
                                                                       im_info,
                                                                       None, None,
                                                                       None)
        n_tubes = len(tubes)
        # init tensor for final frames

        for i in range(tubes.size(0)): # how many frames we have
            # calculate single frame overlaps
            tubes_t = tubes[i]
            gt_tub  = gt_tubes_r[i]

            non_empty   = gt_tub.sum(1).nonzero()
            if non_empty.nelement() == 0:
                continue
            non_empty = non_empty.view(-1)
            gt_tub = gt_tub[non_empty]

            overlaps, overlaps_xy, overlaps_t = bbox_overlaps_batch_3d(tubes_t, gt_tub.unsqueeze(0).type_as(tubes_t)) # check one video each time

            ## for the whole tube
            gt_max_overlaps, _ = torch.max(overlaps, 1)
            gt_max_overlaps = torch.where(gt_max_overlaps > iou_thresh, gt_max_overlaps, torch.zeros_like(gt_max_overlaps).type_as(gt_max_overlaps))

            detected =  gt_max_overlaps.ne(0).sum()
            n_elements = gt_max_overlaps.nelement()
            true_pos += detected
            false_neg += n_elements - detected

            # ## for xy - area
            # gt_max_overlaps_xy, _ = torch.max(overlaps_xy, 1)
            # gt_max_overlaps_xy = torch.where(gt_max_overlaps_xy > iou_thresh, gt_max_overlaps_xy, torch.zeros_like(gt_max_overlaps_xy).type_as(gt_max_overlaps_xy))

            # detected_xy =  gt_max_overlaps_xy.ne(0).sum()
            # n_elements_xy = gt_max_overlaps_xy.nelement()
            # true_pos_xy += detected_xy
            # false_neg_xy += n_elements_xy - detected_xy

            # ## for t - area
            # gt_max_overlaps_t, _ = torch.max(overlaps_t, 1)
            # gt_max_overlaps_t = torch.where(gt_max_overlaps_t > iou_thresh, gt_max_overlaps_t, torch.zeros_like(gt_max_overlaps_t).type_as(gt_max_overlaps_t))
            # detected_t =  gt_max_overlaps_t.ne(0).sum()
            # n_elements_t = gt_max_overlaps_t.nelement()
            # true_pos_t += detected_t
            # false_neg_t += n_elements_t - detected_t

            tubes_sum += 1


    recall     = float(true_pos)      / (float(true_pos)      + float(false_neg))
    # recall_xy  = float(true_pos_xy)   / (float(true_pos_xy)   + float(false_neg_xy))
    # recall_t   = float(true_pos_t)    / (float(true_pos_t)    + float(false_neg_t))
    # sgl_recall = float(sgl_true_pos)  / (float(sgl_true_pos)  + float(sgl_false_neg))

    print(' -----------------------')
    print('| Validation Epoch: {: >3} | '.format(epoch+1))
    print('|                       |')
    print('| Proposed Action Tubes |')
    print('|                       |')
    print('| In {: >6} steps    :  |\n| True_pos   --> {: >6} |\n| False_neg  --> {: >6} | \n| Recall     --> {: >6.4f} |'.format(
        step, true_pos, false_neg, recall))
    # print('|                       |')
    # print('| In xy area            |')
    # print('|                       |')
    # print('| In {: >6} steps    :  |\n| True_pos   --> {: >6} |\n| False_neg  --> {: >6} | \n| Recall     --> {: >6.4f} |'.format(
    #     step, true_pos_xy, false_neg_xy, recall_xy))
    # print('|                       |')
    # print('| In time area          |')
    # print('|                       |')
    # print('| In {: >6} steps    :  |\n| True_pos   --> {: >6} |\n| False_neg  --> {: >6} | \n| Recall     --> {: >6.4f} |'.format(
    #     step, true_pos_t, false_neg_t, recall_t))
    # print('|                       |')
    # print('| Single frame          |')
    # print('|                       |')
    # print('| In {: >6} steps    :  |'.format(step))
    # print('|                       |')
    # print('| True_pos   --> {: >6} |\n| False_neg  --> {: >6} | \n| Recall     --> {: >6.4f} |'.format(
        # sgl_true_pos, sgl_false_neg, sgl_recall))


    print(' -----------------------')
        
if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    dataset_folder = '/gpu-data2/sgal/UCF-101-frames'
    boxes_file = '/gpu-data2/sgal/pyannot.pkl'
    split_txt_path = '/gpu-data2/sgal/UCF101_Action_detection_splits/'

    sample_size = 112
    sample_duration = 16  # len(images)

    batch_size = 1
    # batch_size = 1
    n_threads = 0

    # # get mean
    # mean =  [103.75581543 104.79421473  91.16894564] # jhmdb
    # mean = [103.29825354, 104.63845484,  90.79830328]  # jhmdb from .png
    mean = [112.07945832, 112.87372333, 106.90993363]  # ucf-101 24 classes
    # generate model

    actions = ['__background__', 'Basketball','BasketballDunk','Biking','CliffDiving','CricketBowling',
               'Diving','Fencing','FloorGymnastics','GolfSwing','HorseRiding','IceDancing',
               'LongJump','PoleVault','RopeClimbing','SalsaSpin','SkateBoarding','Skiing',
               'Skijet','SoccerJuggling','Surfing','TennisSwing','TrampolineJumping',
               'VolleyballSpiking','WalkingWithDog']


    cls2idx = {actions[i]: i for i in range(0, len(actions))}

    spatial_transform = Compose([Scale(sample_size),  # [Resize(sample_size),
                                 ToTensor(),
                                 Normalize(mean, [1, 1, 1])])
    temporal_transform = LoopPadding(sample_duration)

    # Init action_net
    model = ACT_net(actions, sample_duration)
    model.create_architecture()
    model = nn.DataParallel(model)
    model.to(device)

    model_data = torch.load('./action_net_model_part1_1.pwf')
    reg_layer_data = torch.load('./reg_layer.pwf')

    model.load_state_dict(model_data)
    model.module.reg_layer.load_state_dict(reg_layer_data)
    model.eval()

    validation(0, device, model, dataset_folder, sample_duration, spatial_transform, temporal_transform, boxes_file, split_txt_path, cls2idx, batch_size, n_threads)
