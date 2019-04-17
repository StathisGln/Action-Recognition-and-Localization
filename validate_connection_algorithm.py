import os
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from resnet_3D import resnet34
from ucf_dataset import Video_UCF, video_names

from create_video_id import get_vid_dict
from create_tubes_from_boxes import create_video_tube

from net_utils import adjust_learning_rate, from_tubes_to_rois
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, ToTensor, Resize)
from temporal_transforms import LoopPadding

from action_net import ACT_net
from model import Model

from resize_rpn import resize_rpn, resize_tube
import pdb

from bbox_transform import clip_boxes_3d, bbox_transform_inv, clip_boxes, bbox_overlaps_batch, bbox_overlaps

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
        if tubes.size(-1) == 7:
            tubes = tubes[:,1:]
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

def validation(epoch, device, model, dataset_folder, sample_duration, spatial_transform, temporal_transform, boxes_file, splt_txt_path, cls2idx, vid2idx,  batch_size, sample_size, n_threads):

    iou_thresh = 0.5 # Intersection Over Union thresh
    
    val_name_loader = video_names(dataset_folder, splt_txt_path, boxes_file, vid2idx, mode='test')
    val_loader = torch.utils.data.DataLoader(val_name_loader, batch_size=batch_size,
                                             shuffle=True)
    model.eval()

    true_pos = 0    # there is a tube that is has > 0.5 overlap and correct label
    false_pos = 0   # there is a tube that has > 0.5 overlap but there is no correct label
    false_neg = 0   # there is no tube that has > 0.5 overlap

    true_pos_t = 0
    false_pos_t = 0
    false_neg_t = 0

    correct_preds = 0
    n_preds = 0
    preds = 0

    ## 2 rois : 1450
    tubes_sum = 0
    for step, data  in enumerate(val_loader):

        if step == 2:
            break
        print('step :',step)

        vid_id, clips, boxes, n_frames, n_actions, h, w  = data
        mode = 'test'

        vid_id = vid_id.to(device)
        n_frames = n_frames.to(device)
        n_actions = n_actions.to(device)
        h = h.to(device)
        w = w.to(device)
        ## create video tube
        boxes_ = boxes[0,:n_actions, :n_frames]

        gt_tubes_dur = torch.zeros((boxes_.size(0),2)).float()
        for i in range(boxes_.size(0)): # for every gt tube
            k = boxes_[i].nonzero()
            if k.nelement() != 0:
                gt_tubes_dur[i,0] = k[0,0]
                gt_tubes_dur[i,1] = k[-1,0]
        
        pre_tubes, _, tubes, _, poss, sgl_frame_pred =  model(1, dataset_folder, \
                                                              vid_names, clips, vid_id,  \
                                                              None, \
                                                              mode, cls2idx, None,n_frames, h, w)
        n_tubes = tubes.size(0)
        found = torch.zeros(n_actions)

        # for i in range(1): # how many tubes we have
        for i in range(tubes.size(0)): # how many frames we have
            # calculate single frame overlaps
            rois_f = torch.zeros(n_frames,4).type_as(tubes)
            means = torch.zeros(boxes_.size(0))
            for j in range(tubes.size(1)):
                p  = poss[i,j,0].item()
                tb = tubes[i,j,[0,1,3,4]]

                if tb[0] == -1:
                    break
                roi_pred = sgl_frame_pred[i,j]
                r_tb = tb.unsqueeze(0).expand(sample_duration, tb.size(0))

                r_tb = bbox_transform_inv(r_tb.unsqueeze(0), \
                                          roi_pred.unsqueeze(0),1)
                r_tb = clip_boxes(r_tb, torch.Tensor([sample_size, sample_size]).unsqueeze(0).expand(r_tb.size(1),2), \
                                  1).squeeze()
                inds = torch.arange(tubes[i,j,2].long(), (tubes[i,j,5]+1).long())
                inds_ = inds - ( p * int(sample_duration/2))

                rois_f[inds]  = r_tb[inds_]
            inds = torch.arange(n_frames.item())
            
            rois_overlaps = bbox_overlaps_batch(rois_f, boxes_[:,:,:4].float())

            for j in range(boxes_.size(0)):
                overlaps_ = rois_overlaps[j,inds, inds]
                non_empty = overlaps_.ne(-1).nonzero()

                if non_empty.nelement() == 0:
                    continue
                non_empty = non_empty.view(-1)
                overlaps_ = overlaps_[non_empty]

                means[j] = overlaps_.mean(0)

            _max_overlap, index_ = torch.max(means, 0)
            if _max_overlap > iou_thresh:
                found[index_] = 1

        detected =  found.ne(0).sum()
        true_pos += detected.item()
        false_neg += n_actions.item() - detected.item()


        tubes_sum += n_actions.item()
        print('true_pos :',true_pos)

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
    #     sgl_true_pos, sgl_false_neg, sgl_recall))


    print(' -----------------------')
        
if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    dataset_folder = '../UCF-101-frames'
    boxes_file = '../pyannot.pkl'
    split_txt_path = '../UCF101_Action_detection_splits/'

    sample_size = 112
    sample_duration = 16  # len(images)

    batch_size = 1
    n_threads = 0

    # # get mean
    mean = [112.07945832, 112.87372333, 106.90993363]  # ucf-101 24 classes

    # generate model

    actions = ['__background__', 'Basketball','BasketballDunk','Biking','CliffDiving','CricketBowling',
               'Diving','Fencing','FloorGymnastics','GolfSwing','HorseRiding','IceDancing',
               'LongJump','PoleVault','RopeClimbing','SalsaSpin','SkateBoarding','Skiing',
               'Skijet','SoccerJuggling','Surfing','TennisSwing','TrampolineJumping',
               'VolleyballSpiking','WalkingWithDog']


    cls2idx = {actions[i]: i for i in range(0, len(actions))}

    ### get videos id
    vid2idx,vid_names = get_vid_dict(dataset_folder)

    spatial_transform = Compose([Scale(sample_size),  # [Resize(sample_size),
                                 ToTensor(),
                                 Normalize(mean, [1, 1, 1])])
    temporal_transform = LoopPadding(sample_duration)

    # Init action_net
    model = Model(actions, sample_duration, sample_size)

    action_model_path = './action_net_model_dropout_08_non_normalize.pwf'
    model_data = model.load_part_model(action_model_path = action_model_path, rnn_path = None)

    model.act_net = nn.DataParallel(model.act_net)
    model         = model.to(device)
    model.act_net = model.act_net.to(device)

    model.eval()

    validation(0, device, model, dataset_folder, sample_duration, spatial_transform, temporal_transform, boxes_file, split_txt_path, cls2idx, vid2idx, batch_size, sample_size, n_threads)
