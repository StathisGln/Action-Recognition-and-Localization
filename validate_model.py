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

from net_utils import adjust_learning_rate
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, ToTensor, Resize)
from temporal_transforms import LoopPadding

from model import Model
from action_net import ACT_net

from resize_rpn import resize_rpn, resize_tube
import pdb
from bbox_transform import clip_boxes_3d

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

def bbox_overlaps_batch(tubes, gt_tubes, tubes_dur, gt_tubes_dur):
    """
    tubes: (N, 4) ndarray of float
    gt_tubes: (b, K, 5) ndarray of float

    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    batch_size = gt_tubes.size(0)

    if tubes.dim() == 3:

        N = tubes.size(1)
        K = gt_tubes.size(1)

        if tubes.size(2) == 4:
            tubes = tubes[:,:,:4].contiguous()
        else:
            tubes = tubes[:,:,1:5].contiguous()

        gt_tubes = gt_tubes[:, :, :4].contiguous()

        gt_tubes_x = (gt_tubes[:, :, 2] - gt_tubes[:, :, 0] + 1)
        gt_tubes_y = (gt_tubes[:, :, 3] - gt_tubes[:, :, 1] + 1)
        gt_tubes_t = (gt_tubes_dur[:,1] - gt_tubes_dur[:,0] + 1)

        gt_tubes_area_xy = (gt_tubes_x * gt_tubes_y).view(batch_size, 1, K)

        tubes_boxes_x = (tubes[:, :, 2] - tubes[:, :, 0] + 1)
        tubes_boxes_y = (tubes[:, :, 3] - tubes[:, :, 1] + 1)
        tubes_boxes_t = (tubes_dur[:,1] - tubes_dur[:,0] + 1)

        tubes_area_xy = (tubes_boxes_x * tubes_boxes_y).view(batch_size,N,1)

        gt_area_zero = (gt_tubes_x == 1) & (gt_tubes_y == 1)
        tubes_area_zero = (tubes_boxes_x == 1) & (tubes_boxes_y == 1)

        boxes = tubes.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)
        query_boxes = gt_tubes.view(
            batch_size, 1, K, 4).expand(batch_size, N, K, 4)

        iw = (torch.min(boxes[:, :, :, 2], query_boxes[:, :, :, 2]) -
              torch.max(boxes[:, :, :, 0], query_boxes[:, :, :, 0]) + 1)
        iw[iw < 0] = 0

        ih = (torch.min(boxes[:, :, :, 3], query_boxes[:, :, :, 3]) -
              torch.max(boxes[:, :, :, 1], query_boxes[:, :, :, 1]) + 1)
        ih[ih < 0] = 0

        ### todo time overlap
        time = tubes_dur.view(N, 1,2).expand(N,K,2)
        gt_time = gt_tubes_dur.view(1,K,2).expand(N,K,2)

        it = (torch.min(gt_time[:,:,1], time[:,:,1]) -
              torch.max(gt_time[:,:,0], time[:,:,0]) + 1)
        it[it < 0] = 0

        ua_xy = tubes_area_xy + gt_tubes_area_xy - (iw * ih )
        overlaps_xy = iw * ih / ua_xy

        ua_t = tubes_boxes_t.view(N,1) + gt_tubes_t.view(1,K) - it
              
        overlaps_t = it / ua_t

        overlaps = overlaps_xy

        # mask the overlap here.
        overlaps.masked_fill_(gt_area_zero.view(
            batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps.masked_fill_(tubes_area_zero.view(
            batch_size, N, 1).expand(batch_size, N, K), -1)
    else:
        raise ValueError('tubes input dimension is not correct.')

    return overlaps, overlaps_xy, overlaps_t



def validation(epoch, device, model, dataset_folder, sample_duration, spatial_transform, temporal_transform, boxes_file, splt_txt_path, cls2idx, vid2idx,  batch_size, n_threads):

    iou_thresh = 0.5 # Intersection Over Union thresh
    # iou_thresh = 0.5 # Intersection Over Union thresh


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

    tubes_sum = 0
    for step, data  in enumerate(val_loader):

        # if step == 10:
        #     break
        # if step == 1:
        #     break
        print('step :',step)

        vid_id, clips, boxes, n_frames, n_actions, h, w, target = data
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
        
        pre_tubes, tubes, prob_out =  model(1, dataset_folder, \
                                            vid_names, clips, vid_id,  \
                                            boxes, \
                                            mode, cls2idx, n_actions,n_frames, h, w)
        print('pre_tubes.shape :',pre_tubes.shape)
        print('tubes :', tubes.shape)
        print('prob_out :',prob_out.shape)
        n_tubes = len(tubes)

        ## get predicted labels
        _, preds = torch.max(prob_out,1)


        ### get duration
        tubes_dur = torch.zeros((tubes.size(0),2)).type_as(tubes)

        for i in range(tubes.size(0)): # for every prospect tube
            k = tubes[i].nonzero()
            if k.nelement() != 0:
                tubes_dur[i,0] = k[0,0]
                tubes_dur[i,1] = k[-1,0] 

        overlaps, overlaps_xy, overlaps_t = bbox_overlaps_batch(tubes.permute(1,0,2), boxes_.permute(1,0,2).type_as(tubes),tubes_dur, gt_tubes_dur) 
        ## for the whole tube
        _, tube_class = torch.max(overlaps,2)
        tube_class = torch.mode(tube_class.permute(1,0))[0]

        overlaps = overlaps.permute(1,2,0)

        for i in range(n_actions):

            overlaps_pos = tube_class.eq(i).nonzero().view(-1)
            if overlaps_pos.nelement() == 0 :
                false_neg += 1
                continue

            overlaps_ = overlaps[overlaps_pos,i]
            overlaps_t_ = overlaps_t[overlaps_pos,i]

            ## overall overlap
            non_zero_frames = boxes_[i].nonzero()
            non_zero_frames = torch.unique(non_zero_frames[:,0]) # get non empty lines

            overlaps_ = overlaps_.permute(1,0)[ non_zero_frames]
            overlaps_ = torch.where(overlaps_ > 0, overlaps_, torch.zeros_like(overlaps_).type_as(overlaps_))
            overlaps_ = overlaps_.permute(1,0)

            overlaps_ = overlaps_.mean(1)
            overlaps_ = torch.where(overlaps_ > iou_thresh, overlaps_, torch.zeros_like(overlaps_).type_as(overlaps_))
            # overlaps_ = torch.where(overlaps_ > 0.05, overlaps_, torch.zeros_like(overlaps_).type_as(overlaps_))
            positive_overlaps = overlaps_.nonzero()
            if positive_overlaps.nelement() != 0 : # found something
                correct_preds += preds[positive_overlaps.view(-1)].eq(target.item()).nonzero().nelement()
                if  preds[positive_overlaps.view(-1)].eq(target.item()).nonzero().nelement() != 0:
                    true_pos += 1
                else:
                    false_pos += 1
                n_preds += positive_overlaps.nelement()
            else:
                false_neg += 1
            ## time overlaps
            overlaps_t_ = torch.where(overlaps_t_ > iou_thresh, overlaps_t_, torch.zeros_like(overlaps_t_).type_as(overlaps_t_))
            positive_overlaps_t = overlaps_t_.nonzero()
            if positive_overlaps_t.nelement() != 0 : # found something
                true_pos_t += 1

            else:
                false_neg_t += 1
                
                
            # true_pos_t += torch.unique(overlaps_t_.nonzero()).nelement()
            # false_neg_t += torch.unique(overlaps_t_.eq(0).nonzero()).nelement()

    if true_pos != 0 and false_pos != 0:
        precision = float(true_pos)    / (float(true_pos)    + float(false_pos))
    else:
        precision = 0.0
    if true_pos != 0 and false_neg != 0:
        recall = float(true_pos)    / (float(true_pos)    + float(false_neg))
    else:
        recall = 0.0
    if true_pos_t != 0 and false_neg_t != 0:
        recall_t = float(true_pos_t)  / (float(true_pos_t)  + float(false_neg_t))
    else:
        recall_t = 0
        
    print(' -----------------------')
    print('| Validation Epoch: {: >3} | '.format(epoch+1))
    print('|                       |')
    print('| Proposed Action Tubes |')
    print('|                       |')
    print('| In {: >6} steps    :  |\n| True_pos   --> {: >6} |\n| False_neg  --> {: >6} | \n| False_pos  --> {: >6} |'.format(
        step, true_pos, false_neg, false_pos))
    print('|                       |')
    print('| Recall     --> {: >6.4f} |'.format(recall), \
        ' \n| Precision  --> {: >6.4f} |'.format(precision))
    print('|                       |')
    print('| In time area          |')
    print('|                       |')
    print('| In {: >6} steps    :  |\n| True_pos   --> {: >6} |\n| False_neg  --> {: >6} | \n| Recall     --> {: >6.4f} |'.format(
        step, true_pos_t, false_neg_t, recall_t))
    print('|                       |')
    print('| Classification        |')
    print('|                       |')
    print('| In {: >6} steps    :  |'.format(step))
    print('|                       |')
    print('| Correct preds :       |\n| {: >6} / {: >6}       |'.format( correct_preds, n_preds))


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
    mean = [103.29825354, 104.63845484,  90.79830328]  # jhmdb from .png
    # mean = [112.07945832, 112.87372333, 106.90993363]  # ucf-101 24 classes
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

    ##########################################
    #          Model Initialization          #
    ##########################################

    action_model_path = './action_net_model.pwf'
    reg_path = './reg_layer.pwf'
    rnn_path = './act_rnn.pwf'

    model = Model(actions, sample_duration, sample_size)
    model.load_part_model(action_model_path, reg_path, rnn_path, )

    if torch.cuda.device_count() > 1:
        print('Using {} GPUs!'.format(torch.cuda.device_count()))

        model.act_net = nn.DataParallel(model.act_net)
        
    model.to(device)
    model.act_net.to(device)
    model.eval()
    torch.no_grad()
    validation(0, device, model, dataset_folder, sample_duration, spatial_transform, temporal_transform, boxes_file, split_txt_path, cls2idx, vid2idx, batch_size, n_threads)
