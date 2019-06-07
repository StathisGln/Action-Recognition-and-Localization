import os
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from resnet_3D import resnet34
from jhmdb_dataset import Video

from net_utils import adjust_learning_rate
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, ToTensor, Resize)
from temporal_transforms import LoopPadding
from model import Model
from resize_rpn import resize_rpn, resize_tube
import pdb

np.random.seed(42)

def validation(epoch, device, model, dataset_folder, sample_duration, spatial_transform, temporal_transform, boxes_file, splt_txt_path, cls2idx, batch_size, n_threads):

    iou_thresh = 0.5 # Intersection Over Union thresh
    iou_thresh_4 = 0.4 # Intersection Over Union thresh
    iou_thresh_3 = 0.3 # Intersection Over Union thresh

    vid_name_loader = video_names(dataset_frames, split_txt_path, boxes_file, vid2idx, mode='test')
    data_loader = torch.utils.data.DataLoader(vid_name_loader, batch_size=n_devs, num_workers=8*n_devs, pin_memory=True,
                                              shuffle=True)    # reset learning rate
    model.eval()

    true_pos = 0
    false_neg = 0

    correct_preds = torch.zeros(1).long().to(device)
    n_preds = torch.zeros(1).long().to(device)
    preds = torch.zeros(1).long().to(device)
    ## 2 rois : 1450
    tubes_sum = 0
    for step, data  in enumerate(data_loader):

        if step == 2:
            break
        print('step :',step)

        vid_id, clips, boxes, n_frames, n_actions, h, w =data

        vid_id = torch.Tensor(vid_id).int()
        clips = clips.unsqueeze(0).to(device)
        boxes = torch.from_numpy(boxes).to(device)
        n_frames = torch.from_numpy(n_frames).to(device)
        n_actions = torch.from_numpy(n_actions).int().to(device)
        im_info = torch.Tensor([h,w,clips.size(2)]).unsqueeze(0).to(device)
        mode = 'test'

        tubes,  \
        prob_out, cls_loss =  model(n_devs, dataset_frames, \
                                vid_names, clips, vid_id,  \
                                None, \
                                mode, cls2idx, None, n_frames, h, w)


        print('tubes.shape :',tubes.shape)
        print('bbox_pred.shape :',bbox_pred.shape)
        exit(-1)
        n_tubes = len(tubes)

        _, cls_int = torch.max(cls_prob,1)
        # print('cls_int :',cls_int, ' target :', target)
        for k in cls_int.cpu().tolist():
            if k == target.data:
                print('Found one')
                correct_preds += 1
            n_preds += 1
        for i in range(gt_tubes_r.size(0)): # how many frames we have
            tubes_t = torch.zeros(n_tubes, 7).type_as(gt_tubes_r)
            for j in range(n_tubes):
                # print('J :',j, 'i :',i)
                # print(' len(tube[j]) :',len(tubes[j]))
                # print('tubes[j] :',tubes[j])
                # print('tubes[j][i] :',tubes[j][i])
                
                if (len(tubes[j]) - 1 < i):
                    continue
                tubes_t[j] = torch.Tensor(tubes[j][i][:7]).type_as(tubes_t)
            
            overlaps, overlaps_xy, overlaps_t = bbox_overlaps_batch_3d(tubes_t.squeeze(0), gt_tubes_r[i].unsqueeze(0)) # check one video each time

            ## for the whole tube
            gt_max_overlaps, _ = torch.max(overlaps, 1)
            gt_max_overlaps = torch.where(gt_max_overlaps > iou_thresh, gt_max_overlaps, torch.zeros_like(gt_max_overlaps).type_as(gt_max_overlaps))
            detected =  gt_max_overlaps.ne(0).sum()
            n_elements = gt_max_overlaps.nelement()
            true_pos += detected
            false_neg += n_elements - detected

            ## for xy - area
            gt_max_overlaps_xy, _ = torch.max(overlaps_xy, 1)
            gt_max_overlaps_xy = torch.where(gt_max_overlaps_xy > iou_thresh, gt_max_overlaps_xy, torch.zeros_like(gt_max_overlaps_xy).type_as(gt_max_overlaps_xy))

            detected_xy =  gt_max_overlaps_xy.ne(0).sum()
            n_elements_xy = gt_max_overlaps_xy.nelement()
            true_pos_xy += detected_xy
            false_neg_xy += n_elements_xy - detected_xy

            ## for t - area
            gt_max_overlaps_t, _ = torch.max(overlaps_t, 1)
            gt_max_overlaps_t = torch.where(gt_max_overlaps_t > iou_thresh, gt_max_overlaps_t, torch.zeros_like(gt_max_overlaps_t).type_as(gt_max_overlaps_t))
            detected_t =  gt_max_overlaps_t.ne(0).sum()
            n_elements_t = gt_max_overlaps_t.nelement()
            true_pos_t += detected_t
            false_neg_t += n_elements_t - detected_t

            tubes_sum += 1


    recall    = true_pos.float()    / (true_pos.float()    + false_neg.float())
    recall_xy = true_pos_xy.float() / (true_pos_xy.float() + false_neg_xy.float())
    recall_t  = true_pos_t.float()  / (true_pos_t.float()  + false_neg_t.float())
    print('recall :',recall)
    print(' -----------------------')
    print('| Validation Epoch: {: >3} | '.format(epoch+1))
    print('|                       |')
    print('| Proposed Action Tubes |')
    print('|                       |')
    print('| In {: >6} steps    :  |\n| True_pos   --> {: >6} |\n| False_neg  --> {: >6} | \n| Recall     --> {: >6.4f} |'.format(
        step, true_pos.cpu().tolist()[0], false_neg.cpu().tolist()[0], recall.cpu().tolist()[0]))
    print('|                       |')
    print('| In xy area            |')
    print('|                       |')
    print('| In {: >6} steps    :  |\n| True_pos   --> {: >6} |\n| False_neg  --> {: >6} | \n| Recall     --> {: >6.4f} |'.format(
        step, true_pos_xy.cpu().tolist()[0], false_neg_xy.cpu().tolist()[0], recall_xy.cpu().tolist()[0]))
    print('|                       |')
    print('| In time area          |')
    print('|                       |')
    print('| In {: >6} steps    :  |\n| True_pos   --> {: >6} |\n| False_neg  --> {: >6} | \n| Recall     --> {: >6.4f} |'.format(
        step, true_pos_t.cpu().tolist()[0], false_neg_t.cpu().tolist()[0], recall_t.cpu().tolist()[0]))
    print('|                       |')
    print('| Classification        |')
    print('|                       |')
    print('| In {: >6} steps    :  |'.format(step))
    print('|                       |')
    print('| Correct preds :       |\n| {: >6} / {: >6}       |'.format( correct_preds.cpu().tolist()[0], n_preds.cpu().tolist()[0]))


    print(' -----------------------')
        
if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    dataset_folder = '/gpu-data/sgal/JHMDB-act-detector-frames'
    split_txt_path =  '/gpu-data/sgal/splits'
    boxes_file = '../temporal_localization/poses.json'

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

    spatial_transform = Compose([Scale(sample_size),  # [Resize(sample_size),
                                 ToTensor(),
                                 Normalize(mean, [1, 1, 1])])
    temporal_transform = LoopPadding(sample_duration)

    # Init action_net
    model = Model(classes)
    model.create_architecture()
    model_data = torch.load('./action_net_model_25epoch.pwf')

    model.load_state_dict(model_data)

    model = nn.DataParallel(model)
    model.to(device)

    model.eval()

    validation(0, device, model, dataset_folder, sample_duration, spatial_transform, temporal_transform, boxes_file, split_txt_path, cls2idx, batch_size, n_threads)
