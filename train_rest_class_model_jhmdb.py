import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from mAP_function import calculate_mAP

from jhmdb_dataset import  video_names, RNN_JHMDB

from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, ToTensor, Resize)
from temporal_transforms import LoopPadding, LoopPadding_still

from create_video_id import get_vid_dict
from net_utils import adjust_learning_rate
from resize_rpn import resize_rpn, resize_tube

from resize_rpn import resize_boxes
import argparse
from box_functions import tube_overlaps, tube_transform_inv
from rest_model import RestNet
from model_all_restnet import Model


np.random.seed(42)

def validation(epoch, device, model, dataset_folder, sample_duration, spatial_transform, temporal_transform, boxes_file, splt_txt_path, cls2idx, batch_size, n_threads):

    iou_thresh = 0.5 # Intersection Over Union thresh
    iou_thresh_4 = 0.4 # Intersection Over Union thresh
    iou_thresh_3 = 0.3 # Intersection Over Union thresh

    confidence_thresh = 0.5

    vid_name_loader = video_names(dataset_folder, split_txt_path, boxes_file, vid2idx, mode='test', classes_idx= cls2idx)
    data_loader = torch.utils.data.DataLoader(vid_name_loader, batch_size=n_devs, num_workers=n_devs, pin_memory=True,
                                              shuffle=True)    # reset learning rate
    # data_loader = torch.utils.data.DataLoader(vid_name_loader, batch_size=1, num_workers=8*n_devs, pin_memory=True,
    #                                           shuffle=True)    # reset learning rate
    model.eval()

    true_pos = 0
    false_neg = 0

    true_pos_4 = 0
    false_neg_4 = 0

    true_pos_3 = 0
    false_neg_3 = 0

    ## 2 rois : 1450
    tubes_sum = 0

    groundtruth_dic = {}
    detection_dic = {}


    for step, data  in enumerate(data_loader):

        # if step == 1:
        #     break
        print('step =>',step)

        vid_id, clips, boxes, n_frames, n_actions, h, w, target =data

        vid_id = vid_id.int()
        clips = clips.to(device)
        boxes = boxes.to(device)
        n_frames = n_frames.to(device)
        n_actions = n_actions.int().to(device)
        target = target.to(device)
        im_info = torch.cat([h,w,torch.ones(clips.size(0)).long()*clips.size(2)]).to(device)
        mode = 'test'

        with torch.no_grad():
            tubes,  \
            prob_out, n_tubes =  model(n_devs, dataset_folder, \
                                    vid_names, clips, vid_id,  \
                                    None, \
                                    mode, cls2idx, None, n_frames, h, w)


        for i in range(clips.size(0)):

            _, cls_int = torch.max(prob_out[i],1)

            # print('cls_int :',cls_int)
            f_prob = torch.zeros(n_tubes[i].long()).type_as(prob_out)

            for j in range(n_tubes[i].long()):
                f_prob[j] = prob_out[i,j,cls_int[j]]
            
            cls_int = cls_int[:n_tubes[i].long()]

            keep_ = (f_prob.ge(confidence_thresh)) & cls_int.ne(0)
            keep_indices = keep_.nonzero().view(-1)

            f_tubes = torch.cat([cls_int.view(-1,1).type_as(tubes),f_prob.view(-1,1).type_as(tubes), \
                                 tubes[i,:n_tubes[i].long(),:n_frames[i]].contiguous().view(-1,n_frames[i]*4)], dim=1)


            f_tubes = f_tubes[keep_indices].contiguous()

            if f_tubes.nelement() != 0 :
                _, best_tube = torch.max(f_tubes[:,1],dim=0)
                f_tubes= f_tubes[best_tube].unsqueeze(0)

            f_boxes = torch.cat([target.type_as(boxes),boxes[i,:,:n_frames[i],:4].contiguous().view(n_frames[i]*4)]).unsqueeze(0)
            v_name = vid_names[vid_id[i]].split('/')[1]

            detection_dic[v_name] = f_tubes.float()
            groundtruth_dic[v_name] = f_boxes.type_as(f_tubes)

            box = boxes[i,:n_actions[i].long(), :n_frames[i].long(),:4].contiguous()
            box = box.view(-1,n_frames[i]*4).contiguous().type_as(tubes)

            overlaps = tube_overlaps(tubes[i,:n_tubes[i].long(),:n_frames[i].long()].view(-1,n_frames[i].long()*4).float(),\
                                     box.view(-1,n_frames[i]*4).float())
            gt_max_overlaps, argmax_gt_overlaps = torch.max(overlaps, 0)

            non_empty_indices =  box.ne(0).any(dim=1).nonzero().view(-1)
            n_elems = non_empty_indices.nelement()

            # 0.5 thresh
            gt_max_overlaps_ = torch.where(gt_max_overlaps > iou_thresh, gt_max_overlaps,\
                                           torch.zeros_like(gt_max_overlaps).type_as(gt_max_overlaps))
            detected =  gt_max_overlaps_[non_empty_indices].ne(0).sum().item()
            true_pos += detected
            false_neg += n_elems - detected

            # 0.4 thresh
            gt_max_overlaps_ = torch.where(gt_max_overlaps > iou_thresh_4, gt_max_overlaps,\
                                           torch.zeros_like(gt_max_overlaps).type_as(gt_max_overlaps))
            # print('gt_max_overlaps_ :',gt_max_overlaps_)
            detected =  gt_max_overlaps_[non_empty_indices].ne(0).sum().item()
            true_pos_4 += detected
            false_neg_4 += n_elems - detected

            # 0.3 thresh
            gt_max_overlaps_ = torch.where(gt_max_overlaps > iou_thresh_3, gt_max_overlaps,\
                                           torch.zeros_like(gt_max_overlaps).type_as(gt_max_overlaps))
            # print('gt_max_overlaps_ :',gt_max_overlaps_)
            detected =  gt_max_overlaps_[non_empty_indices].ne(0).sum().item()
            true_pos_3 += detected
            false_neg_3 += n_elems - detected

    recall    = float(true_pos)     /  (true_pos    + false_neg)  if true_pos > 0 or false_neg > 0 else 0
    recall_4  = float(true_pos_4)  / (true_pos_4  + false_neg_4)  if true_pos_4 > 0 or false_neg_4 > 0 else 0
    recall_3  = float(true_pos_3)  / (true_pos_3  + false_neg_3)  if true_pos_3 > 0 or false_neg_3 > 0 else 0

    print(' -----------------------')
    print('| Validation Epoch: {: >3} | '.format(epoch+1))
    print('|                       |')
    print('| Proposed Action Tubes |')
    print('|                       |')
    print('| Tube recall           |')
    print('|                       |')
    print('| In {: >6} steps    :  |'.format(step))
    print('|                       |')
    print('| Threshold : 0.5       |')
    print('|                       |')
    print('| True_pos   --> {: >6} |\n| False_neg  --> {: >6} | \n| Recall     --> {: >6.4f} |'.format(
        true_pos, false_neg, recall))
    print('|                       |')
    print('| Threshold : 0.4       |')
    print('|                       |')
    print('| True_pos   --> {: >6} |\n| False_neg  --> {: >6} | \n| Recall     --> {: >6.4f} |'.format(
        true_pos_4, false_neg_4, recall_4))
    print('|                       |')
    print('| Threshold : 0.3       |')
    print('|                       |')
    print('| True_pos   --> {: >6} |\n| False_neg  --> {: >6} | \n| Recall     --> {: >6.4f} |'.format(
        true_pos_3, false_neg_3, recall_3))


    print(' -----------------------')
        
    print(' -----------------')
    print('|                  |')
    print('| mAP Thresh : 0.5 |')
    print('|                  |')
    print(' ------------------')
    calculate_mAP(detection_dic, groundtruth_dic, iou_thresh)

    print(' -------------------')
    print('|                   |')
    print('| mAP Thresh : 0.4  |')
    print('|                   |')
    print(' -------------------')
    calculate_mAP(detection_dic, groundtruth_dic, iou_thresh_4)

    print(' ------------------')
    print('|                  |')
    print('| mAP Thresh : 0.3 |')
    print('|                  |')
    print(' ------------------')
    calculate_mAP(detection_dic, groundtruth_dic, iou_thresh_3)
    
        
def training(epoch, device, model, dataset_folder, sample_duration, spatial_transform, temporal_transform, boxes_file, splt_txt_path, cls2idx, batch_size, n_threads, lr, mode = 1):

    vid_name_loader = RNN_JHMDB(dataset_folder, split_txt_path, boxes_file, vid2idx, mode='train', sample_duration=sample_duration)
    # data_loader = torch.utils.data.DataLoader(vid_name_loader, batch_size=batch_size*16,
    #                                           shuffle=True, num_workers=batch_size*8, pin_memory=True)
    data_loader = torch.utils.data.DataLoader(vid_name_loader, batch_size=batch_size*8,
                                              shuffle=True, num_workers=batch_size*4, pin_memory=True)

    # data_loader = torch.utils.data.DataLoader(vid_name_loader, batch_size=2,
    #                                           shuffle=True, num_workers=0, pin_memory=True)

    model.train()
    loss_temp = 0
    
    ## 2 rois : 1450
    for step, data  in enumerate(data_loader):

        # if step == 2:
        #     exit(-1)
        #     break
        # print('step =>',step)

        f_features, n_tubes, target_lbl, len_tubes  = data        

        f_features = f_features.to(device)
        n_tubes = n_tubes.to(device)
        target_lbl = target_lbl.to(device)

        cls_scr,cls_loss = model(f_features, n_tubes, len_tubes, target_lbl)

        loss = cls_loss.mean()
        loss_temp += loss.item()

        # backw\ard
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Train Epoch: {} \tLoss: {:.6f}\t lr : {:.6f}'.format(
        epoch+1,loss_temp/(step+1), lr))

    return model, loss_temp

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train action_net, regression layer and RNN')

    # parser.add_argument('--demo', '-d', help='Run just 2 steps for test if everything works fine', action='store_true')
    # parser.add_argument('--n_1_1', help='Run only part 1.1, training action net only', action='store_true')
    # parser.add_argument('--n_1_2', help='Run only part 1.2, training only regression layer', action='store_true')
    # parser.add_argument('--n_2', help='Run only part 2, train only RNN', action='store_true')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    dataset_frames = '../JHMDB-act-detector-frames'
    split_txt_path =  '../splits'
    boxes_file = '../poses.json'

    sample_size = 112
    # sample_duration = 16  # len(images)
    sample_duration = 8  # len(images)

    n_devs = torch.cuda.device_count()

    # # get mean
    mean = [103.29825354, 104.63845484,  90.79830328]  # jhmdb from .png


    # generate model
    actions = ['__background__','brush_hair', 'clap', 'golf', 'kick_ball', 'pour',
               'push', 'shoot_ball', 'shoot_gun', 'stand', 'throw', 'wave',
               'catch','climb_stairs', 'jump', 'pick', 'pullup', 'run', 'shoot_bow', 'sit',
               'swing_baseball', 'walk' ]

    cls2idx = {actions[i]: i for i in range(0, len(actions))}
    ### get videos id
    vid2idx,vid_names = get_vid_dict(dataset_frames)

    spatial_transform = Compose([Scale(sample_size),  # [Resize(sample_size),
                                 ToTensor(),
                                 Normalize(mean, [1, 1, 1])])
    temporal_transform = LoopPadding_still(sample_duration)
    # temporal_transform = LoopPadding(sample_duration)


    n_classes = len(actions)
    print(' -------------------------------------')
    print('|          - train RestNet -          |')
    print(' -------------------------------------')

    dataset_frames = '../JHMDB-act-detector-frames'
    dataset_features = './JHMDB-features-256-7ver2'

    lr = 0.1
    lr_decay_step = 15
    lr_decay_gamma = 0.1
    
    params = []

    model = RestNet(actions, sample_duration)
    model.create_architecture()

    model = nn.DataParallel(model)
    model = model.to(device)
    
    for key, value in dict(model.named_parameters()).items():
        # print(key, value.requires_grad)
        if value.requires_grad:
            print('key :',key)
            if 'bias' in key:
                params += [{'params':[value],'lr':lr*(True + 1), \
                            'weight_decay': False and 0.0005 or 0}]
            else:
                params += [{'params':[value],'lr':lr, 'weight_decay': 0.0005}]


    lr = lr * 0.1
    optimizer = torch.optim.Adam(params)

    # epochs = 200
    epochs = 60
    # epochs = 5

    file = open('train_loss_jhmdb_linear.txt', 'w')

    n_devs = torch.cuda.device_count()
    for epoch in range(epochs):
        print(' ============\n| Epoch {:0>2}/{:0>2} |\n ============'.format(epoch+1, epochs))

        if epoch % (lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, lr_decay_gamma)
            lr *= lr_decay_gamma

        model, loss = training(epoch, device, model, dataset_features, sample_duration, spatial_transform, temporal_transform, boxes_file, split_txt_path, cls2idx, n_devs, 0, lr, mode=4)
        file.write('epoch :'+str(epoch)+' --> '+str(loss)+'\n')

        if ( epoch + 1 ) % 5 == 0:
            torch.save(model.module.state_dict(), "RestNet.pwf".format(epoch+1))

        if ( epoch + 1 ) % 10 == 0:

            model = Model(actions, sample_duration, sample_size)

            action_model_path = './action_net_model_8frm_2_avg_jhmdb.pwf'
            linear_path = './RestNet.pwf'

            model.load_part_model(action_model_path=action_model_path, rnn_path=linear_path)

            model = nn.DataParallel(model)
            model.to(device)
            model.eval()
    
            validation(0, device, model, dataset_frames, sample_duration, spatial_transform, temporal_transform, boxes_file, split_txt_path, cls2idx, n_devs, 0)


    torch.save(model.module.state_dict(), "RestNet.pwf".format(epoch+1))



