import os
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from conf import conf
from resnet_3D import resnet34
from create_video_id import get_vid_dict
from net_utils import adjust_learning_rate
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, ToTensor, Resize)
from temporal_transforms import LoopPadding

from model_par_transl_v2 import Model

from resize_rpn import resize_rpn, resize_tube
from jhmdb_dataset import Video, video_names
from box_functions import bbox_transform, tube_transform_inv, clip_boxes, tube_overlaps
from bbox_transform import bbox_temporal_overlaps

from plot import *
np.random.seed(42)

def validation(epoch, device, model, dataset_folder, sample_duration, spatial_transform, temporal_transform, boxes_file, splt_txt_path, cls2idx, batch_size, n_threads):

    iou_thresh = 0.5 # Intersection Over Union thresh
    iou_thresh_4 = 0.4 # Intersection Over Union thresh
    iou_thresh_3 = 0.3 # Intersection Over Union thresh

    # vid_name_loader = video_names(dataset_frames, split_txt_path, boxes_file, vid2idx, mode='test', classes_idx=cls2idx)
    vid_name_loader = video_names(dataset_frames, split_txt_path, boxes_file, vid2idx, mode='test', classes_idx=cls2idx, plot=True)
    data_loader = torch.utils.data.DataLoader(vid_name_loader, batch_size=n_devs, num_workers=n_devs, pin_memory=True,
                                              shuffle=True)    # reset learning rate
    # data_loader = torch.utils.data.DataLoader(vid_name_loader, batch_size=1, num_workers=2, pin_memory=True,
    #                                           shuffle=True)    # reset learning rate

    model.eval()

    true_pos = 0
    false_neg = 0

    true_pos_4 = 0
    false_neg_4 = 0

    true_pos_3 = 0
    false_neg_3 = 0

    temp_true_pos = 0
    temp_false_neg = 0

    temp_true_pos_8 = 0
    temp_false_neg_8 = 0

    temp_true_pos_7 = 0
    temp_false_neg_7 = 0
    
    max_overlaps_sum = 0
    temp_max_overlaps_sum = 0

    correct_preds = torch.zeros(1).long().to(device)
    n_preds = torch.zeros(1).long().to(device)
    preds = torch.zeros(1).long().to(device)

    ## 2 rois : 1450
    tubes_sum = 0
    for step, data  in enumerate(data_loader):

        # if step == 25:
        #     break
        # if step == 2:
        #     break
        print('step =>',step)

        # vid_id, clips, boxes, n_frames, n_actions, h, w, target =data
        vid_id, clips, boxes, n_frames, n_actions, h, w, target,plot_cl =data
        vid_id = vid_id.int()

        print('vid_names[vid_id] :',vid_names[vid_id[0]], vid_id[0], 'n_frames :',n_frames[0])
        # print('vid_names[vid_id] :',vid_names[vid_id[1]], vid_id[1], 'n_frames :',n_frames[1])
        clips = clips.to(device)
        boxes = boxes.to(device)
        n_frames = n_frames.to(device)
        n_actions = n_actions.int().to(device)

        im_info = torch.cat([h,w,torch.ones(clips.size(0)).long()*clips.size(2)]).to(device)
        mode = 'noclass'

        with torch.no_grad():
            tubes,  \
            prob_out,_ =  model(n_devs, dataset_frames, \
                                       vid_names, clips, vid_id,  \
                                       None, \
                                       mode, cls2idx, None, n_frames, h, w, None)

        print('tubes.shape :',tubes.shape)
        print('clips.shape :',clips.shape)
        print('tubes.shpae :',tubes.shape)
        print('n_frames :',n_frames)
        print('tubes[:,:10,:40] :',tubes[:,:10,:8].shape)
        # exit(-1)
        # plot_tube(clips[:,:8], [8], tubes[0,:10,:8])
        # plot_tube(plot_cl[:,:8], [8], tubes[0,:10,:8])
        print('plot_cl.shape :',plot_cl.shape)
        plot_tube_with_gt(plot_cl, n_frames, tubes[0,:200],boxes)
        exit(-1)


        for i in range(clips.size(0)):
            
            box = boxes[i,:n_actions[i].int(), :n_frames[i].int(),:4].contiguous()

            # calculate temporal limits of gt tubes
            gt_limits = torch.zeros(n_actions[i].int(),2)
            t = torch.arange(n_frames[i].int().item()).unsqueeze(0).expand(n_actions[i].int().item(),n_frames[i].int().item()).type_as(box)
            z = box.eq(0).all(dim=2)


            gt_limits[:,0] = torch.min(t.masked_fill_(z,1000),dim=1)[0]
            gt_limits[:,1] = torch.max(t.masked_fill_(z,-1),dim=1)[0]+1

            box = box.view(-1,n_frames[i].int()*4)
            tubes_t = tubes[i,:,:n_frames[i].long()]

            # calculate temporal limits of tubes
            tub_limits = torch.zeros(conf.MODEL_POST_NMS_TUBES,2)
            t = torch.arange(n_frames[i].int().item()).unsqueeze(0).expand(conf.MODEL_POST_NMS_TUBES,n_frames[i].int().item()).type_as(tubes_t)

            # tub_limits = torch.zeros(conf.CALC_THRESH,2)
            # t = torch.arange(n_frames[i].int().item()).unsqueeze(0).expand(conf.CALC_THRESH,n_frames[i].int().item()).type_as(tubes_t)

            z = tubes_t.eq(0).all(dim=2)

            tub_limits[:,0] = torch.min(t.masked_fill_(z,1000),dim=1)[0]
            tub_limits[:,1] = torch.max(t.masked_fill_(z,-1),dim=1)[0]+1

            non_empty_indices =  box.ne(0).any(dim=1).nonzero().view(-1)
            n_elems = non_empty_indices.nelement()            

            tubes_t = tubes_t.view(-1, n_frames[i].int()*4)

            ## temporal overlaps
            temp_overlaps = bbox_temporal_overlaps(tub_limits, gt_limits)

            gt_temp_max_overlaps, argmax_temp_gt_overlaps = torch.max(temp_overlaps, 0)
            print('gt_temp_max_overlaps :',gt_temp_max_overlaps.sum())
            temp_max_overlaps_sum += gt_temp_max_overlaps.sum().item()
            print('temp_max_overlaps_sum :',temp_max_overlaps_sum)
            gt_temp_max_overlaps_ = torch.where(gt_temp_max_overlaps > 0.9, gt_temp_max_overlaps, torch.zeros_like(gt_temp_max_overlaps).type_as(gt_temp_max_overlaps))
            detected =  gt_temp_max_overlaps_[non_empty_indices].ne(0).sum()
            temp_true_pos += detected
            temp_false_neg += n_elems - detected

            gt_temp_max_overlaps, argmax_temp_gt_overlaps = torch.max(temp_overlaps, 0)
            gt_temp_max_overlaps_ = torch.where(gt_temp_max_overlaps > 0.8, gt_temp_max_overlaps, torch.zeros_like(gt_temp_max_overlaps).type_as(gt_temp_max_overlaps))
            
            detected =  gt_temp_max_overlaps_[non_empty_indices].ne(0).sum()
            temp_true_pos_8 += detected
            temp_false_neg_8 += n_elems - detected

            gt_temp_max_overlaps, argmax_temp_gt_overlaps = torch.max(temp_overlaps, 0)
            gt_temp_max_overlaps_ = torch.where(gt_temp_max_overlaps > 0.7, gt_temp_max_overlaps, torch.zeros_like(gt_temp_max_overlaps).type_as(gt_temp_max_overlaps))
            
            detected =  gt_temp_max_overlaps_[non_empty_indices].ne(0).sum()
            temp_true_pos_7 += detected
            temp_false_neg_7 += n_elems - detected


            # spatio-temporal overlaps
            overlaps = tube_overlaps(tubes_t.float(), box.float())

            gt_max_overlaps, argmax_gt_overlaps = torch.max(overlaps, 0)

            max_overlaps_sum += gt_max_overlaps.sum().item()

            print('gt_max_overlaps :',gt_max_overlaps)

            # 0.5 thresh
            
            gt_max_overlaps_ = torch.where(gt_max_overlaps > iou_thresh, gt_max_overlaps, torch.zeros_like(gt_max_overlaps).type_as(gt_max_overlaps))
            detected =  gt_max_overlaps_[non_empty_indices].ne(0).sum()
            true_pos += detected
            false_neg += n_elems - detected

            # 0.4 thresh
            gt_max_overlaps_ = torch.where(gt_max_overlaps > iou_thresh_4, gt_max_overlaps, torch.zeros_like(gt_max_overlaps).type_as(gt_max_overlaps))
            detected =  gt_max_overlaps_[non_empty_indices].ne(0).sum()
            true_pos_4 += detected
            false_neg_4 += n_elems - detected

            # 0.3 thresh
            gt_max_overlaps_ = torch.where(gt_max_overlaps > iou_thresh_3, gt_max_overlaps, torch.zeros_like(gt_max_overlaps).type_as(gt_max_overlaps))
            detected =  gt_max_overlaps_[non_empty_indices].ne(0).sum()
            true_pos_3 += detected
            false_neg_3 += n_elems - detected
            

        #     ### TODO add classification step
        # for k in cls_int.cpu().tolist():
        #     if k == target.data:
        #         print('Found one')
        #         correct_preds += 1
        #     n_preds += 1

    recall    = true_pos.float()    / (true_pos.float()    + false_neg.float())
    recall_4  = true_pos_4.float()  / (true_pos_4.float()  + false_neg_4.float())
    recall_3  = true_pos_3.float()  / (true_pos_3.float()  + false_neg_3.float())

    temp_recall    = temp_true_pos.float()    / (temp_true_pos.float()    + temp_false_neg.float())
    temp_recall_8  = temp_true_pos_8.float()  / (temp_true_pos_8.float()  + temp_false_neg_8.float())
    temp_recall_7  = temp_true_pos_7.float()  / (temp_true_pos_7.float()  + temp_false_neg_7.float())

    mabo      = max_overlaps_sum /  (true_pos    + false_neg).float().item()
    temp_mabo = temp_max_overlaps_sum / (true_pos    + false_neg).float().item()
    print('temp_max_overlaps_sum :',temp_max_overlaps_sum)
    print('true_pos    + false_neg) :',true_pos    + false_neg)
    print('temp_mabo :',temp_mabo)
    print(' -----------------------')
    print('| Validation Epoch: {: >3} | '.format(epoch+1))
    print('|                       |')
    print('| Proposed Action Tubes |')
    print('|                       |')
    print('| Single frame          |')
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
    print('|                       |')
    print('| Mean Temp MABO        |')
    print('|                       |')
    print('| {: >6}    |'.format(mabo))
    print('|                       |')
    print('| Temporal overlaps     |')
    print('|                       |')
    print('| Threshold : 0.9       |')
    print('|                       |')
    print('| True_pos   --> {: >6} |\n| False_neg  --> {: >6} | \n| Recall     --> {: >6.4f} |'.format(
        temp_true_pos, temp_false_neg, temp_recall))
    print('|                       |')
    print('| Threshold : 0.8       |')
    print('|                       |')
    print('| True_pos   --> {: >6} |\n| False_neg  --> {: >6} | \n| Recall     --> {: >6.4f} |'.format(
        temp_true_pos_8, temp_false_neg_8, temp_recall_8))
    print('|                       |')
    print('| Threshold : 0.7       |')
    print('|                       |')
    print('| True_pos   --> {: >6} |\n| False_neg  --> {: >6} | \n| Recall     --> {: >6.4f} |'.format(
        temp_true_pos_7, temp_false_neg_7, temp_recall_7))
    print('|                       |')
    print('| Mean      MABO        |')
    print('|                       |')
    print('| {: >6}    |'.format(temp_mabo))


    print(' -----------------------')
        
if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    dataset_frames = '../JHMDB-act-detector-frames'
    split_txt_path =  '../splits'
    boxes_file = '../poses.json'

    n_devs = torch.cuda.device_count()
    sample_size = 112
    # sample_duration = 16  # len(images)
    sample_duration = 8  # len(images)


    batch_size = 1
    # batch_size = 1
    n_threads = 0

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
    temporal_transform = LoopPadding(sample_duration)

    # Init action_net

    action_model_path = './action_net_model_8frm_conf_ucf.pwf'
    
    model = Model(actions, sample_duration, sample_size)
    model.load_part_model(action_model_path=action_model_path)
    # if torch.cuda.device_count() > 1:
    #     print('Using {} GPUs!'.format(torch.cuda.device_count()))
    #     model.act_net = nn.DataParallel(model.act_net)
    # model.act_net.to(device)

    if torch.cuda.device_count() > 1:
        print('Using {} GPUs!'.format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    model.to(device)

    model.eval()

    validation(0, device, model, dataset_frames, sample_duration, spatial_transform, temporal_transform, boxes_file, split_txt_path, cls2idx, batch_size, n_threads)
