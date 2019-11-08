import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from mAP_function import calculate_mAP
from conf import conf

from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, ToTensor, Resize)
from temporal_transforms import LoopPadding

from net_utils import adjust_learning_rate

from create_video_id import get_vid_dict
from jhmdb_dataset import  video_names, RNN_JHMDB

# from model_par_all  import Model
from model_par_all_v2  import Model
# from model_par_all_v3  import Model

from resize_rpn import resize_tube
from box_functions import bbox_transform, tube_transform_inv, clip_boxes, tube_overlaps

def validation(epoch, device, model, dataset_folder, sample_duration, spatial_transform, temporal_transform, boxes_file, splt_txt_path, cls2idx, batch_size, n_threads):

    iou_thresh = 0.5 # Intersection Over Union thresh
    iou_thresh_4 = 0.4 # Intersection Over Union thresh
    iou_thresh_3 = 0.3 # Intersection Over Union thresh

    confidence_thresh = 0.1

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

    n_tubes = conf.ALL_SCORES_THRESH
    for step, data  in enumerate(data_loader):

        # if step == 2:
        #     break
        print('step =>',step)

        vid_id, clips, boxes, n_frames, n_actions, h, w, target =data

        print('n_frames :',n_frames)
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
            prob_out, _ =  model(n_devs, dataset_folder, \
                                 vid_names, clips, vid_id,  \
                                 None, \
                                 mode, cls2idx, None, n_frames, h, w, None)



        for i in range(clips.size(0)):

            _, cls_int = torch.max(prob_out[i],1)

            # print('cls_int :',cls_int)
            f_prob = torch.zeros(n_tubes).type_as(prob_out)

            for j in range(n_tubes):
                f_prob[j] = prob_out[i,j,cls_int[j]]
            
            cls_int = cls_int[:n_tubes]

            keep_ = (f_prob.ge(confidence_thresh)) & cls_int.ne(0)
            keep_indices = keep_.nonzero().view(-1)

            
            f_tubes = torch.cat([cls_int.view(-1,1).type_as(tubes),f_prob.view(-1,1).type_as(tubes), \
                                 tubes[i,:n_tubes,:n_frames[i]].contiguous().view(-1,n_frames[i]*4)], dim=1)

            if keep_indices.nelement() == 0:
                print('no tube over thresh :')
                keep_ =  cls_int.ne(0)
                keep_indices = keep_.nonzero().view(-1)
                if keep_indices.nelement() == 0:
                    f_tubes = torch.zeros((1,f_tubes.size(1)))
            else:
                f_tubes = f_tubes[keep_indices].contiguous()

            if f_tubes.nelement() != 0 :
                _, best_tube = torch.max(f_tubes[:,1],dim=0)
                f_tubes= f_tubes[best_tube].unsqueeze(0)

            f_boxes = torch.cat([target[i].type_as(boxes).unsqueeze(0),\
                                 boxes[i,:,:n_frames[i],:4].contiguous().view(n_frames[i]*4)]).unsqueeze(0)
            v_name = vid_names[vid_id[i]].split('/')[1]

            detection_dic[v_name] = f_tubes.float()
            groundtruth_dic[v_name] = f_boxes.type_as(f_tubes)

            box = boxes[i,:n_actions[i].long(), :n_frames[i].long(),:4].contiguous()
            box = box.view(-1,n_frames[i]*4).contiguous().type_as(tubes)

            overlaps = tube_overlaps(tubes[i,:n_tubes,:n_frames[i].long()].view(-1,n_frames[i].long()*4).float(),\
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

if __name__ == '__main__':
    
    ###################################
    #        JHMDB data inits         #
    ###################################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    dataset_frames = '../JHMDB-act-detector-frames'
    split_txt_path =  '../splits'
    boxes_file = '../poses.json'

    sample_size = 112
    # sample_duration = 16  # len(images)
    sample_duration = 8  # len(images)


    # # # get mean
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

    n_classes = len(actions)


    ##########################################
    #          Model Initialization          #
    ##########################################

    # Init action_net
    # action_model_path = './action_net_model_16frm_max_jhmdb.pwf'
    action_model_path = './action_net_model_8frm_2_avg_jhmdb.pwf'
    # action_model_path = './action_net_model_4frm_max_jhmdb.pwf'

    model = Model(actions, sample_duration, sample_size)    
    # model.load_part_model()
    model.load_part_model(action_model_path=action_model_path, rnn_path=None)
    model.deactivate_action_net_grad()

    lr = 0.001
    lr_decay_step = 20
    lr_decay_gamma = 0.1

    params = []
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
    # optimizer = optim.SGD(tcn_net.parameters(), lr = lr)

    #######################################
    #          Train starts here          #
    #######################################

    n_devs = torch.cuda.device_count()
    batch_size = n_devs
    vid_name_loader = video_names(dataset_frames, split_txt_path, boxes_file, vid2idx, mode='train', classes_idx=cls2idx)
    data_loader = torch.utils.data.DataLoader(vid_name_loader, batch_size=batch_size,
                                              shuffle=True)

    epochs = 60
    # epochs = 5

    if torch.cuda.device_count() > 1:

        print('Using {} GPUs!'.format(torch.cuda.device_count()))
        # model.act_net = nn.DataParallel(model.act_net)
        model = nn.DataParallel(model)

    model.to(device)

    for ep in range(epochs):

        model.train()
        loss_temp = 0

        # start = time.time()
        if (ep +1) % (lr_decay_step ) == 0:
            print('time to reduce learning rate ', lr)
            adjust_learning_rate(optimizer, lr_decay_gamma)
            lr *= lr_decay_gamma


        print(' ============\n| Epoch {:0>2}/{:0>2} |\n ============'.format(ep+1, epochs))
        for step, data  in enumerate(data_loader):

            vid_id, clips, boxes, n_frames, n_actions, h, w, target =data

            mode = 'train'
            # if step == 2:
            #     exit(-1)
            print('step =>',step)

            vid_id = vid_id.int()
            clips = clips.to(device)
            boxes = boxes.to(device)
            n_frames = n_frames.to(device)
            n_actions = n_actions.to(device)
            h = h.to(device)
            w = w.to(device)

            tubes,   \
            prob_out, cls_loss =  model(n_devs, dataset_frames, \
                                        vid_names, clips, vid_id, \
                                        boxes, \
                                        mode, cls2idx, n_actions,n_frames, h, w, target)

            loss =  cls_loss.mean()

            # backw\ard
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_temp += loss.item()

        print('Train Epoch: {} \tLoss: {:.6f}\t lr : {:.6f}'.format(
        ep+1,loss_temp/step, lr))
    
        if ( ep + 1 ) % 5 == 0:
            torch.save(model.state_dict(), "model_jhmdb_epoch_{}.pwf".format(ep+1))
        if (ep + 1) % (10) == 0:
            print(' ============\n| Validation {:0>2}/{:0>2} |\n ============'.format(ep+1, epochs))

            validation(0, device, model, dataset_frames, sample_duration, spatial_transform, temporal_transform, boxes_file, split_txt_path, cls2idx, n_devs, 0)            

    torch.save(model.state_dict(), "model_jhmdb.pwf")


