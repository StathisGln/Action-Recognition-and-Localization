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
from mAP_temp_function import calculate_mAP
# from model_par_transl_v4 import Model
# from model_par_transl_v3 import Model
# from model_temporal_cls import Model
# from model_temporal_cls_v2 import Model
# from model_temporal_cls_v3 import Model
from model_temporal_cls_v4 import Model
# from model_par_transl_softnms import Model
# from model_par_transl_v2 import Model
from resize_rpn import resize_rpn, resize_tube
from ucf_dataset import Video_UCF, video_names, temp_video_names
from box_functions import bbox_transform, tube_transform_inv, clip_boxes, tube_overlaps
from bbox_transform import bbox_temporal_overlaps

np.random.seed(42)

def validation(epoch, device, model, dataset_folder, sample_duration, spatial_transform, temporal_transform, boxes_file, splt_txt_path, cls2idx, batch_size, n_threads):

    iou_thresh = 0.5 # Intersection Over Union thresh
    iou_thresh_4 = 0.4 # Intersection Over Union thresh
    iou_thresh_3 = 0.3 # Intersection Over Union thresh

    vid_name_loader = temp_video_names(dataset_frames, split_txt_path, boxes_file, vid2idx, mode='test')
    data_loader = torch.utils.data.DataLoader(vid_name_loader, batch_size=n_devs, num_workers=n_devs, pin_memory=True,
                                              shuffle=False)    # reset learning rate

    # data_loader = torch.utils.data.DataLoader(vid_name_loader, batch_size=n_devs, num_workers=n_devs, pin_memory=True,
    #                                           shuffle=True)    # reset learning rate
    # data_loader = torch.utils.data.DataLoader(vid_name_loader, batch_size=1, num_workers=2, pin_memory=True,
    #                                           shuffle=True)    # reset learning rate

    model.eval()
    confidence_thresh = 0.6
    confidence_thresh_7 = 0.75
    confidence_thresh_8 = 0.85

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

    groundtruth_dic = {}
    detection_dic = {}

    detection_dic_7 = {}
    detection_dic_8 = {}

    ## 2 rois : 1450
    tubes_sum = 0
    for step, data  in enumerate(data_loader):

        # if step == 25:
        #     break
        if step == 1:
            break
        print('step =>',step)

        vid_id, clips, boxes, n_frames, n_actions, h, w, limits, target =data
        vid_id = vid_id.int()

        print('vid_names[vid_id] :',vid_names[vid_id[0]], vid_id[0], 'n_frames :',n_frames[0])
        # print('vid_names[vid_id] :',vid_names[vid_id[1]], vid_id[1], 'n_frames :',n_frames[1])
        clips = clips.to(device)
        boxes = boxes.to(device)
        n_frames = n_frames.to(device)
        n_actions = n_actions.int().to(device)

        print('target :',target)
        im_info = torch.cat([h,w,torch.ones(clips.size(0)).long()*clips.size(2)]).to(device)
        mode = 'noclass'

        with torch.no_grad():
            tubes,  \
            prob_out,cls_max_indices =  model(n_devs, dataset_frames, \
                                       vid_names, clips, vid_id,  \
                                       None, \
                                       mode, cls2idx, None, n_frames, h, w, None)


        for i in range(clips.size(0)):

            ## remove padding tubes
            tub_limits = tubes[i]
            nonempty_indices = tub_limits.ne(0).any(dim=1).nonzero().view(-1)

            tub_limits = tub_limits[nonempty_indices]
            tub_max_indices = cls_max_indices[i,nonempty_indices]
            tub_prob_out = prob_out[i,nonempty_indices]

            # ## use confidence_thresh
            f_tubes = torch.cat([tub_max_indices.view(-1,1).type_as(tub_limits), tub_prob_out.view(-1,1).type_as(tub_limits),\
                                 tub_limits.view(-1,2)], dim=1)


            ######################################

            box = boxes[i,:n_actions[i].int(), :n_frames[i].int(),:4].contiguous()
            # # calculate temporal limits of gt tubes
            # gt_limits = torch.zeros(n_actions[i].int(),2).type_as(tub_limits)
            # t = torch.arange(n_frames[i].int().item()).unsqueeze(0).expand(n_actions[i].int().item(),n_frames[i].int().item()).type_as(box)
            # z = box.eq(0).all(dim=2)
            
            # gt_limits[:,0] = torch.min(t.masked_fill_(z,1000),dim=1)[0]
            # gt_limits[:,1] = torch.max(t.masked_fill_(z,-1),dim=1)[0]

            # f_boxes = torch.cat([target[i].view(1,1).contiguous().expand(gt_limits.size(0),1).type_as(gt_limits),\
            #                      gt_limits], dim=1)

            # print('target :',target)
            # print('target[i].view(1,1).contiguous().expand(limits.size(0),1).shape :',target[i].view(1,1).contiguous().expand(limits[i].size(0),1).shape)
            # print('target :',target.shape)
            # print('limits.shape:',limits.shape)
            # print('limits.shape:',limits[i].shape)
            f_boxes = torch.cat([target[i].view(1,1).contiguous().expand(limits.size(1),1).type_as(tubes),\
                                 limits[i].contiguous().type_as(tubes)], dim=1)
            v_name = vid_names[vid_id[i]].split('/')[1]

            # for 0.6
            # _,sort_idx = torch.topk(f_tubes[:,1],f_tubes.size(0))
            _,sort_idx = torch.topk(f_tubes[:,1],10)
            f_tubes = f_tubes[sort_idx]

            keep_indices = f_tubes[:,1].gt(confidence_thresh).nonzero().view(-1)

            detection_dic[v_name] = f_tubes[keep_indices].float().contiguous()
            groundtruth_dic[v_name] = f_boxes.type_as(f_tubes)


            keep_indices = f_tubes[:,1].gt(confidence_thresh_7).nonzero().view(-1)
            detection_dic_7[v_name] = f_tubes[keep_indices].float().contiguous()


            keep_indices = f_tubes[:,1].gt(confidence_thresh_8).nonzero().view(-1)
            detection_dic_8[v_name] = f_tubes[keep_indices].float().contiguous()

            # print('detection_dic :',detection_dic)
            # print('detection_dic :',detection_dic_7)
            # print('detection_dic :',detection_dic_8)

            box = box.view(-1,n_frames[i].int()*4)

            non_empty_indices =  box.ne(0).any(dim=1).nonzero().view(-1)
            n_elems = non_empty_indices.nelement()            

            ## temporal overlaps
            # temp_overlaps = bbox_temporal_overlaps(tub_limits, gt_limits.type_as(tub_limits))
            print('tub_limits :',tub_limits)
            print('limits[i] :',limits[i])
            print('tub_limits.shape :',tub_limits.shape)
            exit(-1)
            temp_overlaps = bbox_temporal_overlaps(tub_limits, limits[i].type_as(tub_limits))
            gt_temp_max_overlaps, argmax_temp_gt_overlaps = torch.max(temp_overlaps, 0)
            temp_max_overlaps_sum += gt_temp_max_overlaps.sum().item()
            print('gt_temp_max_overlaps :',gt_temp_max_overlaps)

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


            # # spatio-temporal overlaps
            # overlaps = tube_overlaps(tubes_t.float(), box.float())

            # gt_max_overlaps, argmax_gt_overlaps = torch.max(overlaps, 0)

            # max_overlaps_sum += gt_max_overlaps.sum().item()

            # print('gt_max_overlaps :',gt_max_overlaps)

            # # 0.5 thresh
            
            # gt_max_overlaps_ = torch.where(gt_max_overlaps > iou_thresh, gt_max_overlaps, torch.zeros_like(gt_max_overlaps).type_as(gt_max_overlaps))
            # detected =  gt_max_overlaps_[non_empty_indices].ne(0).sum()
            # true_pos += detected
            # false_neg += n_elems - detected

            # # 0.4 thresh
            # gt_max_overlaps_ = torch.where(gt_max_overlaps > iou_thresh_4, gt_max_overlaps, torch.zeros_like(gt_max_overlaps).type_as(gt_max_overlaps))
            # detected =  gt_max_overlaps_[non_empty_indices].ne(0).sum()
            # true_pos_4 += detected
            # false_neg_4 += n_elems - detected

            # # 0.3 thresh
            # gt_max_overlaps_ = torch.where(gt_max_overlaps > iou_thresh_3, gt_max_overlaps, torch.zeros_like(gt_max_overlaps).type_as(gt_max_overlaps))
            # detected =  gt_max_overlaps_[non_empty_indices].ne(0).sum()
            # true_pos_3 += detected
            # false_neg_3 += n_elems - detected
            

        #     ### TODO add classification step
        # for k in cls_int.cpu().tolist():
        #     if k == target.data:
        #         print('Found one')
        #         correct_preds += 1
        #     n_preds += 1

    # recall    = true_pos.float()    / (true_pos.float()    + false_neg.float())
    # recall_4  = true_pos_4.float()  / (true_pos_4.float()  + false_neg_4.float())
    # recall_3  = true_pos_3.float()  / (true_pos_3.float()  + false_neg_3.float())

    recall    = -1
    recall_4  = -1
    recall_3  = -1


    temp_recall    = temp_true_pos.float()    / (temp_true_pos.float()    + temp_false_neg.float())
    temp_recall_8  = temp_true_pos_8.float()  / (temp_true_pos_8.float()  + temp_false_neg_8.float())
    temp_recall_7  = temp_true_pos_7.float()  / (temp_true_pos_7.float()  + temp_false_neg_7.float())

    mabo      = -1
    temp_mabo = temp_max_overlaps_sum / (temp_true_pos    + temp_false_neg).float().item()

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

    print(' -------------------')
    print('|                   |')
    print('|  mAP Thresh : 0.5 |')
    print('|                   |')
    print('| conf Thresh : 0.6 |')
    print('|                   |')
    print(' -------------------')
    calculate_mAP(detection_dic, groundtruth_dic, iou_thresh)

    print(' -------------------')
    print('|                   |')
    print('|  mAP Thresh : 0.5 |')
    print('|                   |')
    print('| conf Thresh :0.75 |')
    print('|                   |')
    print(' -------------------')
    calculate_mAP(detection_dic_7, groundtruth_dic, iou_thresh)

    print(' -------------------')
    print('|                   |')
    print('|  mAP Thresh : 0.5 |')
    print('|                   |')
    print('| conf Thresh :0.85 |')
    print('|                   |')
    print(' -------------------')
    calculate_mAP(detection_dic_8, groundtruth_dic, iou_thresh)

    print(' $$$$$$$$$$$$$$$$$$$\n')

    print(' -------------------')
    print('|                   |')
    print('|  mAP Thresh : 0.4 |')
    print('|                   |')
    print('| conf Thresh : 0.6 |')
    print('|                   |')
    print(' -------------------')
    calculate_mAP(detection_dic, groundtruth_dic, iou_thresh_4)

    print(' -------------------')
    print('|                   |')
    print('|  mAP Thresh : 0.4 |')
    print('|                   |')
    print('| conf Thresh :0.75 |')
    print('|                   |')
    print(' -------------------')
    calculate_mAP(detection_dic_7, groundtruth_dic, iou_thresh_4)

    print(' -------------------')
    print('|                   |')
    print('|  mAP Thresh : 0.4 |')
    print('|                   |')
    print('| conf Thresh :0.85 |')
    print('|                   |')
    print(' -------------------')
    calculate_mAP(detection_dic_8, groundtruth_dic, iou_thresh_4)

    print(' $$$$$$$$$$$$$$$$$$$\n')

    print(' -------------------')
    print('|                   |')
    print('|  mAP Thresh : 0.3 |')
    print('|                   |')
    print('| conf Thresh : 0.6 |')
    print('|                   |')
    print(' -------------------')
    calculate_mAP(detection_dic, groundtruth_dic, iou_thresh_3)

    print(' -------------------')
    print('|                   |')
    print('|  mAP Thresh : 0.3 |')
    print('|                   |')
    print('| conf Thresh :0.75 |')
    print('|                   |')
    print(' -------------------')
    calculate_mAP(detection_dic_7, groundtruth_dic, iou_thresh_3)

    print(' -------------------')
    print('|                   |')
    print('|  mAP Thresh : 0.3 |')
    print('|                   |')
    print('| conf Thresh :0.85 |')
    print('|                   |')
    print(' -------------------')
    calculate_mAP(detection_dic_8, groundtruth_dic, iou_thresh_3)


    # print(' -------------------')
    # print('|                   |')
    # print('| mAP Thresh : 0.4  |')
    # print('|                   |')
    # print(' -------------------')
    # calculate_mAP(detection_dic, groundtruth_dic, iou_thresh_4)

    # print(' ------------------')
    # print('|                  |')
    # print('| mAP Thresh : 0.3 |')
    # print('|                  |')
    # print(' ------------------')
    # calculate_mAP(detection_dic, groundtruth_dic, iou_thresh_3)

        
if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)
    print('torch.get_num_threads() :',torch.get_num_threads())
    torch.set_num_threads(4)
    print('torch.get_num_threads() :',torch.get_num_threads())

    dataset_frames = '../UCF-101-frames'
    boxes_file = '../pyannot.pkl'
    split_txt_path = '../UCF101_Action_detection_splits/'

    n_devs = torch.cuda.device_count()
    sample_size = 112
    sample_duration = 16  # len(images)
    # sample_duration = 8  # len(images)


    batch_size = 1
    # batch_size = 1
    n_threads = 0

    # # get mean
    mean = [112.07945832, 112.87372333, 106.90993363]  # ucf-101 24 classes

    # generate model

    # actions = ['__background__', 'Basketball','BasketballDunk','Biking','CliffDiving','CricketBowling',
    #            'Diving','Fencing','FloorGymnastics','GolfSwing','HorseRiding','IceDancing',
    #            'LongJump','PoleVault','RopeClimbing','SalsaSpin','SkateBoarding','Skiing',
    #            'Skijet','SoccerJuggling','Surfing','TennisSwing','TrampolineJumping',
    #            'VolleyballSpiking','WalkingWithDog']
    actions = [ 'Basketball','BasketballDunk','Biking','CliffDiving','CricketBowling',
               'Diving','Fencing','FloorGymnastics','GolfSwing','HorseRiding','IceDancing',
               'LongJump','PoleVault','RopeClimbing','SalsaSpin','SkateBoarding','Skiing',
               'Skijet','SoccerJuggling','Surfing','TennisSwing','TrampolineJumping',
                'VolleyballSpiking','WalkingWithDog', '__background__']



    cls2idx = {actions[i]: i for i in range(0, len(actions))}

    ### get videos id
    vid2idx,vid_names = get_vid_dict(dataset_frames)

    spatial_transform = Compose([Scale(sample_size),  # [Resize(sample_size),
                                 ToTensor(),
                                 Normalize(mean, [1, 1, 1])])
    temporal_transform = LoopPadding(sample_duration)

    
    # Init action_net
    print('conf.TEST.RPN_POST_NMS_TOP_N :',conf.TEST.RPN_POST_NMS_TOP_N)
    # action_model_path = './action_net_model_8frm_conf_ucf.pwf'
    action_model_path = './action_net_model_16frm_conf_ucf.pwf'
    rnn_path = './action_net_cls_ucf_16frm.pwf'

    
    model = Model(actions, sample_duration, sample_size)
    model.load_part_model(action_model_path=action_model_path, rnn_path=rnn_path)
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

