import os
import numpy as np
import json

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from resnet_3D import resnet34
from create_video_id import get_vid_dict
from net_utils import adjust_learning_rate
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, ToTensor, Resize)
from temporal_transforms import LoopPadding
# from model import Model
from model_all_svm import Model
from resize_rpn import resize_rpn, resize_tube
from jhmdb_dataset import Video, video_names

from box_functions import bbox_transform, tube_transform_inv, clip_boxes, tube_overlaps
from mAP_function import calculate_mAP
from plot import plot_tube_with_gt
np.random.seed(42)

def validation(epoch, device, model, dataset_folder, sample_duration, spatial_transform, temporal_transform, boxes_file, splt_txt_path, cls2idx, batch_size, n_threads):

    iou_thresh = 0.5 # Intersection Over Union thresh
    iou_thresh_4 = 0.4 # Intersection Over Union thresh
    iou_thresh_3 = 0.3 # Intersection Over Union thresh

    confidence_thresh = 0.5
    vid_name_loader = video_names(dataset_folder, split_txt_path, boxes_file, vid2idx, mode='test', classes_idx=cls2idx, plot=True)
    # data_loader = torch.utils.data.DataLoader(vid_name_loader, batch_size=n_devs, num_workers=8*n_devs, pin_memory=True,
    #                                           shuffle=True)    # reset learning rate
    data_loader = torch.utils.data.DataLoader(vid_name_loader, batch_size=1, num_workers=8*n_devs, pin_memory=True,
                                              shuffle=True)    # reset learning rate
    # data_loader = torch.utils.data.DataLoader(vid_name_loader, batch_size=1, num_workers=8*n_devs, pin_memory=True,
    #                                           shuffle=False)    # reset learning rate


    model.eval()
    
    true_pos = 0
    false_neg = 0

    true_pos_4 = 0
    false_neg_4 = 0

    true_pos_3 = 0
    false_neg_3 = 0

    correct_preds = torch.zeros(1).long().to(device)
    n_preds = torch.zeros(1).long().to(device)
    preds = torch.zeros(1).long().to(device)

    ## 2 rois : 1450
    tubes_sum = 0

    groundtruth_dic = {}
    detection_dic = {}

    for step, data  in enumerate(data_loader):

        # if step == 20:
        #     break
        # if step == 3:
        #     break

        print('step =>',step)

        # vid_id, clips, boxes, n_frames, n_actions, h, w, target =data
        vid_id, clips, boxes, n_frames, n_actions, h, w, target, clips_plot =data
        vid_id = vid_id.int()
        print('vid_id :',vid_id)
        print('vid_name_loader[vid_id] :',vid_names[vid_id])
        # print('target :',target)
        
        clips = clips.to(device)
        boxes = boxes.to(device)
        n_frames = n_frames.to(device)
        n_actions = n_actions.int().to(device)
        target = target.to(device)

        im_info = torch.cat([h,w,torch.ones(clips.size(0)).long()*clips.size(2)]).to(device)
        mode = 'test'
        print('target :',target)
        print('n_frames :',n_frames)
        # print('vid_id :',vid_id)
        # print('vid_names[vid_id] :',vid_names[vid_id])
        with torch.no_grad():
            tubes,  \
            prob_out, n_tubes =  model(n_devs, dataset_folder, \
                                    vid_names, clips, vid_id,  \
                                    None, \
                                    mode, cls2idx, None, n_frames, h, w)



        # print('out...')
        # get predictions

        for i in range(batch_size):

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
            # print('f_tubes.shape :',f_tubes.shape)
            # print('f_tubes.shape :',f_tubes[:, :2])


            f_boxes = torch.cat([target.type_as(boxes),boxes[i,:,:n_frames[i],:4].contiguous().view(n_frames[i]*4)]).unsqueeze(0)
            v_name = vid_names[vid_id[i]].split('/')[1]
            # print('f_tubes :',f_tubes.cpu().detach().numpy())
            # print('f_boxes :',f_boxes.cpu().detach().numpy())
            detection_dic[v_name] = f_tubes.float()
            groundtruth_dic[v_name] = f_boxes.type_as(f_tubes)
            # with open(os.path.join('outputs','detection',v_name+'.json'), 'w') as f:
            #     json.dump(f_tubes.cpu().tolist(), f)
            # with open(os.path.join('outputs','groundtruth',v_name+'.json'), 'w') as f:
            #     json.dump(f_boxes.cpu().tolist(), f)


        for i in range(clips.size(0)):
            # print('boxes.shape:',boxes.shape)
            # print('tubes.shape :',tubes.shape)
            box = boxes[i,:n_actions[i].long(), :n_frames[i].long(),:4].contiguous()
            box = box.view(-1,n_frames[i]*4).contiguous().type_as(tubes)

            overlaps = tube_overlaps(tubes[i,:n_tubes[i].long(),:n_frames[i].long()].view(-1,n_frames*4).float(),\
                                     box.view(-1,n_frames[i]*4).float())
            gt_max_overlaps, argmax_gt_overlaps = torch.max(overlaps, 0)

            non_empty_indices =  box.ne(0).any(dim=1).nonzero().view(-1)
            n_elems = non_empty_indices.nelement()

            # print('gt_max_overlaps :',gt_max_overlaps )
            # print('argmax_gt_overlaps :',argmax_gt_overlaps)
            # print('prob_out[argmax_gt_overlaps] :',prob_out[i,argmax_gt_overlaps])
            # print('cls_int[argmax_gt_overlaps] :',cls_int[argmax_gt_overlaps])
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
            
    recall    = float(true_pos)     /  (true_pos    + false_neg)
    recall_4  = float(true_pos_4)  / (true_pos_4  + false_neg_4)
    recall_3  = float(true_pos_3)  / (true_pos_3  + false_neg_3)

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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    dataset_folder = '../JHMDB-act-detector-frames'
    split_txt_path =  '../splits'
    boxes_file = '../poses.json'


    n_devs = torch.cuda.device_count()
    sample_size = 112
    # sample_duration = 4  # len(images)
    sample_duration = 8  # len(images)
    # sample_duration = 16  # len(images)

    batch_size = 1
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
    vid2idx,vid_names = get_vid_dict(dataset_folder)

    spatial_transform = Compose([Scale(sample_size),  # [Resize(sample_size),
                                 ToTensor(),
                                 Normalize(mean, [1, 1, 1])])
    temporal_transform = LoopPadding(sample_duration)

    # Init action_net
    # action_model_path = './action_net_model_16frm_max_jhmdb.pwf'
    action_model_path = './action_net_model_8frm_2_avg_jhmdb.pwf'
    # action_model_path = './action_net_model_8frm_64_jhmdb.pwf'    
    # action_model_path = './action_net_model_4frm_max_jhmdb.pwf'

    # linear_path = './linear_jhmdb.pwf'
    # linear_path = './linear_jhmdb_5.pwf'
    linear_path = './'

    model = Model(actions, sample_duration, sample_size)

    # model.load_part_model()
    model.load_part_model(action_model_path=action_model_path, rnn_path=linear_path)

    if torch.cuda.device_count() > 1:
        print('Using {} GPUs!'.format(torch.cuda.device_count()))
    model = nn.DataParallel(model)
    model.to(device)

    # model_path = './model_linear.pwf'
    # # model_path = './model.pwf'
    # model_data = torch.load(model_path)
    # model.load_state_dict(model_data)

    model.eval()
    
    validation(0, device, model, dataset_folder, sample_duration, spatial_transform, temporal_transform, boxes_file, split_txt_path, cls2idx, batch_size, n_threads)
