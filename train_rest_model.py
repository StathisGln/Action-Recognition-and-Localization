import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from conf import conf
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, ToTensor, Resize)
from temporal_transforms import LoopPadding

from net_utils import adjust_learning_rate

from create_video_id import get_vid_dict
from ucf_dataset import Video_UCF, video_names

from model_par import Model
from resize_rpn import resize_tube
from mAP_function import calculate_mAP


def validation(epoch, device, model, dataset_folder, sample_duration, spatial_transform, temporal_transform, boxes_file, splt_txt_path, cls2idx, batch_size, n_threads):

    iou_thresh = 0.5 # Intersection Over Union thresh
    iou_thresh_4 = 0.4 # Intersection Over Union thresh
    iou_thresh_3 = 0.3 # Intersection Over Union thresh

    confidence_thresh = 0.1
    vid_name_loader = video_names(dataset_frames, split_txt_path, boxes_file, vid2idx, mode='test')
    data_loader = torch.utils.data.DataLoader(vid_name_loader, batch_size=n_devs,  pin_memory=True,
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

    temp_true_pos = 0
    temp_false_neg = 0

    temp_true_pos_8 = 0
    temp_false_neg_8 = 0

    temp_true_pos_7 = 0
    temp_false_neg_7 = 0
    
    correct_preds = torch.zeros(1).long().to(device)
    n_preds = torch.zeros(1).long().to(device)
    preds = torch.zeros(1).long().to(device)

    groundtruth_dic = {}
    detection_dic = {}

    ## 2 rois : 1450
    tubes_sum = 0
    for step, data  in enumerate(data_loader):

        if step == 25:
            break
        if step == 2:
            break
        
        print('step =>',step)

        vid_id, clips, boxes, n_frames, n_actions, h, w, target =data
        vid_id = vid_id.int()
        print( 'n_frames :',n_frames)
        clips = clips.to(device)
        boxes = boxes.to(device)
        n_frames = n_frames.to(device)
        n_actions = n_actions.int().to(device)

        im_info = torch.cat([h,w,torch.ones(clips.size(0)).long()*clips.size(2)]).to(device)
        mode = 'test'

        with torch.no_grad():
            tubes,  \
            prob_out,_ =  model(n_devs, dataset_frames, \
                                       vid_names, clips, vid_id,  \
                                       None, \
                                       mode, cls2idx, None, n_frames, h, w, None)

        print('prob_out.shape :',prob_out.shape)

        for i in range(clips.size(0)):
            
            ### first calculate probabilities
            _, cls_int = torch.max(prob_out[i],1)
            print('cls_int :',cls_int.shape)
            f_prob = torch.zeros(conf.CALC_THRESH).type_as(prob_out)
            print('f_prob.shape :',f_prob.shape)
            for j in range(conf.CALC_THRESH):
                f_prob[j] = prob_out[i,j,cls_int[j]]
            
            cls_int = cls_int[:conf.CALC_THRESH]
            
            keep_ = (f_prob.ge(confidence_thresh)) & cls_int.ne(0)
            keep_indices = keep_.nonzero().view(-1)

            f_tubes = torch.cat([cls_int.view(-1,1).type_as(tubes),f_prob.view(-1,1).type_as(tubes), \
                                 tubes[i,:conf.CALC_THRESH,:n_frames[i]].contiguous().view(-1,n_frames[i]*4)], dim=1)


            f_tubes = f_tubes[keep_indices].contiguous()

            
            # if f_tubes.nelement() != 0 :
            #     _, best_tube = torch.max(f_tubes[:,1],dim=0)
            #     f_tubes= f_tubes[best_tube].unsqueeze(0)
            # # print('f_tubes.shape :',f_tubes.shape)
            # # print('f_tubes.shape :',f_tubes[:, :2])


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

            ## proposal calculate
            
            box = boxes[i,:n_actions[i].int(), :n_frames[i].int(),:4].contiguous()

            # calculate temporal limits of gt tubes
            gt_limits = torch.zeros(n_actions[i].int(),2)
            t = torch.arange(n_frames[i].int().item()).unsqueeze(0).expand(n_actions[i].int().item(),n_frames[i].int().item()).type_as(box)
            z = box.eq(0).all(dim=2)

            gt_limits[:,0] = torch.min(t.masked_fill_(z,1000),dim=1)[0]
            gt_limits[:,1] = torch.max(t.masked_fill_(z,-1),dim=1)[0]+1

            box = box.view(-1,n_frames[i].int()*4)
            tubes_t = tubes[i,:conf.CALC_THRESH,:n_frames[i].long()]

            # calculate temporal limits of tubes
            tub_limits = torch.zeros(conf.CALC_THRESH,2)
            t = torch.arange(n_frames[i].int().item()).unsqueeze(0).expand(conf.CALC_THRESH,n_frames[i].int().item()).type_as(tubes_t)
            z = tubes_t.eq(0).all(dim=2)

            tub_limits[:,0] = torch.min(t.masked_fill_(z,1000),dim=1)[0]
            tub_limits[:,1] = torch.max(t.masked_fill_(z,-1),dim=1)[0]+1

            non_empty_indices =  box.ne(0).any(dim=1).nonzero().view(-1)
            n_elems = non_empty_indices.nelement()            

            tubes_t = tubes_t.view(-1, n_frames[i].int()*4)

            ## temporal overlaps
            temp_overlaps = bbox_temporal_overlaps(tub_limits, gt_limits)

            gt_temp_max_overlaps, argmax_temp_gt_overlaps = torch.max(temp_overlaps, 0)

            print('gt_temp_max_overlaps :',gt_temp_max_overlaps)
            print('gt_limits :',gt_limits)
            print('tub_limits[argmax_temp_gt_overlaps] :',tub_limits[argmax_temp_gt_overlaps])

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
    
    print('&&&&&&&&&&&&&&&&&&&&')
    
if __name__ == '__main__':
    
    #################################
    #        UCF data inits         #
    #################################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_devs = torch.cuda.device_count()
    print("Device being used:", device)

    dataset_frames = '../UCF-101-frames'
    boxes_file = '../pyannot.pkl'
    split_txt_path = '../UCF101_Action_detection_splits/'

    sample_size = 112
    # sample_duration = 16  # len(images)
    sample_duration = 8  # len(images)
    batch_size = n_devs

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
    vid2idx,vid_names = get_vid_dict(dataset_frames)

    spatial_transform = Compose([Scale(sample_size),  # [Resize(sample_size),
                                 ToTensor(),
                                 Normalize(mean, [1, 1, 1])])
    temporal_transform = LoopPadding(sample_duration)

    n_classes = len(actions)


    ##########################################
    #          Model Initialization          #
    ##########################################
    action_model_path = './action_net_model_8frm_conf_ucf.pwf'

    model = Model(actions, sample_duration, sample_size)
    model.load_part_model(action_model_path=action_model_path)
    model.deactivate_action_net_grad()

    # if torch.cuda.device_count() > 1:

    #     print('Using {} GPUs!'.format(torch.cuda.device_count()))

    model = nn.DataParallel(model)
    model.to(device)

    lr = 0.1
    lr_decay_step = 15
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

    batch_size = n_devs
    vid_name_loader = video_names(dataset_frames, split_txt_path, boxes_file, vid2idx, mode='train')
    data_loader = torch.utils.data.DataLoader(vid_name_loader, batch_size=batch_size,
                                              shuffle=True, num_workers=2, pin_memory=True)

    epochs = 60
    # epochs = 5

    n_devs = torch.cuda.device_count()

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

            # if step == 2:
            #     exit(-1)
            #     break

            print('step :',step)
            vid_id, clips, boxes, n_frames, n_actions, h, w, target =data
            print('vid_id :',vid_id)
            print('n_frames :',n_frames)
            vid_id = vid_id.to(device)
            clips = clips.to(device)
            boxes  = boxes.to(device)
            n_frames = n_frames.to(device)
            n_actions = n_actions.int().to(device)
            im_info = torch.stack([h,w],dim=1).to(device)
            mode = 'train'

            tubes,   \
            prob_out, cls_loss =  model(n_devs, dataset_frames, \
                                        vid_names, clips, vid_id, \
                                        boxes, \
                                        mode, cls2idx, n_actions,n_frames, h, w,target)

            loss =  cls_loss.mean()

            # backw\ard
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_temp += loss.item()

        print('Train Epoch: {} \tLoss: {:.6f}\t lr : {:.6f}'.format(
        ep+1,loss_temp/(step+1), lr))

        if ( ep + 1 ) % 1 == 0:
            torch.save(model.state_dict(), "model_epoch_{}.pwf".format(ep+1))

        # if ( ep + 1 ) % 5 == 0: # validation time

        #     validation(0, device, model, dataset_frames, sample_duration, spatial_transform, temporal_transform,\
        #                    boxes_file, split_txt_path, cls2idx, n_devs, 0)

    torch.save(model.state_dict(), "model.pwf")


