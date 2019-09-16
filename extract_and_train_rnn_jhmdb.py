import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from jhmdb_dataset import  video_names, RNN_JHMDB

from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, ToTensor, Resize)
from temporal_transforms import LoopPadding, LoopPadding_still

from create_video_id import get_vid_dict
from net_utils import adjust_learning_rate
from resize_rpn import resize_rpn, resize_tube

# from model import Model
from model_all import Model
from action_net import ACT_net
from resize_rpn import resize_boxes
import argparse
from box_functions import tube_overlaps, tube_transform_inv
from act_rnn_wrapper import _RNN_wrapper

np.random.seed(42)

def validation(epoch, device, model, dataset_folder, sample_duration, spatial_transform, temporal_transform, boxes_file, splt_txt_path, cls2idx, batch_size, n_threads):

    iou_thresh = 0.5 # Intersection Over Union thresh
    iou_thresh_4 = 0.4 # Intersection Over Union thresh
    iou_thresh_3 = 0.3 # Intersection Over Union thresh

    vid_name_loader = video_names(dataset_folder, split_txt_path, boxes_file, vid2idx, mode='test', classes_idx= cls2idx)
    data_loader = torch.utils.data.DataLoader(vid_name_loader, batch_size=n_devs, num_workers=2*n_devs, pin_memory=True,
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

    # for precision
    preds = 0
    tp = 0
    tp_4 = 0
    tp_3 = 0
    fp = 0
    fp_4 = 0
    fp_3 = 0
    fn = 0
    fn_4 = 0
    fn_3 = 0
    
    ## 2 rois : 1450
    tubes_sum = 0
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
            prob_out, tubes_size =  model(n_devs, dataset_folder, \
                                    vid_names, clips, vid_id,  \
                                    None, \
                                    mode, cls2idx, None, n_frames, h, w)

        

        # print('tubes.shape :',tubes.dim())
        # print('prob_out.shape :',prob_out.shape)
        # print('clips.size(0) :',clips.size(0))
        # print('clips.size(0) :',clips.shape)
        # exit(-1)
        if tubes.dim() == 1:
        
            for i in range(clips.size(0)):

                box = boxes[i,:n_actions, :n_frames,:4].contiguous()
                box = box.view(-1,n_frames*4)

                non_empty_indices =  box.ne(0).any(dim=1).nonzero().view(-1)
                n_elems = non_empty_indices.nelement()            
                false_neg += n_elems
                false_neg_4 += n_elems
                false_neg_3 += n_elems 
            continue

        prob_out = F.softmax(prob_out,2)
        _, predictions = torch.max(prob_out,dim=2)

        for i in range(clips.size(0)):

            targ = target[i].expand(n_actions[i])
            
            tubes_ = tubes[i,:tubes_size[i].int().item(),:n_frames[i].int().item()]
            preds_ = predictions[i,:tubes_size[i].int().item()]

            box = boxes[i,:n_actions[i], :n_frames[i],:4].contiguous()
            box = box.view(-1,n_frames[i]*4).contiguous().type_as(tubes)
            overlaps = tube_overlaps(tubes_.view(-1,n_frames[i]*4).float(), box.view(-1,n_frames[i]*4).float())
            gt_max_overlaps, argmax_gt_overlaps = torch.max(overlaps, 0)
            max_overlaps, argmax_overlaps = torch.max(overlaps, 1)
            # print('max_overlaps :',max_overlaps.shape)
            # print('gt_max_overlaps.shape :',gt_max_overlaps.shape)
            # print('targ :',targ)
            non_empty_indices =  box.ne(0).any(dim=1).nonzero().view(-1)
            n_elems = non_empty_indices.nelement()
            preds += n_elems
            tubes_sum += tubes.size(0)

            # 0.5 thresh
            gt_max_overlaps_ = torch.where(gt_max_overlaps > iou_thresh, gt_max_overlaps, torch.zeros_like(gt_max_overlaps).type_as(gt_max_overlaps))
            detected =  gt_max_overlaps_[non_empty_indices].ne(0).sum().item()
            true_pos += detected
            false_neg += n_elems - detected

            max_overlaps_ =  torch.where(max_overlaps > iou_thresh, max_overlaps, torch.zeros_like(max_overlaps).type_as(max_overlaps)) 
            non_zero = max_overlaps_.nonzero().view(-1)
            bg_idx = max_overlaps_.eq(0).nonzero().view(-1)
            fn += preds_[bg_idx].ne(0).sum().item() # add to false negative all non-background tubes with no gt tube overlaping
            predictions_ = preds_[non_zero] # overlaping predictions
            # print('target :',target)
            # print('target.shape :',target.shape)
            # print('targ :',targ)
            # print('predictions_.shape :',predictions_.shape)
            # print('non_zero :',non_zero.cpu().numpy())
            # print('non_zero.shape :',non_zero.shape)
            # print('argmax_overlaps.shape :',argmax_overlaps.shape)
            # print('argmax_overlaps :',argmax_overlaps.cpu().numpy())
            # print('argmax_overlaps[non_zero].shape :',argmax_overlaps[non_zero].shape)
            # print('argmax_overlaps[non_zero] :',argmax_overlaps[non_zero])
            # print('targ :',targ)
            fn += (predictions_== targ[argmax_overlaps[non_zero]]).eq(0).sum()

            predictions_ = predictions_[(predictions_== targ[argmax_overlaps[non_zero]]).ne(0).nonzero().view(-1)]
            unique_labels =torch.unique(predictions_) # unique labels
            for i in unique_labels:
                fp += predictions_.eq(i).sum().item() -1
                tp += 1

            # 0.4 thresh
            gt_max_overlaps_ = torch.where(gt_max_overlaps > iou_thresh_4, gt_max_overlaps, torch.zeros_like(gt_max_overlaps).type_as(gt_max_overlaps))
            detected =  gt_max_overlaps_[non_empty_indices].ne(0).sum().item()
            true_pos_4 += detected
            false_neg_4 += n_elems - detected

            max_overlaps_ =  torch.where(max_overlaps > iou_thresh_4, max_overlaps, torch.zeros_like(max_overlaps).type_as(max_overlaps)) 
            non_zero = max_overlaps_.nonzero().view(-1)
            bg_idx = max_overlaps_.eq(0).nonzero().view(-1)
            fn_4 += preds_[bg_idx].ne(0).sum().item() # add to false negative all non-background tubes with no gt tube overlaping
            predictions_ = preds_[non_zero] # overlaping predictions

            fn_4 += (predictions_== targ[argmax_overlaps[non_zero]]).eq(0).sum()

            predictions_ = predictions_[(predictions_== targ[argmax_overlaps[non_zero]]).ne(0).nonzero().view(-1)]
            unique_labels =torch.unique(predictions_) # unique labels
            for i in unique_labels:
                fp_4 += predictions_.eq(i).sum().item() -1
                tp_4 += 1

            # 0.3 thresh
            gt_max_overlaps_ = torch.where(gt_max_overlaps > iou_thresh_3, gt_max_overlaps, torch.zeros_like(gt_max_overlaps).type_as(gt_max_overlaps))

            detected =  gt_max_overlaps_[non_empty_indices].ne(0).sum().item()
            true_pos_3 += detected
            false_neg_3 += n_elems - detected

            max_overlaps_ =  torch.where(max_overlaps > iou_thresh_3, max_overlaps, torch.zeros_like(max_overlaps).type_as(max_overlaps)) 
            non_zero = max_overlaps_.nonzero().view(-1)
            bg_idx = max_overlaps_.eq(0).nonzero().view(-1)
            fn_3 += preds_[bg_idx].ne(0).sum().item() # add to false negative all non-background tubes with no gt tube overlaping
            predictions_ = preds_[non_zero] # overlaping predictions

            fn_3 += (predictions_== targ[argmax_overlaps[non_zero]]).eq(0).sum()

            predictions_ = predictions_[(predictions_== targ[argmax_overlaps[non_zero]]).ne(0).nonzero().view(-1)]
            unique_labels =torch.unique(predictions_) # unique labels
            for i in unique_labels:
                fp_3 += predictions_.eq(i).sum().item() -1
                tp_3 += 1

    recall    = float(true_pos)     /  (true_pos    + false_neg)  if true_pos > 0 or false_neg > 0 else 0
    recall_4  = float(true_pos_4)  / (true_pos_4  + false_neg_4)  if true_pos_4 > 0 or false_neg_4 > 0 else 0
    recall_3  = float(true_pos_3)  / (true_pos_3  + false_neg_3)  if true_pos_3 > 0 or false_neg_3 > 0 else 0

    precision   = float(tp)   / (tp   + fp  ) if tp > 0 or fp > 0 else 0
    precision_4 = float(tp_4) / (tp_4 + fp_4) if tp_4 > 0 or fp_4 > 0 else 0
    precision_3 = float(tp_3) / (tp_3 + fp_3) if tp_3 > 0 or fp_3 > 0 else 0
 
    print(' -----------------------\n')
    print('| Validation Epoch: {: >3} |\n'.format(epoch+1))
    print('|                       |')
    print('| we have {: >6} tubes  |'.format(tubes_sum))
    print('|                       |')
    print('| Proposed Action Tubes |')
    print('|                       |')
    print('| Single frame          |')
    print('|                       |')
    print('| In {: >6} steps    :  |'.format(step))
    print('|                       |')
    print('| Precision             |')
    print('|                       |')
    print('| Threshold : 0.5       |')
    print('|                       |')
    print('| True_pos   --> {: >6} |\n| False_pos  --> {: >6} |\n| False_neg  --> {: >6} | \n| Precision  --> {: >6.4f} |'.format(
        tp, fp, fn, precision))
    print('|                       |')
    print('| Threshold : 0.4       |')
    print('|                       |')
    print('| True_pos   --> {: >6} |\n| False_pos  --> {: >6} |\n| False_neg  --> {: >6} | \n| Precision  --> {: >6.4f} |'.format(
        tp_4, fp_4, fn_4, precision_4))
    print('|                       |')
    print('| Threshold : 0.3       |')
    print('|                       |')
    print('| True_pos   --> {: >6} |\n| False_pos  --> {: >6} |\n| False_neg  --> {: >6} | \n| Precision  --> {: >6.4f} |'.format(
        tp_3, fp_3, fn_3, precision_3))
    print('|                       |')
    print('| Recall                |')
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

    file_p = open('validation_jhmdb_linear.txt', 'a')
    file_p.write(' -----------------------\n')
    file_p.write('| Validation Epoch: {: >3} | '.format(epoch+1))
    file_p.write('|                       |\n')
    file_p.write('| we have {: >6} tubes  |\n'.format(tubes_sum))
    file_p.write('|                       |\n')
    file_p.write('| Proposed Action Tubes |\n')
    file_p.write('|                       |\n')
    file_p.write('| Single frame          |\n')
    file_p.write('|                       |\n')
    file_p.write('| In {: >6} steps    :  |\n'.format(step))
    file_p.write('|                       |\n')
    file_p.write('| Precision             |\n')
    file_p.write('|                       |\n')
    file_p.write('| Threshold : 0.5       |\n')
    file_p.write('|                       |\n')
    file_p.write('| True_pos   --> {: >6} |\n| False_pos  --> {: >6} |\n| False_neg  --> {: >6} | \n| Precision  --> {: >6.4f} |\n'.format(
        tp, fp, fn, precision))
    file_p.write('|                       |\n')
    file_p.write('| Threshold : 0.4       |\n')
    file_p.write('|                       |\n')
    file_p.write('| True_pos   --> {: >6} |\n| False_pos  --> {: >6} |\n| False_neg  --> {: >6} | \n| Precision  --> {: >6.4f} |\n'.format(
        tp_4, fp_4, fn_4, precision_4))
    file_p.write('|                       |\n')
    file_p.write('| Threshold : 0.3       |\n')
    file_p.write('|                       |\n')
    file_p.write('| True_pos   --> {: >6} |\n| False_pos  --> {: >6} |\n| False_neg  --> {: >6} | \n| Precision  --> {: >6.4f} |\n'.format(
        tp_3, fp_3, fn_3, precision_3))
    file_p.write('|                       |\n')
    file_p.write('| Recall                |\n')
    file_p.write('|                       |\n')
    file_p.write('| Threshold : 0.5       |\n')
    file_p.write('|                       |\n')
    file_p.write('| True_pos   --> {: >6} |\n| False_neg  --> {: >6} | \n| Recall     --> {: >6.4f} |\n'.format(
        true_pos, false_neg, recall))
    file_p.write('|                       |\n')
    file_p.write('| Threshold : 0.4       |\n')
    file_p.write('|                       |\n')
    file_p.write('| True_pos   --> {: >6} |\n| False_neg  --> {: >6} | \n| Recall     --> {: >6.4f} |\n'.format(
        true_pos_4, false_neg_4, recall_4))
    file_p.write('|                       |\n')
    file_p.write('| Threshold : 0.3       |\n')
    file_p.write('|                       |\n')
    file_p.write('| True_pos   --> {: >6} |\n| False_neg  --> {: >6} | \n| Recall     --> {: >6.4f} |\n'.format(
        true_pos_3, false_neg_3, recall_3))


    file_p.write(' -----------------------')
        
def training(epoch, device, model, dataset_folder, sample_duration, spatial_transform, temporal_transform, boxes_file, splt_txt_path, cls2idx, batch_size, n_threads, lr, mode = 1):

    vid_name_loader = RNN_JHMDB(dataset_folder, split_txt_path, boxes_file, vid2idx, mode='train', sample_duration=sample_duration)
    data_loader = torch.utils.data.DataLoader(vid_name_loader, batch_size=batch_size,
                                              shuffle=True, num_workers=32, pin_memory=True)
    # data_loader = torch.utils.data.DataLoader(data, batch_size=2,
    #                                           shuffle=True, num_workers=0, pin_memory=True)

    model.train()
    loss_temp = 0
    
    ## 2 rois : 1450
    for step, data  in enumerate(data_loader):

        # if step == 2:
        #     break
        # print('step =>',step)

        f_features, n_tubes, target_lbl  = data        

        f_features = f_features.to(device)
        n_tubes = n_tubes.to(device)
        target_lbl = target_lbl.to(device)

        cls_loss = model(f_features, n_tubes, target_lbl)

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

    ###################################################
    #          Part 1-1 - extract features -          #
    ###################################################

    print(' ----------------------------------------')
    print('|          - extract features -          |')
    print(' ----------------------------------------')

    # action_model_path = './action_net_model_both_single_frm_jhmdb.pwf'

    # action_model_path = './action_net_model_8frm_64_jhmdb.pwf'
    action_model_path = './action_net_model_8frm_2_avg_jhmdb.pwf'

    # Init whole model
    model = Model(actions, sample_duration, sample_size)
    model.load_part_model(action_model_path=action_model_path)

    if torch.cuda.device_count() > 1:
        print('Using {} GPUs!'.format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    model.to(device)

    batch_size = 1
    vid_name_loader = video_names(dataset_frames, split_txt_path, boxes_file, vid2idx, mode='train', classes_idx=cls2idx)
    data_loader = torch.utils.data.DataLoader(vid_name_loader, batch_size=batch_size,
                                              shuffle=True, num_workers=32, pin_memory=True)
    # out_path = 'JHMDB-features-256-mod7'
    out_path = 'JHMDB-features-256-7ver3'

    mode = 'extract'
    for step, data  in enumerate(data_loader):

        # if step == 2:
        #     break
        print('step =>',step)
        # clips, h, w, gt_tubes_r, gt_rois, n_actions, n_frames, im_info = data
        vid_id, clips, boxes, n_frames, n_actions, h, w, target =data
        print('vid_names[vid_id] :',vid_names[vid_id])
        
        print('n_frames :',n_frames)
        if not(os.path.exists(os.path.join(out_path, vid_names[vid_id]))):
            os.makedirs(os.path.join(out_path, vid_names[vid_id]))

        vid_id = vid_id.to(device)
        clips_ = clips.to(device)
        boxes  = boxes.to(device)
        n_frames = n_frames.to(device)
        n_actions = n_actions.int().to(device)
        im_info = torch.stack([h,w],dim=1).to(device)

        inputs = Variable(clips_)

        feats, tubes, labels, tube_len =  model(n_devs, dataset_frames, \
                                    vid_names, clips_, vid_id,  \
                                    boxes, \
                                    mode, cls2idx, n_actions, n_frames, h, w)
        print('feats.shape :',feats.shape)
        print('labels :',labels)
        print('tube_len :',tube_len)
        print('tubes :',tubes.shape)

        torch.save(feats, os.path.join(out_path, vid_names[vid_id], 'feats.pt'))
        torch.save(labels, os.path.join(out_path, vid_names[vid_id], 'labels.pt'))
        torch.save(tube_len, os.path.join(out_path, vid_names[vid_id], 'tube_len.pt'))
        torch.save(tubes, os.path.join(out_path, vid_names[vid_id], 'tubes.pt'))

    exit(-1)
    print(' ---------------------------------')
    print('|          - train RNN -          |')
    print(' ---------------------------------')

    dataset_frames = '../JHMDB-act-detector-frames'
    dataset_features = 'JHMDB-features-linear'

    lr = 0.1
    lr_decay_step = 50
    lr_decay_gamma = 0.1
    
    params = []

    model =_RNN_wrapper(64*sample_duration,128,len(actions), 64, sample_duration)
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

        if ( epoch + 1 ) % 20 == 0:
            torch.save(model.module.act_rnn.state_dict(), "linear_jhmdb_{}.pwf".format(epoch+1))

    torch.save(model.module.act_rnn.state_dict(), "linear_jhmdb_{}.pwf".format(epoch+1))
    # print(' ============\n| Validation {:0>2}/{:0>2} |\n ============'.format(epoch+1, epochs))
    # # # Init whole model
    # action_model_path = './action_net_model_jhmdb_16frm_64.pwf'
    # model = Model(actions, sample_duration, sample_size)
    # model.load_part_model(action_model_path=action_model_path, rnn_path= './linear_jhmdb.pwf')
    
    # model = nn.DataParallel(model)
    # model = model.to(device)

    
    # validation(epoch, device, model, dataset_frames, sample_duration, spatial_transform, temporal_transform, boxes_file, split_txt_path, cls2idx, n_devs, 0)


    torch.save(model.module.act_rnn.state_dict(), "model_with_linear_jhmdb.pwf".format(epoch+1))






