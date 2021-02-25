import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from lib.dataloaders.ucf_dataset import Video_UCF

from lib.utils.spatial_transforms import (
    Compose, Normalize, Scale, ToTensor)
from lib.utils.temporal_transforms import LoopPadding

from lib.utils.create_video_id import get_vid_dict
from lib.utils.net_utils import adjust_learning_rate

from lib.models.action_net import ACT_net
import argparse
from lib.utils.box_functions import tube_overlaps, tube_transform_inv, clip_boxes

np.random.seed(42)

def validation(epoch, device, model, dataset_folder, sample_duration, spatial_transform, temporal_transform, boxes_file, splt_txt_path, cls2idx, batch_size, n_threads):

    iou_thresh = 0.5 # Intersection Over Union thresh
    iou_thresh_4 = 0.4 # Intersection Over Union thresh
    iou_thresh_3 = 0.3 # Intersection Over Union thresh
    data = Video_UCF(dataset_folder, frames_dur=sample_duration, spatial_transform=spatial_transform,
                 temporal_transform=temporal_transform, json_file = boxes_file,
                 split_txt_path=splt_txt_path, mode='test', classes_idx=cls2idx)
    data_loader = torch.utils.data.DataLoader(data, batch_size=4,
                                              shuffle=True, num_workers=0, pin_memory=True)
    model.eval()

    sgl_true_pos = 0
    sgl_false_neg = 0

    sgl_true_pos_4 = 0
    sgl_false_neg_4 = 0

    sgl_true_pos_3 = 0
    sgl_false_neg_3 = 0

    ## 2 rois : 1450
    tubes_sum = 0
    for step, data  in enumerate(data_loader):

        # if step == 2:
        #     break
        # print('step :',step)

        clips, h, w, gt_tubes_r, gt_rois, n_actions, n_frames, im_info = data
        clips_   = clips.to(device)
        gt_tubes_r_ = gt_tubes_r.to(device)
        gt_rois_ = gt_rois.to(device)
        n_actions_ = n_actions.to(device)
        im_info_ = im_info.to(device)
        start_fr = torch.zeros(clips_.size(0)).to(device)

        tubes, _, _, _, _, _, _, \
        sgl_rois_bbox_pred, _  = model(clips,
                                       im_info,
                                       None, None,
                                       None)
        n_tubes = len(tubes)

        tubes = tubes.view(-1, sample_duration*4+2)
        tubes[:,1:-1] = tube_transform_inv(tubes[:,1:-1],\
                                           sgl_rois_bbox_pred.view(-1,sample_duration*4),(1.0,1.0,1.0,1.0))
        tubes = tubes.view(n_tubes,-1, sample_duration*4+2)
        tubes[:,:,1:-1] = clip_boxes(tubes[:,:,1:-1], im_info, tubes.size(0))

        for i in range(tubes.size(0)): # how many frames we have
            
            tubes_t = tubes[i,:,1:-1].contiguous()
            gt_rois_t = gt_rois_[i,:,:,:4].contiguous().view(-1,sample_duration*4)
            rois_overlaps = tube_overlaps(tubes_t,gt_rois_t)
            
            gt_max_overlaps_sgl, _ = torch.max(rois_overlaps, 0)

            non_empty_indices =  gt_rois_t.ne(0).any(dim=1).nonzero().view(-1)
            n_elems = non_empty_indices.nelement()            

            # 0.5
            gt_max_overlaps_sgl_ = torch.where(gt_max_overlaps_sgl > iou_thresh, gt_max_overlaps_sgl, torch.zeros_like(gt_max_overlaps_sgl).type_as(gt_max_overlaps_sgl))

            sgl_detected =  gt_max_overlaps_sgl_.ne(0).sum()
            sgl_true_pos += sgl_detected
            sgl_false_neg += n_elems - sgl_detected

            # 0.4
            gt_max_overlaps_sgl_ = torch.where(gt_max_overlaps_sgl > iou_thresh_4, gt_max_overlaps_sgl, torch.zeros_like(gt_max_overlaps_sgl).type_as(gt_max_overlaps_sgl))
            sgl_detected =  gt_max_overlaps_sgl_.ne(0).sum()
            sgl_true_pos_4 += sgl_detected
            sgl_false_neg_4 += n_elems - sgl_detected

            # 0.3
            gt_max_overlaps_sgl_ = torch.where(gt_max_overlaps_sgl > iou_thresh_3, gt_max_overlaps_sgl, torch.zeros_like(gt_max_overlaps_sgl).type_as(gt_max_overlaps_sgl))
            sgl_detected =  gt_max_overlaps_sgl_.ne(0).sum()
            sgl_true_pos_3 += sgl_detected
            sgl_false_neg_3 += n_elems - sgl_detected

    recall   = float(sgl_true_pos)  / (float(sgl_true_pos)  + float(sgl_false_neg))
    recall_4 = float(sgl_true_pos_4)  / (float(sgl_true_pos_4)  + float(sgl_false_neg_4))
    recall_3 = float(sgl_true_pos_3)  / (float(sgl_true_pos_3)  + float(sgl_false_neg_3))

    f = open('../images_etc/recall_ucf.txt', 'a')
    f.write('| Validation Epoch: {: >3} |\n'.format(epoch+1))
    f.write('| Threshold : 0.5       |\n')
    f.write('| True_pos   --> {: >6} |\n| False_neg  --> {: >6} | \n| Recall     --> {: >6.4f} |\n'.format(
        sgl_true_pos, sgl_false_neg, recall))
    f.write('| Threshold : 0.4       |\n')
    f.write('| True_pos   --> {: >6} |\n| False_neg  --> {: >6} | \n| Recall     --> {: >6.4f} |\n'.format(
        sgl_true_pos_4, sgl_false_neg_4, recall_4))
    f.write('| Threshold : 0.3       |\n')
    f.write('| True_pos   --> {: >6} |\n| False_neg  --> {: >6} | \n| Recall     --> {: >6.4f} |\n'.format(
        sgl_true_pos_3, sgl_false_neg_3, recall_3))
    f.close()

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
        sgl_true_pos, sgl_false_neg, recall))
    print('|                       |')
    print('| Threshold : 0.4       |')
    print('|                       |')
    print('| True_pos   --> {: >6} |\n| False_neg  --> {: >6} | \n| Recall     --> {: >6.4f} |'.format(
        sgl_true_pos_4, sgl_false_neg_4, recall_4))
    print('|                       |')
    print('| Threshold : 0.3       |')
    print('|                       |')
    print('| True_pos   --> {: >6} |\n| False_neg  --> {: >6} | \n| Recall     --> {: >6.4f} |'.format(
        sgl_true_pos_3, sgl_false_neg_3, recall_3))


    print(' -----------------------')
        
def training(epoch, device, model, dataset_folder, sample_duration, spatial_transform, temporal_transform, boxes_file, splt_txt_path, cls2idx, batch_size, n_threads, lr, mode = 1):

    data = Video_UCF(dataset_folder, frames_dur=sample_duration, spatial_transform=spatial_transform,
                 temporal_transform=temporal_transform, json_file = boxes_file,
                 split_txt_path=splt_txt_path, mode='train', classes_idx=cls2idx)

    # data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size*16,
    #                                           shuffle=True, num_workers=32, pin_memory=True)
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size*8,
                                              shuffle=True, num_workers=32, pin_memory=True)

    # data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size*4,
    #                                           shuffle=True, num_workers=32, pin_memory=True)

    # data_loader = torch.utils.data.DataLoader(data, batch_size=2,
    #                                           shuffle=True, num_workers=0, pin_memory=True)

    model.train()
    loss_temp = 0
    
    ## 2 rois : 1450
    for step, data  in enumerate(data_loader):

        # if step == 2:
        #     exit(-1)
        #     break

        clips, h, w, gt_tubes_r, gt_rois, n_actions, n_frames, im_info = data
        clips_ = clips.to(device)
        gt_tubes_r_ = gt_tubes_r.to(device)
        gt_rois_ = gt_rois.to(device)
        n_actions_ = n_actions.to(device)
        im_info_ = im_info.to(device)
        start_fr = torch.zeros(clips_.size(0)).to(device)
        
        inputs = Variable(clips_)

        tubes, _, \
        rpn_loss_cls,  rpn_loss_bbox, \
        rpn_loss_cls_16,\
        rpn_loss_bbox_16,  rois_label, \
        sgl_rois_bbox_pred, sgl_rois_bbox_loss,  = model(inputs, \
                                                         im_info_,
                                                         gt_tubes_r_, gt_rois_,
                                                         start_fr)
        if mode == 3:
            loss = sgl_rois_bbox_loss.mean()
        elif mode == 4:
            loss = rpn_loss_cls.mean() +  rpn_loss_bbox.mean() 
        elif mode == 5:
            loss = rpn_loss_cls.mean() +  rpn_loss_bbox.mean() + sgl_rois_bbox_loss.mean()


        loss_temp += loss.item()


        # # backw\ard
        # if step % 4 == 0:
        #     optimizer.zero_grad()

        # loss.backward()

        # if step % 4 == 0:
        #     optimizer.step()


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

    dataset_frames = '../UCF-101-frames'
    boxes_file = '../pyannot.pkl'
    split_txt_path = '../UCF101_Action_detection_splits/'

    sample_size = 112
    # sample_duration = 4  # len(images)
    sample_duration = 8  # len(images)
    # sample_duration = 16  # len(images)

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


    #######################################################
    #          Part 1-1 - train nTPN - without reg         #
    #######################################################

    print(' -----------------------------------------------------')
    print('|          Part 1-1 - train TPN - without reg         |')
    print(' -----------------------------------------------------')

    # Init action_net
    act_model = ACT_net(actions, sample_duration)
    act_model.create_architecture()
    if torch.cuda.device_count() > 1:
        print('Using {} GPUs!'.format(torch.cuda.device_count()))

    act_model = nn.DataParallel(act_model)

    act_model.to(device)

    lr = 0.1
    lr_decay_step = 10
    lr_decay_gamma = 0.1
    
    params = []

    # for p in act_model.module.reg_layer.parameters() : p.requires_grad=False

    for key, value in dict(act_model.named_parameters()).items():
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

    epochs = 60
    # epochs = 5

    n_devs = torch.cuda.device_count()
    for epoch in range(epochs):
        print(' ============\n| Epoch {:0>2}/{:0>2} |\n ============'.format(epoch+1, epochs))

        if epoch % (lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, lr_decay_gamma)
            lr *= lr_decay_gamma


        # act_model, loss = training(epoch, device, act_model, dataset_frames, sample_duration, spatial_transform, temporal_transform, boxes_file, split_txt_path, cls2idx, n_devs*4, 0, lr, mode=4)
        act_model, loss = training(epoch, device, act_model, dataset_frames, sample_duration, spatial_transform, temporal_transform, boxes_file, split_txt_path, cls2idx, n_devs*4, 0, lr, mode=5)

        if ( epoch + 1 ) % 5 == 0:
            torch.save(act_model.state_dict(), "action_net_model_{}frm_256_7_RoiAlignAvg_ucf.pwf".format(sample_duration))

        if (epoch + 1) % (5) == 0:
            print(' ============\n| Validation {:0>2}/{:0>2} |\n ============'.format(epoch+1, epochs))
            validation(epoch, device, act_model, dataset_frames, sample_duration, spatial_transform, temporal_transform, boxes_file, split_txt_path, cls2idx, n_devs, 0)

    torch.save(act_model.state_dict(), "action_net_model_{}frm_256_7_RoiAlignAvg_ucf.pwf".format(sample_duration))

