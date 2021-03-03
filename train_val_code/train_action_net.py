import os
import sys
import numpy as np

file_folder = os.path.dirname(os.path.realpath(__file__))
curr_dir = os.getcwd()

if file_folder==curr_dir:
    sys.path.append(os.path.realpath('../'))
elif os.path.dirname(file_folder) == curr_dir:
    sys.path.append('./')
else:
    raise('Please run this file from the root folder')

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

from config.dataset_config import  cfg as dataset_cfg, set_dataset

from lib.utils.spatial_transforms import (
    Compose, Normalize, Scale, ToTensor)
from lib.utils.temporal_transforms import LoopPadding

from lib.utils.create_video_id import get_vid_dict
from lib.utils.net_utils import adjust_learning_rate

from lib.models.action_net import ACT_net
import argparse
from lib.utils.box_functions import tube_overlaps, tube_transform_inv


np.random.seed(42)

        
def validation(epoch, device, model, data_loader,  n_threads):

    iou_thresh = 0.5 # Intersection Over Union thresh
    iou_thresh_4 = 0.4 # Intersection Over Union thresh
    iou_thresh_3 = 0.3 # Intersection Over Union thresh
    data = Video_Dataset(dataset_folder, frames_dur=sample_duration, spatial_transform=spatial_transform,
                         temporal_transform=temporal_transform, bboxes_file= boxes_file,
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
        batch_size = len(tubes)

        tubes = tubes.view(-1, sample_duration*4+2)
        tubes[:,1:-1] = tube_transform_inv(tubes[:,1:-1],\
                                           sgl_rois_bbox_pred.view(-1,sample_duration*4),(1.0,1.0,1.0,1.0))
        tubes = tubes.view(batch_size,-1, sample_duration*4+2)

        for i in range(tubes.size(0)): # how many frames we have
            
            tubes_t = tubes[i,:,1:-1].contiguous()
            gt_rois_t = gt_rois_[i,:,:,:4].contiguous().view(-1,sample_duration*4)
            rois_overlaps = tube_overlaps(tubes_t,gt_rois_t)
            
            gt_max_overlaps_sgl, _ = torch.max(rois_overlaps, 0)
            n_elems = gt_tubes_r[i,:,-1].ne(0).sum().item()

            # 0.5
            gt_max_overlaps_sgl_ = torch.where(gt_max_overlaps_sgl > iou_thresh, gt_max_overlaps_sgl, torch.zeros_like(gt_max_overlaps_sgl).type_as(gt_max_overlaps_sgl))
            # print('gt_max_overlaps_sgl_.shape :',gt_max_overlaps_sgl_.shape)
            # print('gt_max_overlaps_sgl_.shape :',gt_max_overlaps_sgl_)
            sgl_detected =  gt_max_overlaps_sgl_.ne(0).sum()
            sgl_true_pos += sgl_detected
            sgl_false_neg += n_elems - sgl_detected
            # print('sgl_detected :',sgl_detected)
            # print('sgl_detected :',sgl_true_pos)
            # print('sgl_detected :',sgl_false_neg)

            # 0.4
            gt_max_overlaps_sgl_ = torch.where(gt_max_overlaps_sgl > iou_thresh_4, gt_max_overlaps_sgl, torch.zeros_like(gt_max_overlaps_sgl).type_as(gt_max_overlaps_sgl))
            sgl_detected =  gt_max_overlaps_sgl_.ne(0).sum()
            sgl_true_pos_4 += sgl_detected
            sgl_false_neg_4 += n_elems - sgl_detected
            # print('sgl_detected :',sgl_detected)
            # print('sgl_detected :',sgl_true_pos)
            # print('sgl_detected :',sgl_false_neg)
            

            # 0.3
            gt_max_overlaps_sgl_ = torch.where(gt_max_overlaps_sgl > iou_thresh_3, gt_max_overlaps_sgl, torch.zeros_like(gt_max_overlaps_sgl).type_as(gt_max_overlaps_sgl))
            sgl_detected =  gt_max_overlaps_sgl_.ne(0).sum()
            sgl_true_pos_3 += sgl_detected
            sgl_false_neg_3 += n_elems - sgl_detected

    # print('sgl_true_pos :',sgl_true_pos)

    recall   = float(sgl_true_pos)  / (float(sgl_true_pos)  + float(sgl_false_neg))
    recall_4 = float(sgl_true_pos_4)  / (float(sgl_true_pos_4)  + float(sgl_false_neg_4))
    recall_3 = float(sgl_true_pos_3)  / (float(sgl_true_pos_3)  + float(sgl_false_neg_3))

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
        
def training(epoch, device, model, data_loader, mode = 1):

    model.train()
    loss_temp = 0
    

    for step, data  in enumerate(data_loader):

        if step == 2:
            break

        clips, h, w, gt_tubes_r, gt_rois, n_actions, n_frames, im_info = data.values()
        clips_ = clips.to(device)
        gt_tubes_r_ = gt_tubes_r.to(device)
        gt_rois_ = gt_rois.to(device)
        n_actions_ = n_actions.to(device)
        im_info_ = im_info.to(device)
        start_fr = torch.zeros(clips_.size(0)).to(device)
        
        inputs = Variable(clips_)
        print(f'clips_ {clips_.device}, {gt_rois_.device}')
        tubes, _, \
        rpn_loss_cls,  rpn_loss_bbox, \
        rpn_loss_cls_16,\
        rpn_loss_bbox_16,  rois_label, \
        sgl_rois_bbox_pred, sgl_rois_bbox_loss,  = model(inputs, \
                                                         im_info_,
                                                         gt_tubes_r_, gt_rois_,
                                                         start_fr)
        if mode == 1:
            loss = rpn_loss_cls.mean() + rpn_loss_bbox.mean() + act_loss_bbox.mean() + rpn_loss_cls_16.mean() \
                   + rpn_loss_bbox_16.mean() +  act_loss_bbox_16.mean() 
        elif mode == 3:
            loss = sgl_rois_bbox_loss.mean()

        elif mode == 4:
            loss = rpn_loss_cls.mean() +  rpn_loss_bbox.mean()  + sgl_rois_bbox_loss.mean()

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
    parser.add_argument('--demo', '-d', help='Run just 2 steps for test if everything works fine', action='store_true')
    parser.add_argument('--dataset', default='KTH', help='Choose dataset')
    parser.add_argument('--batch_size', '-b', default=32, help='batch_size')
    parser.add_argument('--epochs', '-e', default=40, help='batch_size')
    parser.add_argument('--frames_folder', '-f', default='./')
    parser.add_argument('--sample_size', default=112)
    parser.add_argument('--sample_dur', default=16)

    writer = SummaryWriter('./runs/action_net/KTH/training1')

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')

    print("Device being used:", device)

    args = parser.parse_args()
    dataset = args.dataset
    root_path = args.frames_folder

    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    sample_size = int(args.sample_size)
    sample_duration = int(args.sample_dur)

    data_root_2 = '/home/stath/thesis_code/sgal/' ## TODO remove after running to cloud
    model_path = os.path.abspath(os.path.join(data_root_2, 'resnet-34-kinetics.pth'))
    print(f'Model path {model_path}')
    print(f'Dataset : {dataset}, Batch size {batch_size}, Epochs {epochs}')
    set_dataset(dataset)

    if dataset.upper().startswith('UCF'):
        from lib.dataloaders.ucf_dataset import Video_Dataset
    elif dataset.upper().startswith('KTH'):
        from lib.dataloaders.kth_dataset import Video_Dataset
    else:
        raise('Unknown Dataset')


    dataset_frames = os.path.abspath(os.path.join(root_path,dataset_cfg.dataset.dataset_frames_folder))
    boxes_file = os.path.abspath(os.path.join(root_path,dataset_cfg.dataset.boxes_file))
    split_txt_path = os.path.abspath(os.path.join(root_path,dataset_cfg.dataset.split_txt_path))

    ### get videos id
    actions = dataset_cfg.dataset.classes
    cls2idx = {actions[i]: i for i in range(0, len(actions))}
    vid2idx,vid_names = get_vid_dict(dataset_frames)



    # # get mean
    # mean = [112.07945832, 112.87372333, 106.90993363]  # ucf-101 24 classes
    mean = [0.5,0.5,0.5]
    std  = [0.5,0.5,0.5]

    spatial_transform = Compose([Scale(sample_size),  # [Resize(sample_size),
                                 ToTensor(),
                                 Normalize(mean, std)])
    temporal_transform = LoopPadding(sample_duration)

    n_classes = len(actions)


    #######################################################
    #          Part 1-1 - train nTPN - without reg         #
    #######################################################

    print(' -----------------------------------------------------')
    print('|          Part 1-1 - train TPN - without reg         |')
    print(' -----------------------------------------------------')

    ## Define Dataloaders
    train_data = Video_Dataset(video_path=dataset_frames, frames_dur=sample_duration, spatial_transform=spatial_transform,
                               temporal_transform=temporal_transform, bboxes_file=boxes_file,
                               split_txt_path=split_txt_path, mode='train', classes_idx=cls2idx)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                                    shuffle=True, num_workers=2, pin_memory=True)

    # Init action_net
    act_model = ACT_net(actions, sample_duration, device=device)

    act_model.create_architecture(model_path=model_path)

    if torch.cuda.device_count() > 1:
        print('Using {} GPUs!'.format(torch.cuda.device_count()))
        act_model = nn.DataParallel(act_model)

    act_model.to(device)

    lr = 0.1
    lr_decay_step = 10
    lr_decay_gamma = 0.1
    
    params = []

    for key, value in dict(act_model.named_parameters()).items():
        if value.requires_grad:
            print('key :',key)
            if 'bias' in key:
                params += [{'params':[value],'lr':lr*(True + 1), \
                            'weight_decay': False and 0.0005 or 0}]
            else:
                params += [{'params':[value],'lr':lr, 'weight_decay': 0.0005}]

    lr = lr * 0.1
    optimizer = torch.optim.Adam(params)

    epochs = 40

    n_devs = torch.cuda.device_count()

    for epoch in range(epochs):
        print(' ============\n| Epoch {:0>2}/{:0>2} |\n ============'.format(epoch+1, epochs))

        if (epoch + 1) % (lr_decay_step ) == 0:
            adjust_learning_rate(optimizer, lr_decay_gamma)
            lr *= lr_decay_gamma
            print('adjust learning rate {}...'.format(lr))

        act_model, loss = training(epoch, device, act_model, train_data_loader, mode=4)

        if ( epoch + 1 ) % 5 == 0:
            torch.save(act_model.state_dict(), "action_net_model_steady_anchors_roi_align_ok.pwf".format(epoch+1))

        if ( epoch + 1 ) % 5 == 0:
            print(' ============\n| Validation {:0>2}/{:0>2} |\n ============'.format(epoch+1, epochs))
            validation(epoch+1, device, act_model, dataset_frames, sample_duration, spatial_transform, temporal_transform, boxes_file, split_txt_path, cls2idx, n_devs*4, 0)

    torch.save(act_model.state_dict(), "action_net_model_steady_anchors_roi_align_ok.pwf")
