import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from ucf_dataset import Video_UCF

from lib.utils.spatial_transforms import (
    Compose, Normalize, Scale, ToTensor)
from lib.utils.temporal_transforms import LoopPadding

from create_video_id import get_vid_dict
from net_utils import adjust_learning_rate

from action_net import ACT_net
import argparse

np.random.seed(42)

        
def training(epoch, device, model, dataset_folder, sample_duration, spatial_transform, temporal_transform, boxes_file, splt_txt_path, cls2idx, batch_size, n_threads, lr, mode = 1):

    data = Video_UCF(dataset_folder, frames_dur=sample_duration, spatial_transform=spatial_transform,
                 temporal_transform=temporal_transform, json_file = boxes_file,
                 split_txt_path=splt_txt_path, mode='train', classes_idx=cls2idx)
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size*8,
                                              shuffle=True, num_workers=32, pin_memory=True)
    # data_loader = torch.utils.data.DataLoader(data, batch_size=4,
    #                                           shuffle=True, num_workers=2, pin_memory=True)

    model.train()
    loss_temp = 0
    
    ## 2 rois : 1450
    for step, data  in enumerate(data_loader):

        # if step == 5:
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
        # print('rpn_loss_cls :',rpn_loss_cls)
        # print('rpn_loss_bbox :',rpn_loss_bbox, ' 1/T * rpn_loss_bbox :',1/16 * rpn_loss_bbox)
        if mode == 1:
            loss = rpn_loss_cls.mean() + rpn_loss_bbox.mean() + act_loss_bbox.mean() + rpn_loss_cls_16.mean() \
                   + rpn_loss_bbox_16.mean() +  act_loss_bbox_16.mean() 
        elif mode == 3:
            loss = sgl_rois_bbox_loss.mean()

        elif mode == 4:
            loss = rpn_loss_cls.mean() +  rpn_loss_bbox.mean() # + sgl_rois_bbox_loss.mean()

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
    sample_duration = 16

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
    # epochs = 40

    # epochs = 1
    # epochs = 40
    n_devs = torch.cuda.device_count()
    for epoch in range(epochs):
        print(' ============\n| Epoch {:0>2}/{:0>2} |\n ============'.format(epoch+1, epochs))

        if (epoch + 1) % (lr_decay_step ) == 0:
            adjust_learning_rate(optimizer, lr_decay_gamma)
            lr *= lr_decay_gamma
            print('adjust learning rate {}...'.format(lr))

        act_model, loss = training(epoch, device, act_model, dataset_frames, sample_duration, spatial_transform, temporal_transform, boxes_file, split_txt_path, cls2idx, n_devs*8, 0, lr, mode=4)

        if ( epoch + 1 ) % 5 == 0:
            torch.save(act_model.state_dict(), "action_net_model_steady_anchors_roi_align_08_05.pwf".format(epoch+1))
    torch.save(act_model.state_dict(), "action_net_model_steady_anchors_roi_align.pwf")
