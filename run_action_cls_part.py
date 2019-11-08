import os
import numpy as np
import glob

from  tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

# from resnet_3D import resnet34
from ucf_dataset import Video_UCF, video_names

from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, ToTensor, Resize)
from temporal_transforms import LoopPadding

from action_cls_part import ACT_cls
from resize_rpn import resize_rpn, resize_tube
import pdb

np.random.seed(42)

if __name__ == '__main__':

    # torch.cuda.device_count()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    dataset_frames = '../UCF-101-frames'
    boxes_file = '../pyannot.pkl'
    split_txt_path = '../UCF101_Action_detection_splits/'

    # dataset_frames = '../UCF-101-frames'
    # boxes_file = '../pyannot.pkl'
    # split_txt_path = '../UCF101_Action_detection_splits/'

    n_devs = torch.cuda.device_count()
    sample_size = 112
    sample_duration = 16  # len(images)
    # sample_duration = 8  # len(images)

    batch_size = 1
    n_threads = 0

    # # get mean
    # mean =  [103.75581543 104.79421473  91.16894564] # jhmdb
    # mean = [103.29825354, 104.63845484,  90.79830328]  # jhmdb from .png
    mean = [112.07945832, 112.87372333, 106.90993363]  # ucf-101 24 classes

    # generate model
    last_fc = False

    actions = ['__background__', 'Basketball','BasketballDunk','Biking','CliffDiving','CricketBowling',
               'Diving','Fencing','FloorGymnastics','GolfSwing','HorseRiding','IceDancing',
               'LongJump','PoleVault','RopeClimbing','SalsaSpin','SkateBoarding','Skiing',
               'Skijet','SoccerJuggling','Surfing','TennisSwing','TrampolineJumping',
               'VolleyballSpiking','WalkingWithDog']

    cls2idx = {actions[i]: i for i in range(0, len(actions))}

    ### get videos id

    spatial_transform = Compose([Scale(sample_size),  # [Resize(sample_size),
                                 ToTensor(),
                                 Normalize(mean, [1, 1, 1])])
    temporal_transform = LoopPadding(sample_duration)

    n_classes = len(actions)

    # Init action_net

    model = ACT_cls(actions, sample_duration)
    model.create_architecture()

    model.to(device)

    data = Video_UCF(dataset_frames, frames_dur=sample_duration, spatial_transform=spatial_transform,
                 temporal_transform=temporal_transform, json_file = boxes_file,
                 split_txt_path=split_txt_path, mode='train', classes_idx=cls2idx)

    clips, h, w,  n_frames, im_info, target = data[14]
    clips2, h2, w2,  n_frames2, im_info2, target2 = data[15]

    
    clips_ = clips.unsqueeze(0).to(device)
    im_info_ = im_info.unsqueeze(0).to(device)
    target_ = torch.Tensor([target]).unsqueeze(0).to(device)

    clips_2 = clips2.unsqueeze(0).to(device)
    im_info_2 = im_info2.unsqueeze(0).to(device)
    target_2 = torch.Tensor([target2]).unsqueeze(0).to(device)

    clips_ = torch.cat((clips_,clips_2))
    im_info_ = torch.Tensor([[112,112,16],[112,112,16]]).to(device)
    target_ = torch.Tensor([target_, target_2]).to(device)

    # clips_ = torch.cat((clips_,clips_2,clips_,clips_2))
    # gt_tubes_r_ = torch.cat((gt_tubes_r_, gt_tubes_r_2,gt_tubes_r_, gt_tubes_r_2))
    # gt_rois_ = torch.cat((gt_rois_, gt_rois_2,gt_rois_, gt_rois_2))
    # n_actions_ = torch.cat((n_actions_, n_actions_2,n_actions_, n_actions_2))
    # start_fr = torch.cat((start_fr,start_fr_2,start_fr,start_fr_2))
    # im_info_ = torch.Tensor([[112,112,16],[112,112,16],[112,112,16],[112,112,16]]).to(device)


    print('im_info_.shape :',im_info_.shape)
    print('**********Starts**********')
    # exit(-1)
    inputs = Variable(clips_)
    cls_scr, cls_loss  = model(inputs, \
                               im_info_,
                               target_)

    print('**********VGIKE**********')

