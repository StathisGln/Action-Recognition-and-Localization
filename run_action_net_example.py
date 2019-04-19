import os
import numpy as np
import glob

from  tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from resnet_3D import resnet34
from ucf_dataset import Video_UCF, video_names

from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, ToTensor, Resize)
from temporal_transforms import LoopPadding
from action_net import ACT_net
from resize_rpn import resize_rpn, resize_tube
import pdb

np.random.seed(42)

if __name__ == '__main__':

    # torch.cuda.device_count()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    dataset_frames = '/gpu-data2/sgal/UCF-101-frames'
    boxes_file = '/gpu-data2/sgal/pyannot.pkl'
    split_txt_path = '/gpu-data2/sgal/UCF101_Action_detection_splits/'

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

    model = ACT_net(actions, sample_duration)
    model.create_architecture()

    model.to(device)

    data = Video_UCF(dataset_frames, frames_dur=sample_duration, spatial_transform=spatial_transform,
                 temporal_transform=temporal_transform, json_file = boxes_file,
                 split_txt_path=split_txt_path, mode='train', classes_idx=cls2idx)

    clips, h, w, gt_tubes_r, gt_rois, n_actions, n_frames, im_info = data[14]
    clips_ = clips.unsqueeze(0).to(device)
    gt_tubes_r_ = gt_tubes_r.unsqueeze(0).to(device)
    gt_rois_ = gt_rois.unsqueeze(0).to(device)
    n_actions_ = torch.from_numpy(n_actions).to(device)
    im_info_ = im_info.unsqueeze(0).to(device)
    start_fr = torch.zeros((1)).to(device)

    print('**********Starts**********')

    inputs = Variable(clips_)
    tubes, _, \
    rpn_loss_cls,  rpn_loss_bbox, \
    rpn_loss_cls_16,\
    rpn_loss_bbox_16,  rois_label, \
    sgl_rois_bbox_pred, sgl_rois_bbox_loss,  = model(inputs, \
                                                im_info_,
                                                gt_tubes_r_, gt_rois_,
                                                start_fr)

    print('**********VGIKE**********')
    # print('rois.shape :',rois.shape)
    # print('rois :',rois)

