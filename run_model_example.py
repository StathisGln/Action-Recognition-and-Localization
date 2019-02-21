import os
import numpy as np
import glob

from  tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from video_dataset import Video_UCF
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, ToTensor, Resize)
from temporal_transforms import LoopPadding
from model import Model
from resize_rpn import resize_rpn, resize_tube
import pdb

np.random.seed(42)

if __name__ == '__main__':

    # torch.cuda.device_count()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    dataset_folder = '/gpu-data/sgal/UCF-101-frames'
    boxes_file = '/gpu-data/sgal/pyannot.pkl'
    # boxes_file = '/gpu-data/sgal/UCF-bboxes.json'
    # dataset_folder = '../UCF-101-frames'
    # boxes_file = '../UCF-101-frames/UCF-bboxes.json'

    sample_size = 112
    sample_duration = 16  # len(images)

    batch_size = 1
    n_threads = 0

    # # get mean
    # mean =  [103.75581543 104.79421473  91.16894564] # jhmdb
    # mean = [103.29825354, 104.63845484,  90.79830328]  # jhmdb from .png
    mean = [112.07945832, 112.87372333, 106.90993363]  # ucf-101 24 classes
    # generate model
    last_fc = False

    actions = ['Basketball','BasketballDunk','Biking','CliffDiving','CricketBowling',
               'Diving','Fencing','FloorGymnastics','GolfSwing','HorseRiding','IceDancing',
               'LongJump','PoleVault','RopeClimbing','SalsaSpin','SkateBoarding','Skiing',
               'Skijet','SoccerJuggling','Surfing','TennisSwing','TrampolineJumping',
               'VolleyballSpiking','WalkingWithDog']

    cls2idx = {actions[i]: i for i in range(0, len(actions))}

    spatial_transform = Compose([Scale(sample_size),  # [Resize(sample_size),
                                 ToTensor(),
                                 Normalize(mean, [1, 1, 1])])
    temporal_transform = LoopPadding(sample_duration)

    data = Video_UCF(dataset_folder, frames_dur=sample_duration, spatial_transform=spatial_transform,
                 temporal_transform=temporal_transform, json_file=boxes_file,
                 mode='train', classes_idx=cls2idx)


    n_classes = len(actions)

    # Init action_net
    model = Model(actions)
    model.create_architecture()
    if torch.cuda.device_count() > 1:
        print('Using {} GPUs!'.format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    model.to(device)

    # clips, h, w, gt_tubes, gt_rois, n_actions = data[14]
    # clips, h, w, gt_tubes, n_actions = data[1451]
    clips, h, w, gt_tubes, gt_rois, n_actions = data[144]
    # clips2, h2, w2, gt_tubes2, gt_rois2, n_actions2 = data[1450]

    print('n_actions :',n_actions)
    clips = clips.unsqueeze(0).to(device)
    gt_tubes = gt_tubes.unsqueeze(0).to(device)
    gt_rois = gt_rois.unsqueeze(0).to(device)
    n_actions = torch.Tensor([n_actions]).unsqueeze(0).to(device)
    im_info = torch.Tensor([112,112,clips.size(2)]).unsqueeze(0).to(device)

    print('clips.shape :',clips.shape )

    print('gt_tubes.shape :',gt_tubes.shape )
    print('gt_tubes :',gt_tubes )

    print('gt_rois.shape :',gt_rois.shape )
    # print('gt_rois :',gt_rois )

    print('im_info :',im_info)
    print('im_info.shape :',im_info.shape)

    print('n_actions :',n_actions)
    print('n_actions.shape :',n_actions.shape)

    print('**********Starts**********')

    ret = model(clips,
                im_info,
                gt_tubes, gt_rois,
                n_actions)
    # rois,  bbox_pred, cls_prob, \
    # rpn_loss_cls,  rpn_loss_bbox, \
    # act_loss_cls,  act_loss_bbox, rois_label = model(clips,
    #                                                  im_info,
    #                                                  gt_tubes, None,
    #                                                  n_actions)

    print('**********VGIKE**********')
    # print('rois.shape :',rois.shape)
    # print('rois :',rois)

