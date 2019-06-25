import os
import numpy as np
import glob

from  tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from ucf_dataset import Video_UCF, video_names
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, ToTensor, Resize)
from temporal_transforms import LoopPadding
from model import Model

from create_video_id import get_vid_dict
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

    n_devs = torch.cuda.device_count()
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

    # Init action_net
    model = Model(actions, sample_duration, sample_size)
    model.load_part_model()
    if torch.cuda.device_count() > 1:
        print('Using {} GPUs!'.format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    model.to(device)
    vid_name_loader = video_names(dataset_frames, split_txt_path, boxes_file, vid2idx, mode='train')

    data_loader = torch.utils.data.DataLoader(vid_name_loader, batch_size=n_devs, num_workers=8*n_devs, pin_memory=True,
                                              shuffle=True)    # reset learning rate


    # clips, h, w, gt_tubes, gt_rois, n_actions = data[14]
    # clips, h, w, gt_tubes, n_actions = data[1451]
    # for i in range(200):

    # for i in range(200,500):
    #     vid_id, clips, boxes, n_frames, n_actions, h, w =vid_name_loader[i]
    #     print(i, n_actions)

    #     if n_actions > 1:
    #         print(i)
    #         exit(-1)

    # exit(-1)
    vid_id, clips, boxes, n_frames, n_actions, h, w, target =vid_name_loader[14]
    # vid_id, clips, boxes, n_frames, n_actions, h, w =vid_name_loader[209]


    vid_id = torch.Tensor(vid_id).int()
    clips = clips.unsqueeze(0).to(device)
    boxes = torch.from_numpy(boxes).to(device)
    n_frames = torch.from_numpy(n_frames).to(device)
    n_actions = torch.from_numpy(n_actions).int().to(device)
    im_info = torch.Tensor([h,w,clips.size(2)]).unsqueeze(0).to(device)
    mode = 'train'
    print('**********Starts**********')

    tubes,  \
    prob_out, cls_loss =  model(n_devs, dataset_frames, \
                                vid_names, clips, vid_id,  \
                                boxes, \
                                mode, cls2idx, n_actions,n_frames, h, w)

    # rois,  bbox_pred, cls_prob, \
    # rpn_loss_cls,  rpn_loss_bbox, \
    # act_loss_cls,  act_loss_bbox, rois_label = model(clips,
    #                                                  im_info,
    #                                                  gt_tubes, None,
    #                                                  n_actions)

    print('**********VGIKE**********')
    print('rois.shape :',tubes.shape)
    print('rois :',tubes)

