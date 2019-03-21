import os
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from ucf_dataset import video_names

from model import Model

np.random.seed(42)

if __main__ == '__name__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)
    
    dataset_folder = '/gpu-data2/sgal/UCF-101-pickle'
    dataset_frames = '/gpu-data2/sgal/UCF-101-frames'
    boxes_file = '/gpu-data2/sgal/pyannot.pkl'
    split_txt_path = '/gpu-data2/sgal/UCF101_Action_detection_splits/'

    sample_size = 112
    sample_duration = 16  # len(images)

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

    ####################################################
    #          Prepare for feature extraction          #
    ####################################################

    n_devs = torch.cuda.device_count()
    model = Model(actions, sample_duration, sample_size)
    model.load_part_model()
    model.deactivate_action_net_grad()


    if torch.cuda.device_count() > 1:

        print('Using {} GPUs!'.format(torch.cuda.device_count()))
        # model.act_net = nn.DataParallel(model.act_net)
        model = nn.DataParallel(model)

    # model.act_net = model.act_net.to(device)
    model = model.to(device)
    # init data_loaders
    
    vid_name_loader = video_names(dataset_frames, split_txt_path, boxes_file, vid2idx, mode='train')
    data_loader = torch.utils.data.DataLoader(vid_name_loader, batch_size=n_devs, num_workers=8*n_devs, pin_memory=True,
                                              shuffle=True)    # reset learning rate

    for step, data  in enumerate(data_loader):

        vid_id, clips, boxes, n_frames, n_actions, h, w = data
        mode = 'train'

        vid_id = vid_id.to(device)
        n_frames = n_frames.to(device)
        n_actions = n_actions.to(device)
        h = h.to(device)
        w = w.to(device)

        tubes,  bbox_pred, \
        prob_out, rpn_loss_cls, \
        rpn_loss_bbox, act_loss_bbox,  cls_loss =  model(n_devs, dataset_folder, \
                                                         vid_names, clips, vid_id,  \
                                                         boxes, \
                                                         mode, cls2idx, n_actions,n_frames, h, w)

