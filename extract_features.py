import os
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from ucf_dataset import Video_UCF, video_names

from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, ToTensor, Resize)
from temporal_transforms import LoopPadding

from create_video_id import get_vid_dict
from net_utils import adjust_learning_rate
from resize_rpn import resize_rpn, resize_tube, resize_boxes


from model import Model
import argparse

np.random.seed(42)

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    dataset_frames = '../UCF-101-frames'
    boxes_file = '../pyannot.pkl'
    split_txt_path = '../UCF101_Action_detection_splits'

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

    #####################################
    #          Start procedure          #
    #####################################

    # Init action_net
    n_devs = torch.cuda.device_count()
    
    model = Model(actions, sample_duration, sample_size)
    model.load_part_model()

    model.act_net = nn.DataParallel(model.act_net)
    model.act_net = model.act_net.to(device)
    model = model.to(device)

    vid_name_loader = video_names(dataset_frames, split_txt_path, boxes_file, vid2idx, mode='test')

    # data_loader = torch.utils.data.DataLoader(vid_name_loader, batch_size=n_devs, num_workers=8*n_devs, pin_memory=True,
    data_loader = torch.utils.data.DataLoader(vid_name_loader, batch_size=1, num_workers=1, pin_memory=True,

                                              shuffle=False)    # reset learning rate

    output_folder = '../UCF-101-features'
    for step, data  in enumerate(data_loader):
        
        vid_id, clips, boxes, n_frames, n_actions, h, w = data
        print('step :', step, 'vid_names[vid_id] :',vid_names[vid_id])
        class_folder = vid_names[vid_id].split('/')[0]

        vid_name =  vid_names[vid_id].split('/')[1]

        output_path_class_folder = os.path.join(output_folder, class_folder)
        if not os.path.exists(output_path_class_folder):
            os.mkdir(output_path_class_folder)
        output_path = os.path.join(output_path_class_folder, vid_name)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
                                   
        mode = 'train'
        vid_id = vid_id.to(device)
        n_frames = n_frames.to(device)
        n_actions = n_actions.to(device)
        h = h.to(device)
        w = w.to(device)

        video_tubes,  f_features, \
        final_tubes, \
        target_lbl, _, _ =  model(n_devs, dataset_frames, \
                                  vid_names, clips, vid_id,  \
                                  boxes, \
                                  mode, cls2idx, n_actions,n_frames, h, w)
        
        torch.save(video_tubes, os.path.join(output_path,'video_tubes.pt'))
        torch.save(f_features,  os.path.join(output_path,'f_features.pt'))
        torch.save(final_tubes, os.path.join(output_path,'final_tubes.pt'))
        torch.save(target_lbl,  os.path.join(output_path,'target_lbl.pt'))
        # if step == 1:
        #     break
