import os
import numpy as np
import glob

from  tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from resnet_3D import resnet34
from simple_dataset import Video
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

    dataset_folder = '/gpu-data2/sgal/UCF-101-frames'
    boxes_file = '/gpu-data/sgal/pyannot.pkl'

    sample_size = 112
    sample_duration = 16  # len(images)

    batch_size = 1
    n_threads = 0

    # # get mean
    mean = [103.29825354, 104.63845484,  90.79830328]  # jhmdb from .png

    # generate model
    classes = ['brush_hair', 'clap', 'golf', 'kick_ball', 'pour',
               'push', 'shoot_ball', 'shoot_gun', 'stand', 'throw', 'wave',
               'catch','climb_stairs', 'jump', 'pick', 'pullup', 'run', 'shoot_bow', 'sit',
               'swing_baseball', 'walk' ]


    cls2idx = {classes[i]: i for i in range(0, len(classes))}

    spatial_transform = Compose([Scale(sample_size),  # [Resize(sample_size),
                                 ToTensor(),
                                 Normalize(mean, [1, 1, 1])])
    temporal_transform = LoopPadding(sample_duration)

    # data = Video(dataset_folder, frames_dur=sample_duration, spatial_transform=spatial_transform,
    #              temporal_transform=temporal_transform, json_file = boxes_file,
    #              split_txt_path=split_txt_path, mode='train', classes_idx=cls2idx)
    # # data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
    # #                                           shuffle=True, num_workers=n_threads, pin_memory=True)

    n_classes = len(classes)

    # Init action_net
    model = Model(classes, sample_duration, sample_size)
    model.create_architecture()
    if torch.cuda.device_count() > 1:
        print('Using {} GPUs!'.format(torch.cuda.device_count()))

        model = nn.DataParallel(model)

    model.to(device)

    print('**********Start**********')
    # ret = model(clips_t, target_t,
    #             im_info_t,
    #             gt_tubes_t, None,
    #             n_frames_t, num_boxes, max_dim=1, phase=2)
    vid_path = 'PoleVault/v_PoleVault_g06_c02'
    mode = 'train'
    ret = model(device, dataset_folder, vid_path, spatial_transform, temporal_transform, boxes_file, mode, cls2idx)
    print('**********VGIKE**********')
    # print('rois.shape :',rois.shape)
    # print('rois :',rois)

