import os
import numpy as np
import glob

from  tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from resnet_3D import resnet34
from jhmdb_dataset import Video
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

    dataset_folder = '/gpu-data/sgal/JHMDB-act-detector-frames'
    splt_txt_path =  '/gpu-data/sgal/splits'
    boxes_file = '../temporal_localization/poses.json'

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

    data = Video(dataset_folder, frames_dur=sample_duration, spatial_transform=spatial_transform,
                 temporal_transform=temporal_transform, json_file = boxes_file,
                 split_txt_path=splt_txt_path, mode='train', classes_idx=cls2idx)
    # data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
    #                                           shuffle=True, num_workers=n_threads, pin_memory=True)

    n_classes = len(classes)

    # Init action_net
    model = Model(classes)
    model.create_architecture()
    if torch.cuda.device_count() > 1:
        print('Using {} GPUs!'.format(torch.cuda.device_count()))

        model = nn.DataParallel(model)

    model.to(device)

    # clips, h, w, gt_tubes, n_actions = data[1451]
    clips, (h, w), gt_tubes_r, gt_rois, n_actions, n_frames = data[144]

    # clips = torch.stack((clips,clips),dim=0).to(device) 

    # clips = torch.stack((clips,clips),dim=0).to(device)
    # gt_tubes = torch.stack((gt_tubes_r,gt_tubes2_r),dim=0).to(device)
    # n_actions = torch.Tensor((n_actions,n_actions2)).to(device)
    # im_info = torch.Tensor([[sample_size, sample_size, sample_duration]] * gt_tubes.size(1)).to(device)

    # clips = torch.stack((clips,clips),dim=0).to(device)
    # gt_tubes = torch.stack((gt_tubes_r,gt_tubes2_r),dim=0).to(device)
    # n_actions = torch.Tensor((n_actions,n_actions2)).to(device)
    im_info = torch.Tensor([[sample_size, sample_size, n_frames]] ).to(device)
    clips = clips.unsqueeze(0).to(device)
    gt_tubes_r = gt_tubes_r.unsqueeze(0).to(device)
    gt_rois = gt_rois.unsqueeze(0).to(device)
    n_actions = n_actions.unsqueeze(0).to(device)
    
    print('n_actions :',n_actions)
    print('clips :',clips.shape)
    print('gt_tubes :',gt_tubes_r)
    print('gt_tubes.shape :',gt_tubes_r.shape)

    print('im_info :',im_info)
    print('im_info.shape :',im_info.shape)

    print('n_actions :',n_actions)
    print('n_actions.shape :',n_actions.shape)

    # rois,  bbox_pred, rpn_loss_cls, \
    # rpn_loss_bbox,  act_loss_bbox, rois_label = model(clips,
    #                                                   im_info,
    #                                                   gt_tubes, None,
    #                                                   n_actions)
    ret = model(clips,
                im_info,
                gt_tubes_r, gt_rois,
                n_actions)

    print('**********VGIKE**********')
    # print('rois.shape :',rois.shape)
    # print('rois :',rois)

