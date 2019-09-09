import os
import numpy as np
import glob

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from resnet_3D import resnet34
from jhmdb_dataset import Video
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, ToTensor, Resize)
from temporal_transforms import LoopPadding

# from action_net import ACT_net
from action_prog import Act_Prog
from resize_rpn import resize_rpn, resize_tube
import pdb

np.random.seed(42)

if __name__ == '__main__':

    # torch.cuda.device_count()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    dataset_folder = '../JHMDB-act-detector-frames'
    split_txt_path =  '../splits'
    boxes_file = '../poses.json'

    sample_size = 112
    # sample_duration = 16  # len(images)
    sample_duration = 8  # len(images)

    batch_size = 1
    n_threads = 0

    # # get mean
    mean = [103.29825354, 104.63845484,  90.79830328]  # jhmdb from .png

    # generate model
    actions = ['__background__','brush_hair', 'clap', 'golf', 'kick_ball', 'pour',
               'push', 'shoot_ball', 'shoot_gun', 'stand', 'throw', 'wave',
               'catch','climb_stairs', 'jump', 'pick', 'pullup', 'run', 'shoot_bow', 'sit',
               'swing_baseball', 'walk' ]


    cls2idx = {actions[i]: i for i in range(0, len(actions))}

    spatial_transform = Compose([Scale(sample_size),  # [Resize(sample_size),
                                 ToTensor(),
                                 Normalize(mean, [1, 1, 1])])
    temporal_transform = LoopPadding(sample_duration)

    data = Video(dataset_folder, frames_dur=sample_duration, spatial_transform=spatial_transform,
                 temporal_transform=temporal_transform, json_file = boxes_file,
                 split_txt_path=split_txt_path, mode='train', classes_idx=cls2idx)
    # data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
    #                                           shuffle=True, num_workers=n_threads, pin_memory=True)

    n_classes = len(actions)

    model_path = './action_net_model_8frm_2_avg_jhmdb.pwf'

    # Init action_net
    model = Act_Prog(actions, sample_duration)
    model.create_architecture(model_path)


    model = nn.DataParallel(model)
    model.to(device)

    # model_path = './action_net_model_8frm_2_avg_jhmdb.pwf'
    # model_data = torch.load(model_path)
    # model.module.act_net.load_state_dict(model_data)

    # clips, h, w, gt_tubes, gt_rois, target, n_frames, im_info = data[24]
    clips, h, w, gt_tubes, gt_rois, target, n_frames, im_info, rate = data[24]

    clips_t = clips.unsqueeze(0).to(device)
    target_t = torch.Tensor([target]).unsqueeze(0).to(device)
    gt_tubes_t = gt_tubes.float().unsqueeze(0).to(device)
    gt_rois_t = gt_rois.float().unsqueeze(0).to(device)
    im_info_t = im_info.unsqueeze(0).to(device)
    n_frames_t = torch.Tensor([n_frames]).long().unsqueeze(0).to(device)
    num_boxes = torch.Tensor([[1],[1],[1]]).unsqueeze(0).to(device)
    start_fr = torch.zeros((1,1)).to(device)
    rate_ = torch.Tensor([rate]).unsqueeze(1).to(device)

    print('clips_t.shape :',clips_t.shape)
    print('gt_rois_t.shape :',gt_rois_t.shape)
    print('gt_tubes_t.shape :',gt_tubes_t.shape)
    print('im_info_t.shape :',im_info_t.shape)
    print('**********Start**********')


    inputs = Variable(clips_t)
    ret = model(clips_t, \
                im_info_t, \
                gt_tubes_t, gt_rois_t, \
                start_fr, rate_)

    print('**********VGIKE**********')
    # print('rois.shape :',rois.shape)
    # print('rois :',rois)

