import os
import numpy as np
import glob

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from resnet_3D import resnet34
from video_dataset import Video
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, ToTensor, Resize)
from temporal_transforms import LoopPadding

if __name__ == '__main__':

    # torch.cuda.device_count()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    dataset_folder = '/gpu-data/sgal/UCF-101-frames'
    boxes_file = '/gpu-data/sgal/UCF-bboxes.json'
    sample_size = 112
    sample_duration = 8 #16  # len(images)

    batch_size = 1
    n_threads = 0

    # # get mean
    # mean =  [103.75581543 104.79421473  91.16894564] # jhmdb
    mean = [103.29825354, 104.63845484,  90.79830328]  # jhmdb from .png

    # generate model
    last_fc = False

    classes = ['BasketballDunk', 'CliffDiving', 'CricketBowling', 'Fencing', 'FloorGymnastics',
               'IceDancing', 'LongJump', 'PoleVault', 'RopeClimbing', 'SalsaSpin', 'SkateBoarding',
               'Skiing', 'Skijet', 'Surfing', 'Basketball','Biking', 'Diving', 'GolfSwing', 'HorseRiding',
               'SoccerJuggling', 'TennisSwing', 'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog']

    cls2idx = {classes[i]: i for i in range(0, len(classes))}

    spatial_transform = Compose([Scale(sample_size),  # [Resize(sample_size),
                                 ToTensor(),
                                 Normalize(mean, [1, 1, 1])])
    temporal_transform = LoopPadding(sample_duration)

    data = Video(dataset_folder, frames_dur=sample_duration, spatial_transform=spatial_transform,
                 temporal_transform=temporal_transform, json_file=boxes_file,
                 mode='train', classes_idx=cls2idx)
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                              shuffle=True, num_workers=n_threads, pin_memory=True)

    clips, target, (h, w), gt_bboxes = next(data_loader.__iter__())
    print('clips.shape :',clips.shape)
    print('target.shape :',target.shape)
    print('h :', h, ' w :', w)
    print('gt_bboxes.shape :', gt_bboxes.shape)
    # n_classes = len(classes)
    # resnet_shortcut = 'A'

    # model = resnet34(num_classes=400, shortcut_type=resnet_shortcut,
    #                  sample_size=sample_size, sample_duration=sample_duration,
    #                  last_fc=last_fc)
    # model = model.cuda()
    # model = nn.DataParallel(model, device_ids=None)

    # model_data = torch.load('./resnet-34-kinetics.pth')
    # model.load_state_dict(model_data['state_dict'])
    # model.eval()

    # lr = 0.001
    # rpn_model = _RPN(512).cuda()

    # clips, target, (h, w), gt_bboxes = next(data_loader.__iter__())
    # boxes = gt_bboxes.cpu().numpy().tolist()
    # inputs = Variable(clips)
    # outputs = model(inputs)
    # rois, rpn_loss_cls, rpn_loss_box = rpn_model(outputs,
    #                                              torch.Tensor(
    #                                                  [[h, w]] * gt_bboxes.size(1)).cuda(),
    #                                              gt_bboxes.permute(1, 0, 2).unsqueeze(0).cuda(), len(gt_bboxes))
