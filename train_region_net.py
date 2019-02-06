import os
import numpy as np
import glob

from  tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from resnet_3D import resnet34
from video_dataset import Video
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, ToTensor, Resize)
from temporal_transforms import LoopPadding
from region_net import _RPN

import pdb

np.random.seed(42)

if __name__ == '__main__':

    # torch.cuda.device_count()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    # dataset_folder = '/gpu-data/sgal/UCF-101-frames'
    # boxes_file = '/gpu-data/sgal/UCF-bboxes.json'
    dataset_folder = '../UCF-101-frames'
    boxes_file = '../UCF-101-frames/UCF-bboxes.json'

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

    classes = ['basketballdunk', 'basketballshooting','cliffdiving', 'cricketbowling', 'fencing', 'floorgymnastics',
               'icedancing', 'longjump', 'polevault', 'ropeclimbing', 'salsaspin', 'skateboarding',
               'skiing', 'skijet', 'surfing', 'biking', 'diving', 'golfswing', 'horseriding',
               'soccerjuggling', 'tennisswing', 'trampolinejumping', 'volleyballspiking', 'walking']

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

    n_classes = len(classes)
    resnet_shortcut = 'A'

    ## ResNet 34 init
    model = resnet34(num_classes=400, shortcut_type=resnet_shortcut,
                     sample_size=sample_size, sample_duration=sample_duration,
                     last_fc=last_fc)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)

    model_data = torch.load('./resnet-34-kinetics.pth')
    model.load_state_dict(model_data['state_dict'])
    model.eval()

    lr = 0.001

    # Init rpn1
    rpn_model = _RPN(256).cuda()

    params = []
    for key, value in dict(rpn_model.named_parameters()).items():
        # print(key, value.requires_grad)
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params':[value],'lr':lr*(True + 1), \
                            'weight_decay': False and 0.0005 or 0}]
            else:
                params += [{'params':[value],'lr':lr, 'weight_decay': 0.0005}]

    lr = lr * 0.1
    optimizer = torch.optim.Adam(params)

    # epochs = 20
    epochs = 1
    for epoch in range(epochs):
        print(' ============\n| Epoch {:0>2}/{:0>2} |\n ============'.format(epoch, epochs))

        loss_temp = 0
        # start = time.time()
        rpn_model.train()
        model.eval()

        for step, (    clips,  (h, w), gt_tubes, gt_rois) in enumerate(data_loader):
            inputs = Variable(clips, requires_grad=True)
            outputs = model(inputs)
            print('step {}'.format(step))
            rois, rpn_loss_cls, rpn_loss_box = rpn_model(outputs,
                                                         torch.Tensor(
                                                             [[h, w]] * gt_tubes.size(1)).cuda(),
                                                         gt_tubes.cuda(), gt_rois, len(gt_tubes))
            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() 
            loss_temp += loss.item()

            # backw\ard
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Train Epoch: {} \tLoss: {:.6f}\t'.format(
            epoch,loss_temp/step))
        # if ( epoch + 1 ) % 5 == 0:
        #     torch.save(rpn_model.state_dict(), "rpn_model_{0:03d}.pwf".format(epoch))
        torch.save(rpn_model.state_dict(), "rpn_model_{0:03d}.pwf".format(epoch))
