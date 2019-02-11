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
from action_net import ACT_net
from resize_rpn import resize_rpn, resize_tube
import pdb

np.random.seed(42)

if __name__ == '__main__':

    # torch.cuda.device_count()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    dataset_folder = '/gpu-data/sgal/UCF-101-frames'
    boxes_file = './pyannot.pkl'
    # boxes_file = '/gpu-data/sgal/UCF-bboxes.json'
    # dataset_folder = '../UCF-101-frames'
    # boxes_file = '../UCF-101-frames/UCF-bboxes.json'

    sample_size = 112
    sample_duration = 16  # len(images)

    batch_size = 1
    n_threads = 4

    # # get mean
    # mean =  [103.75581543 104.79421473  91.16894564] # jhmdb
    # mean = [103.29825354, 104.63845484,  90.79830328]  # jhmdb from .png
    mean = [112.07945832, 112.87372333, 106.90993363]  # ucf-101 24 classes
    # generate model
    last_fc = False

    # classes = ['basketballdunk', 'basketballshooting','cliffdiving', 'cricketbowling', 'fencing', 'floorgymnastics',
    #            'icedancing', 'longjump', 'polevault', 'ropeclimbing', 'salsaspin', 'skateboarding',
    #            'skiing', 'skijet', 'surfing', 'biking', 'diving', 'golfswing', 'horseriding',
    #            'soccerjuggling', 'tennisswing', 'trampolinejumping', 'volleyballspiking', 'walking']

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

    data = Video(dataset_folder, frames_dur=sample_duration, spatial_transform=spatial_transform,
                 temporal_transform=temporal_transform, json_file=boxes_file,
                 mode='train', classes_idx=cls2idx)
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                              shuffle=True, num_workers=n_threads, pin_memory=True)

    n_classes = len(actions)
    resnet_shortcut = 'A'

    lr = 0.001

    # Init action_net
    model = ACT_net(actions)

    if torch.cuda.device_count() > 1:
        print('Using {} GPUs!'.format(torch.cuda.device_count()))

        model = nn.DataParallel(model)

    model.to(device)

    params = []
    for key, value in dict(model.named_parameters()).items():
        # print(key, value.requires_grad)
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params':[value],'lr':lr*(True + 1), \
                            'weight_decay': False and 0.0005 or 0}]
            else:
                params += [{'params':[value],'lr':lr, 'weight_decay': 0.0005}]

    lr = lr * 0.1
    optimizer = torch.optim.Adam(params)

    epochs = 20
    # epochs = 5
    for epoch in range(epochs):
        print(' ============\n| Epoch {:0>2}/{:0>2} |\n ============'.format(epoch+1, epochs))

        loss_temp = 0
        # start = time.time()

        ## 2 rois : 1450
        for step,     data in enumerate(data_loader):
            print('&&&&&&&&&&')
            clips,  (h, w), gt_tubes, gt_rois = data
            # print('gt_tubes : ',gt_tubes)
            # print('gt_rois.shape : ',gt_rois.shape)
            gt_tubes = gt_tubes[:,0,:].unsqueeze(1).to(device)
            gt_rois = gt_rois[:,0,:,:].unsqueeze(1).to(device)
            # print('gt_tubes : ',gt_tubes)
            # print('gt_tubes.shape : ',gt_tubes.shape)
            # print('gt_tubes[0,0,5] - gt_tube[0,0,2]+1 :',gt_tubes[0,0,5] - gt_tubes[0,0,2]+1)
            # print('gt_tubes[0,0,5] - gt_tube[0,0,2]+1 != 16 :',gt_tubes[0,0,5] - gt_tubes[0,0,2]+1 != 16)
            if (gt_tubes[0,0,5] - gt_tubes[0,0,2]+1 != 16):
                # print('Only background, continue...')
                continue
            
            # print('gt_tubes :',gt_tubes)
            gt_rois =  gt_rois.squeeze(0)
            h = h.to(device)
            w = w.to(device)
            # print('gt_tubes.shape :',gt_tubes.shape )
            # print('gt_rois.shape :',gt_rois.shape)

            # gt_tubes_r = resize_tube(gt_tubes, h,w,sample_size)
            # gt_rois_r = resize_rpn(gt_rois, h,w,112)

            # inputs = Variable(clips)
            # print('gt_tubes.shape :',gt_tubes.shape )
            # print('gt_rois.shape :',gt_rois.shape)
            rois,  bbox_pred, rpn_loss_cls, \
            rpn_loss_bbox,  act_loss_bbox, rois_label = model(clips,
                                                              torch.Tensor([[h, w]] * gt_tubes.size(1)).to(device),
                                                              gt_tubes, gt_rois,
                                                              torch.Tensor(len(gt_tubes)).to(device))

            loss = rpn_loss_cls.mean() + rpn_loss_bbox.mean() + act_loss_bbox.mean()
            loss_temp += loss.item()

            # backw\ard
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # print('loss_temp :',loss_temp)
        print('Train Epoch: {} \tLoss: {:.6f}\t'.format(
            epoch,loss_temp/step))
        if ( epoch + 1 ) % 5 == 0:
            torch.save(model.state_dict(), "action_model_{0:03d}.pwf".format(epoch+1))
        torch.save(model.state_dict(), "action_model_pre_{0:03d}.pwf".format(epoch))
