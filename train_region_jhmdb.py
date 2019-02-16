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
from region_net import _RPN

import pdb

np.random.seed(42)

if __name__ == '__main__':

    # torch.cuda.device_count()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    dataset_folder = '/gpu-data/sgal/JHMDB-act-detector-frames'
    splt_txt_path =  '/gpu-data/sgal/splits'
    boxes_file = './poses.json'

    sample_size = 112
    sample_duration = 16  # len(images)

    batch_size = 1
    n_threads = 4

    mean = [103.29825354, 104.63845484,  90.79830328]  # jhmdb from .png

    # generate model
    last_fc = False

    # classes = ['basketballdunk', 'basketballshooting','cliffdiving', 'cricketbowling', 'fencing', 'floorgymnastics',
    #            'icedancing', 'longjump', 'polevault', 'ropeclimbing', 'salsaspin', 'skateboarding',
    #            'skiing', 'skijet', 'surfing', 'biking', 'diving', 'golfswing', 'horseriding',
    #            'soccerjuggling', 'tennisswing', 'trampolinejumping', 'volleyballspiking', 'walking']

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

    model_data = torch.load('../temporal_localization/resnet-34-kinetics.pth')
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
    epochs = 10
    for epoch in range(epochs):
        print(' ============\n| Epoch {:0>2}/{:0>2} |\n ============'.format(epoch, epochs))

        loss_temp = 0
        # start = time.time()
        rpn_model.train()
        model.eval()

        ## 2 rois : 1450
        for step, data  in enumerate(data_loader):
            # print('&&&&&&&&&&')
            print('step -->\t',step)
            # clips,  (h, w), gt_tubes, gt_rois = data
            clips,  (h, w), gt_tubes_r, n_actions = data
            clips = clips.to(device)
            gt_tubes_r = gt_tubes_r.to(device)

            inputs = Variable(clips)
            # print('gt_tubes.shape :',gt_tubes.shape )
            # print('gt_rois.shape :',gt_rois.shape)
            outputs = model(inputs)
            print('outputs: ', outputs[0][0][0][0])
            im_info = torch.Tensor([[sample_size, sample_size, sample_duration]] * gt_tubes_r.size(1)).to(device)

            rois, rpn_loss_cls, rpn_loss_box = rpn_model(outputs,
                                                         im_info,
                                                         gt_tubes_r, None, n_actions)
            loss = rpn_loss_cls.mean() + rpn_loss_box.mean()
            loss_temp += loss.item()

            # backw\ard
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('loss_temp :',loss_temp)
        print('Train Epoch: {} \tLoss: {:.6f}\t'.format(
            epoch,loss_temp/step))
        if ( epoch + 1 ) % 5 == 0:
            torch.save(rpn_model.state_dict(), "rpn_model_{0:03d}.pwf".format(epoch+1))
        torch.save(rpn_model.state_dict(), "rpn_model_pre_{0:03d}.pwf".format(epoch))
