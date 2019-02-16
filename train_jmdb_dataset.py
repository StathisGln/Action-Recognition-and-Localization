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
from action_net import ACT_net
from resize_rpn import resize_rpn, resize_tube
import pdb

np.random.seed(42)

torch.backends.cudnn.benchmark = True # for accelerating

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
    # batch_size = 1
    n_threads = 2

    # # get mean
    # mean =  [103.75581543 104.79421473  91.16894564] # jhmdb
    mean = [103.29825354, 104.63845484,  90.79830328]  # jhmdb from .png
    # mean = [112.07945832, 112.87372333, 106.90993363]  # ucf-101 24 classes
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
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                              shuffle=True, num_workers=n_threads, pin_memory=True)

    n_classes = len(classes)
    resnet_shortcut = 'A'

    lr = 0.001

    # Init action_net
    model = ACT_net(classes)
    model.create_architecture()
    if torch.cuda.device_count() > 1:
        print('Using {} GPUs!'.format(torch.cuda.device_count()))

        model = nn.DataParallel(model)

    model.to(device)

    params = []
    for key, value in dict(model.named_parameters()).items():
        # print(key, value.requires_grad)
        if value.requires_grad:
            print('key :',key)
            if 'bias' in key:
                params += [{'params':[value],'lr':lr*(True + 1), \
                            'weight_decay': False and 0.0005 or 0}]
            else:
                params += [{'params':[value],'lr':lr, 'weight_decay': 0.0005}]

    lr = lr * 0.1
    optimizer = torch.optim.Adam(params)

    epochs = 30
    # epochs = 1
    for epoch in range(epochs):
        print(' ============\n| Epoch {:0>2}/{:0>2} |\n ============'.format(epoch+1, epochs))

        loss_temp = 0
        # start = time.time()


        ## 2 rois : 1450
        for step, data  in enumerate(data_loader):
            # print('&&&&&&&&&&')
            # print('step -->\t',step)
            # clips,  (h, w), gt_tubes, gt_rois = data
            clips,  (h, w), gt_tubes_r, n_actions = data
            clips = clips.to(device)
            gt_tubes_r = gt_tubes_r.to(device)
            # print('gt_tubes_r :',gt_tubes_r)
            # print('gt_tubes :',gt_tubes)
            # h = h.to(device)
            # w = w.to(device)
            # gt_tubes = gt_tubes.to(device)
            n_actions = n_actions.to(device)
            im_info = torch.Tensor([[sample_size, sample_size, sample_duration]] * gt_tubes_r.size(1)).to(device)
            # print('gt_tubes_r.shape :',gt_tubes_r.shape )
            inputs = Variable(clips)
            rois,  bbox_pred, rpn_loss_cls, rpn_loss_bbox, \
            act_loss_cls,  act_loss_bbox, rois_label = model(inputs,
                                                              im_info,
                                                              gt_tubes_r, None,
                                                              n_actions)
            # print('rois :',rois)
            # print('rpn_loss_bbox :',rpn_loss_bbox)
            # print('rpn_loss_cls :',rpn_loss_cls)
            # loss = rpn_loss_cls.mean() + rpn_loss_bbox.mean() + act_loss_bbox.mean() + act_loss_cls.mean()
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
            torch.save(model.state_dict(), "jmdb_model_{0:03d}.pwf".format(epoch+1))
        # torch.save(model.state_dict(), "jmdb_model_pre_{0:03d}.pwf".format(epoch))

