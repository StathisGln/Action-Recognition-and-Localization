import os
import numpy as np
import glob

from  tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from video_dataset import video_names
from create_video_id import get_vid_dict

from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, ToTensor, Resize)
from temporal_transforms import LoopPadding

from model import Model

np.set_printoptions(threshold=np.nan)
np.random.seed(42)

if __name__ == '__main__':

    # torch.cuda.device_count()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    dataset_folder = '/gpu-data2/sgal/UCF-101-frames'
    boxes_file = '/gpu-data2/sgal/pyannot.pkl'
    spt_path = '/gpu-data2/sgal/UCF101_Action_detection_splits/'
    ### get videos id

    vid2idx,vid_names = get_vid_dict(dataset_folder)
    
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

    n_classes = len(classes)

    # Init action_net
    model = Model(classes, sample_duration, sample_size)
    model.create_architecture()

    # if torch.cuda.device_count() > 1:

    #     print('Using {} GPUs!'.format(torch.cuda.device_count()))
    #     model = nn.DataParallel(model)

    model.to(device)

    vid_name_loader = video_names(dataset_folder, boxes_file, vid2idx)
    data_loader = torch.utils.data.DataLoader(vid_name_loader, batch_size=batch_size,
                                              shuffle=False)
    # vid_id : [895]

    for step, t in enumerate(data_loader):
        continue
    
    # vid_path, n_actions, boxes = vid_names[1505]
    vid_id, boxes, n_frames,n_actions = vid_name_loader[3105]
    print('vid_id :',vid_id)
    print('n_action :',n_actions)
    print('n_frames :',n_frames)
    print('boxes.shape :',boxes.shape)
    # # vid_path = 'PoleVault/v_PoleVault_g06_c02'
    mode = 'train'
    print('**********Start**********')    



    vid_id_ = torch.Tensor([vid_id]).to(device).long()
    n_frames_ = torch.Tensor([n_frames]).to(device).long()
    n_actions_ = torch.Tensor([n_actions]).to(device).long()
    boxes_ = torch.from_numpy(boxes).unsqueeze(0).to(device)
    # boxes_ = boxes_[:n_actions_, :n_frames_]
    tubes,  bbox_pred, \
    prob_out, rpn_loss_cls, \
    rpn_loss_bbox, act_loss_bbox,  cls_loss =  model( dataset_folder, \
                                                      vid_names, vid_id_, spatial_transform, \
                                                      temporal_transform, boxes_, \
                                                      mode, cls2idx, n_actions_, n_frames_)

    print('**********VGIKE**********')

    print('tubes.shape :',tubes.shape)
    # print('tubes :',tubes)    

    print('bbox_pred.shape :',bbox_pred.shape)
    print('prob_out.shape :',prob_out.shape)
    print('rpn_loss_cls :',rpn_loss_cls)
    print('rpn_loss_bbox :',rpn_loss_bbox)
    print('act_loss_bbox :',act_loss_bbox)
    print('cls_loss :',cls_loss)
