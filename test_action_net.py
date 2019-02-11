import os
import numpy as np
import glob
import json

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader


# from video_dataset import Video
from jhmdb_dataset import Video
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, ToTensor, Resize)
from temporal_transforms import LoopPadding

from action_net import ACT_net

import cv2

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)


    sample_size = 112
    sample_duration = 16  # len(images)
    batch_size = 1
    n_threads = 0

    # # get mean
    # mean =  [103.75581543 104.79421473  91.16894564] # jhmdb
    mean = [103.29825354, 104.63845484,  90.79830328]  # jhmdb from .png

    # generate model
    last_fc = False

    scale_size = [sample_size,sample_size]
    spatial_transform = Compose([Scale(sample_size),  # [Resize(sample_size),
                                 ToTensor(),
                                 Normalize(mean, [1, 1, 1])])
    temporal_transform = LoopPadding(sample_duration)

    # ## UCF code
    # dataset_folder = '/gpu-data/sgal/UCF-101-frames'
    # boxes_file = './pyannot.pkl'
    # actions = ['Basketball','BasketballDunk','Biking','CliffDiving','CricketBowling',
    #            'Diving','Fencing','FloorGymnastics','GolfSwing','HorseRiding','IceDancing',
    #            'LongJump','PoleVault','RopeClimbing','SalsaSpin','SkateBoarding','Skiing',
    #            'Skijet','SoccerJuggling','Surfing','TennisSwing','TrampolineJumping',
    #            'VolleyballSpiking','WalkingWithDog']

    # cls2idx = {actions[i]: i for i in range(0, len(actions))}


    # data = Video(dataset_folder, frames_dur=sample_duration, spatial_transform=spatial_transform,
    #              temporal_transform=temporal_transform, json_file=boxes_file,
    #              mode='test', classes_idx=cls2idx, scale_size = scale_size)
    # # data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
    # #                                           shuffle=True, num_workers=n_threads, pin_memory=True)

    ## JHMDB code

    dataset_folder = '/gpu-data/sgal/JHMDB-act-detector-frames'
    splt_txt_path =  '/gpu-data/sgal/splits'
    boxes_file = './poses.json'

    classes = ['brush_hair', 'clap', 'golf', 'kick_ball', 'pour',
               'push', 'shoot_ball', 'shoot_gun', 'stand', 'throw', 'wave',
               'catch','climb_stairs', 'jump', 'pick', 'pullup', 'run', 'shoot_bow', 'sit',
               'swing_baseball', 'walk' ]


    cls2idx = {classes[i]: i for i in range(0, len(classes))}

    data = Video(dataset_folder, frames_dur=sample_duration, spatial_transform=spatial_transform,
                 temporal_transform=temporal_transform, json_file = boxes_file,
                 split_txt_path=splt_txt_path, mode='test', classes_idx=cls2idx)
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                              shuffle=True, num_workers=n_threads, pin_memory=True)

    
    
    # clips,  (h, w), gt_tubes, final_rois = data[906]
    # clips,  (h, w), gt_tubes, final_rois = data[905]
    # clips, (h,w), gt_tubes, gt_rois, path,frame_indices = data[1024]
    clips, (h,w), gt_tubes, gt_rois, path,frame_indices = data[100]

    
    print('path :',path)
    print('clips.shape :',clips.shape)
    clips = clips.unsqueeze(0)
    gt_tubes = gt_tubes.unsqueeze(0)
    print('gt_rois.shape :',gt_rois.shape)
    print('gt_rois :', gt_rois)

    n_classes = len(classes)
    resnet_shortcut = 'A'

    # Init action_net
    model = ACT_net(classes)
    model = nn.DataParallel(model)
    model.to(device)

    model_data = torch.load('./jmdb_model_020.pwf')
    model.load_state_dict(model_data)
    model.eval()


    rois,  bbox_pred, rpn_loss_cls, \
    rpn_loss_bbox,  act_loss_bbox, rois_label = model(clips,
                                                      torch.Tensor([[h,w]] * gt_tubes.size(1)).to(device),
                                                      gt_tubes, gt_rois,
                                                      torch.Tensor(len(gt_tubes)).to(device))

    print('h %d w %d ' % (h,w))
    rois[:,[0,2]] =rois[:,[0,2]].clamp_(min=0, max=112)
    rois[:,[1,3]] =rois[:,[1,3]].clamp_(min=0, max=112)
    print('rois.shape :',rois.shape)
    rois = rois[:,:,:-1]
    print('rois.shape :',rois.shape)

    rois = rois.view(300,16,-1).permute(1,0,2).cpu().numpy()
    colors = [ (255,0,0), (0,255,0), (0,0,255)]
    clips = clips.squeeze().permute(1,2,3,0)
    print('rois.shape :',rois.shape)
    for i in range(len(frame_indices)):
        # img = cv2.imread(os.path.join(path, 'image_{:0>5}.jpg'.format(frame_indices[i])))
        # img = cv2.imread(os.path.join(path, '{:0>5}.png'.format(frame_indices[i])))
        img = clips[i].numpy()
        print(img.shape)
        img_tmp = img.copy()
        # if img.all():
        #     print('Image {} not found '.format(os.path.join(path, 'image_{:0>5}.jpg'.format(frame_indices[i]))))
        #     break
        for j in range(20):
            cv2.rectangle(img_tmp,(int(rois[i,j,0]),int(rois[i,j,1])),(int(rois[i,j,2]),int(rois[i,j,3])), (255,0,0),3)
        # print('out : ./out/{:0>3}.jpg'.format(i))
        cv2.imwrite('./out_frames/action_{:0>3}.jpg'.format(i), img_tmp)

