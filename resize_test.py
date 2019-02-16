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
from resize_rpn import resize_rpn, resize_tube

from action_net import ACT_net
from bbox_transform import bbox_transform_inv, clip_boxes_3d, clip_boxes_batch, bbox_transform_inv_3d
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
                                 ToTensor()])
    temporal_transform = LoopPadding(sample_duration)

    spatial_transform2 = Compose([  # [Resize(sample_size),
                                 ToTensor()])

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
    boxes_file = '../temporal_localization/poses.json'

    classes = ['brush_hair', 'clap', 'golf', 'kick_ball', 'pour',
               'push', 'shoot_ball', 'shoot_gun', 'stand', 'throw', 'wave',
               'catch','climb_stairs', 'jump', 'pick', 'pullup', 'run', 'shoot_bow', 'sit',
               'swing_baseball', 'walk' ]


    cls2idx = {classes[i]: i for i in range(0, len(classes))}

    data = Video(dataset_folder, frames_dur=sample_duration, spatial_transform=spatial_transform,
                 temporal_transform=temporal_transform, json_file = boxes_file,
                 split_txt_path=splt_txt_path, mode='test', classes_idx=cls2idx)
    data2 = Video(dataset_folder, frames_dur=sample_duration, spatial_transform=spatial_transform2,
                 temporal_transform=temporal_transform, json_file = boxes_file,
                 split_txt_path=splt_txt_path, mode='test', classes_idx=cls2idx)


    
    
    # clips,  (h, w), gt_tubes, final_rois = data[906]
    # clips,  (h, w), gt_tubes, final_rois = data[905]
    # clips, (h,w), gt_tubes, gt_rois, path,frame_indices = data[1024]
    clips, (h,w), gt_tubes_r,n_actions, path,frame_indices = data[10]
    clips2, (h,w), gt_tubes_r,n_actions, path,frame_indices = data2[10]
    gt_tubes_r = gt_tubes_r.unsqueeze(0)
    # print(h,w)
    # print('path :',path)
    # print('clips.shape :',clips.shape)
    # print('clips2.shape :',clips2.shape)
    # clips = clips.unsqueeze(0)
    # gt_tubes = gt_tubes.unsqueeze(0)
    # print('gt_tubes.shape :',gt_tubes.shape )
    # print('gt_tubes :',gt_tubes)
    # clips = clips.to(device)
    # gt_tubes_new = gt_tubes.clone()
    # gt_tubes_r = resize_tube(gt_tubes_new, torch.Tensor([h]),torch.Tensor([w]),sample_size).to(device)
    # gt_tubes_r = gt_tubes_r.to(device)
    im_info = torch.Tensor([[sample_size, sample_size, sample_duration]] * gt_tubes_r.size(1)).to(device)

    colors = [ (255,0,0), (0,255,0), (0,0,255)]
    clips = clips.squeeze().permute(1,2,3,0)
    clips2 = clips2.squeeze().permute(1,2,3,0)
    # print('rois.shape :',rois.shape)
    for i in range(len(frame_indices)):
        # img = cv2.imread(os.path.join(path, 'image_{:0>5}.jpg'.format(frame_indices[i])))
        # img = cv2.imread(os.path.join(path, '{:0>5}.png'.format(frame_indices[i])))
        img = clips[i].cpu().numpy()
        # print(img.shape)
        img_tmp = img.copy()
        
        img2 =clips2[i].cpu().numpy()
        img_tmp2 = img2.copy()
        # # if img.all():
        # #     print('Image {} not found '.format(os.path.join(path, 'image_{:0>5}.jpg'.format(frame_indices[i]))))
        # #     break
        # for j in range(10):
        #     cv2.rectangle(img_tmp,(int(rois[0,j,0]),int(rois[0,j,1])),(int(rois[0,j,3]),int(rois[0,j,4])), (255,0,0),3)
        # # print('out : ./out/{:0>3}.jpg'.format(i))
        # cv2.imwrite('./out_frames/action_{:0>3}.jpg'.format(i), img_tmp)
        for j in range(10):
            cv2.rectangle(img_tmp,(int(gt_tubes_r[0,0,0]),int(gt_tubes_r[0,0,1])),(int(gt_tubes_r[0,0,3]),int(gt_tubes_r[0,0,4])), (0,255,0),3)
            # cv2.rectangle(img_tmp2,(int(gt_tubes[0,0,0]),int(gt_tubes[0,0,1])),(int(gt_tubes[0,0,3]),int(gt_tubes[0,0,4])), (0,255,0),3)
        cv2.imwrite('./out_frames/both_{:0>3}.jpg'.format(i), img_tmp)
        cv2.imwrite('./out_frames/regular_{:0>3}.jpg'.format(i), img_tmp2)

