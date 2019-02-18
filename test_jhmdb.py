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
    boxes_file = '../temporal_localization/poses.json'

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
    clips, (h,w), gt_tubes,n_actions, path,frame_indices = data[90]
    print(h,w)
    print('path :',path)
    print('clips.shape :',clips.shape)
    clips = clips.unsqueeze(0)
    gt_tubes = gt_tubes.unsqueeze(0)
    print('gt_tubes.shape :',gt_tubes.shape )
    clips = clips.to(device)
    gt_tubes_r = resize_tube(gt_tubes, torch.Tensor([h]),torch.Tensor([w]),sample_size).to(device)
    gt_tubes_r = gt_tubes_r.to(device)
    im_info = torch.Tensor([[sample_size, sample_size, sample_duration]] * gt_tubes_r.size(1)).to(device)

    n_classes = len(classes)
    resnet_shortcut = 'A'

    # Init action_net
    model = ACT_net(classes)
    model.create_architecture()
    data = model.act_rpn.RPN_cls_score.weight.data.clone()


    # model_data = torch.load('../temporal_localization/jmdb_model_015.pwf')
    model_data = torch.load('./jmdb_model_030.pwf')
    # # model_data = torch.load('../temporal_localization/r')

    model.load_state_dict(model_data)

    model = nn.DataParallel(model)
    model.to(device)

    model.eval()
    print('im_info :',im_info)
    print('-----Starts-----')
    rois,  bbox_pred, rpn_loss_cls, \
    rpn_loss_bbox,  act_loss_bbox, rois_label = model(clips,
                                                      im_info,
                                                      None, None, None)
    # rpn_loss_bbox,  act_loss_bbox, rois_label = model(clips,
    #                                                   torch.Tensor([[h,w]] * gt_tubes.size(1)).to(device),
    #                                                   gt_tubes, gt_rois,
    #                                                   torch.Tensor(len(gt_tubes)).to(device))
    print('-----Eksww-----')
    print('rois :',rois.shape)
    # print('rois :',rois.)
    rois = rois[:,:,1:]
    # print('bbox_pred.shape :',bbox_pred.shape)
    # pred_boxes = bbox_transform_inv_3d(rois, bbox_pred, 1)
    # print('pred_boxes.shape :',pred_boxes.shape)
    # pred_boxes = clip_boxes_3d(pred_boxes, im_info.data, 1)
    # print('pred_boxes.shape :',pred_boxes.shape)
    # rois = pred_boxes
    # print('h %d w %d ' % (h,w))
    # print('rois :',rois)
    # print('rois.shape :',rois.shape)
    # rois[:,[0,2]] =rois[:,[0,2]].clamp_(min=0, )
    # rois[:,[1,3]] =rois[:,[1,3]].clamp_(min=0,)
    # print('rois.shape :',rois.shape)
    # print('rois :',rois[0][0])

    # print('rois.shape :',rois.shape)
    # print('rois :',rois)

    colors = [ (255,0,0), (0,255,0), (0,0,255)]
    clips = clips.squeeze().permute(1,2,3,0)

    # print('rois.shape :',rois.shape)
    print('rois :',rois)
    rois = torch.round(rois)
    print('rois :',rois)
    for i in range(1):
        # img = cv2.imread(os.path.join(path, 'image_{:0>5}.jpg'.format(frame_indices[i])))
        # img = cv2.imread(os.path.join(path, '{:0>5}.png'.format(frame_indices[i])))
        img = clips[i].cpu().numpy()
        print(img.shape)
        img_tmp = img.copy()
        # if img.all():
        #     print('Image {} not found '.format(os.path.join(path, 'image_{:0>5}.jpg'.format(frame_indices[i]))))
        #     break

        for j in range(10):
            img_tmp = img.copy()
            cv2.rectangle(img_tmp,(int(rois[0,j,0]),int(rois[0,j,1])),(int(rois[0,j,3]),int(rois[0,j,4])), (255,0,0),3)

            # print('out : ./out/{:0>3}.jpg'.format(i))
            cv2.imwrite('./out_frames/action_rois_{}_{:0>3}.jpg'.format(j,i), img_tmp)
        # for j in range(10):
        #     cv2.rectangle(img_tmp,(int(gt_tubes_r[0,0,0]),int(gt_tubes_r[0,0,1])),(int(gt_tubes_r[0,0,3]),int(gt_tubes_r[0,0,4])), (0,255,0),3)
        # cv2.imwrite('./out_frames/both_{:0>3}.jpg'.format(i), img_tmp)
        # img2 = clips2[i].cpu().numpy()
        # img_tmp2 = img2.copy()
        # cv2.rectangle(img_tmp,(int(gt_tubes_r[0,0,0]),int(gt_tubes_r[0,0,1])),(int(gt_tubes_r[0,0,3]),int(gt_tubes_r[0,0,4])), (0,255,0),3)
        # cv2.imwrite('./out_frames/reg_{:0>3}.jpg'.format(i), img_tmp2)
