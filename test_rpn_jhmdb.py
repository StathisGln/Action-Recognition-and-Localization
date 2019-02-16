import os
import numpy as np
import glob
import json

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


    # generate model
    last_fc = False

    scale_size = [sample_size,sample_size]
    spatial_transform = Compose([Scale(sample_size),  # [Resize(sample_size),
                                 ToTensor(),
                                 Normalize(mean, [1, 1, 1])])
    temporal_transform = LoopPadding(sample_duration)

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


    # for i in range(700,850):
    #     k = data[i]
    #     print('abs path :',k[4], ' i :',i)
    
    
    # clips,  (h, w), gt_tubes, final_rois = data[906]
    # clips,  (h, w), gt_tubes, final_rois = data[905]
    clips, (h,w), gt_tubes,n_actions, path,frame_indices = data[90]
    clips = clips.unsqueeze(0).cuda()
    n_classes = len(classes)

    resnet_shortcut = 'A'

    model = resnet34(num_classes=400, shortcut_type=resnet_shortcut,
                     sample_size=sample_size, sample_duration=sample_duration,
                     last_fc=last_fc)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)

    model_data = torch.load('../temporal_localization/resnet-34-kinetics.pth')
    model.load_state_dict(model_data['state_dict'])
    model.eval()

    rpn_model = _RPN(256).cuda()
    rpn_data = torch.load('../temporal_localization/rpn_model_pre_000.pwf')
    rpn_model.load_state_dict(rpn_data)
    rpn_model.eval()

    inputs = Variable(clips).cuda()
    outputs = model(inputs)
    im_info = torch.Tensor([[112,112,16]]).cuda()
    rois, rpn_loss_cls, rpn_loss_box = rpn_model(outputs,
                                                 im_info,
                                                 None,None,None)
    print('rois :',rois.shape)
    print('rois :',rois)
    rois = rois[:,:,1:]
    print('bbox_pred.shape :',bbox_pred.shape)
    pred_boxes = bbox_transform_inv_3d(rois, bbox_pred, 1)
    print('pred_boxes.shape :',pred_boxes.shape)
    pred_boxes = clip_boxes_3d(pred_boxes, im_info.data, 1)
    print('pred_boxes.shape :',pred_boxes.shape)
    rois = pred_boxes[:,:,6:]
    print('h %d w %d ' % (h,w))
    rois[:,[0,2]] =rois[:,[0,2]].clamp_(min=0, )
    rois[:,[1,3]] =rois[:,[1,3]].clamp_(min=0,)
    print('rois.shape :',rois.shape)
    print('rois :',rois[0][0])

    # print('rois.shape :',rois.shape)
    # print('rois :',rois)

    colors = [ (255,0,0), (0,255,0), (0,0,255)]
    clips = clips.squeeze().permute(1,2,3,0)
    # print('rois.shape :',rois.shape)
    print('rois :',rois)
    rois = torch.round(rois)
    print('rois :',rois)
    for i in range(len(frame_indices)):
        # img = cv2.imread(os.path.join(path, 'image_{:0>5}.jpg'.format(frame_indices[i])))
        # img = cv2.imread(os.path.join(path, '{:0>5}.png'.format(frame_indices[i])))
        img = clips[i].cpu().numpy()
        print(img.shape)
        img_tmp = img.copy()
        # if img.all():
        #     print('Image {} not found '.format(os.path.join(path, 'image_{:0>5}.jpg'.format(frame_indices[i]))))
        #     break
        for j in range(3):
            cv2.rectangle(img_tmp,(int(rois[0,j,1]),int(rois[0,j,2])),(int(rois[0,j,3]),int(rois[0,j,4])), (255,0,0),3)

        # print('out : ./out/{:0>3}.jpg'.format(i))
        cv2.imwrite('./out_frames/action_{:0>3}.jpg'.format(i), img_tmp)
        # for j in range(10):
        #     cv2.rectangle(img_tmp,(int(gt_tubes_r[0,0,0]),int(gt_tubes_r[0,0,1])),(int(gt_tubes_r[0,0,3]),int(gt_tubes_r[0,0,4])), (0,255,0),3)
        # cv2.imwrite('./out_frames/both_{:0>3}.jpg'.format(i), img_tmp)
        # img2 = clips2[i].cpu().numpy()
        # img_tmp2 = img2.copy()
        # cv2.rectangle(img_tmp,(int(gt_tubes_r[0,0,0]),int(gt_tubes_r[0,0,1])),(int(gt_tubes_r[0,0,3]),int(gt_tubes_r[0,0,4])), (0,255,0),3)
        # cv2.imwrite('./out_frames/reg_{:0>3}.jpg'.format(i), img_tmp2)
