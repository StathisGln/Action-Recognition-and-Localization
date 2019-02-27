import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, ToTensor, Resize)
from temporal_transforms import LoopPadding
from simple_dataset import Video
from net_utils import adjust_learning_rate
from resize_rpn import resize_rpn, resize_tube
from create_tubes_from_boxes import  create_tube
from resnet_3D import resnet34
from roi_align_3d.modules.roi_align  import RoIAlignAvg
from tcn import TCN
from tcn_net import tcn_net
# torch.set_default_tensor_type('torch.cuda.FloatTensor')

def validate_tcn(model,  val_data, val_data_loader):

    ###
    max_dim = 1
    correct = 0

    for step, data  in enumerate(val_data_loader):

        # if step == 2:
        #     break
        clips,  (h, w), target, boxes, n_frames = data

        clips = clips.to(device)
        boxes = boxes.to(device)
        n_actions = torch.Tensor([[1]]).to(device)
        im_info = torch.Tensor([[sample_size, sample_size, n_frames]] ).to(device)    
        target = target.to(device)
        gt_tubes_r, gt_rois = preprocess_data(device, clips, n_frames, boxes, h, w, sample_size, sample_duration,target, 'train')


        output, tcn_loss = model( clips, target, gt_tubes_r, n_frames, max_dim=1)
        _, cls = torch.max(output,1)
        print('cls :', cls ,' target :', target)
        if cls == target :
            correct += 1

    print(' ------------------- ')
    print('|  In {: >6} steps  |'.format(step))
    print('|                   |')
    print('|  Correct : {: >6} |'.format(correct))
    print(' ------------------- ')
    

def preprocess_data(device, clip, n_frames, gt_bboxes, h, w, sample_size, sample_duration, target, mode):


    gt_bboxes[:,:,[0,2]] = gt_bboxes[:,:,[0,2]].clamp_(min=0, max=w.item())
    gt_bboxes[:,:,[1,3]] = gt_bboxes[:,:,[1,3]].clamp_(min=0, max=h.item())

    gt_bboxes = torch.round(gt_bboxes)
    gt_bboxes_r = resize_rpn(gt_bboxes, h.item(),w.item(),sample_size)

    ## add gt_bboxes_r class_int
    target = target.unsqueeze(0).unsqueeze(2)
    target = target.expand(1, gt_bboxes_r.size(1), 1).type_as(gt_bboxes)

    gt_bboxes_r = torch.cat((gt_bboxes_r[:,:, :4],target),dim=2).type_as(gt_bboxes)
    im_info_tube = torch.Tensor([[w,h,]*gt_bboxes_r.shape[0]]).to(device)

    if mode == 'train' or mode == 'val':

        if n_frames < 17:
            indexes = [0]
        else:
            indexes = range(0, (n_frames -sample_duration  ), int(sample_duration/2))

        gt_tubes = torch.zeros((gt_bboxes.size(0),len(indexes),7)).to(device)
        # print('n_frames :',n_frames)
        for i in indexes :
            lim = min(i+sample_duration, (n_frames.item()-1))
            # print('lim :', lim)
            vid_indices = torch.arange(i,lim).long().to(device)
            # print('vid_indices :',vid_indices)

            gt_rois_seg = gt_bboxes_r[:, vid_indices]
            gt_tubes_seg = create_tube(gt_rois_seg.unsqueeze(0),torch.Tensor([[sample_size,sample_size]]).type_as(gt_bboxes), sample_duration)
            # print('gt_tubes_seg.shape :',gt_tubes_seg.shape)
            gt_tubes_seg[:,:,2] = i
            gt_tubes_seg[:,:,5] = i+sample_duration-1
            gt_tubes[0,int(i/sample_duration*2)] = gt_tubes_seg


            gt_tubes = torch.round(gt_tubes)

    else:
        gt_tubes =  create_tube(np.expand_dims(gt_bboxes_r,0), np.array([[sample_size,sample_size]]), n_frames)                

    f_rois = torch.zeros((1,n_frames,5)).type_as(gt_bboxes) # because only 1 action can have simultaneously
    b_frames = gt_bboxes_r.size(1)

    f_rois[:,:b_frames,:] = gt_bboxes_r

    # print('gt_tubes :',gt_tubes)
    # if (n_frames < 16):
    #     print(f_rois)

    if (b_frames != n_frames):
        print('\n LATHOSSSSSS\n', 'n_frames :', n_frames, ' b_frames :',b_frames)
        exit(1)
    # print('f_rois.shape :',f_rois.shape)
    # print('gt_tubes :',gt_tubes)
    # print(gt_bboxes)
    return gt_tubes, f_rois

if __name__ == '__main__':
    
    ###################################
    #        JHMDB data inits         #
    ###################################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    dataset_folder = '/gpu-data2/sgal/JHMDB-act-detector-frames'
    splt_txt_path  = '/gpu-data2/sgal/splits'
    boxes_file     = '/gpu-data2/sgal/poses.json'

    sample_size = 112
    sample_duration = 16  # len(images)

    batch_size = 1
    n_threads = 0

    # # get mean
    mean = [103.29825354, 104.63845484,  90.79830328]  # jhmdb from .png

    # generate model
    classes = ['__background__', 'brush_hair', 'clap', 'golf', 'kick_ball', 'pour',
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

    ###################################
    #          TCN valiables          #
    ###################################
    
    # input_channels = 512
    input_channels = 256
    nhid = 25 ## number of hidden units per levels
    levels = 3
    channel_sizes = [nhid] * levels
    kernel_size = 2 # 7
    dropout = 0.05

    ######################################
    #          TCN initilzation          #
    ######################################

    model = tcn_net(classes, sample_duration, sample_size, input_channels, channel_sizes, kernel_size=kernel_size, dropout=dropout)

    model.create_architecture()

    model_data = torch.load('./tcn_model.pwf')
    model.load_state_dict(model_data)
    model.to(device)
    val_data = Video(dataset_folder, frames_dur=sample_duration, spatial_transform=spatial_transform,
                 temporal_transform=temporal_transform, json_file = boxes_file,
                 split_txt_path=splt_txt_path, mode='val', classes_idx=cls2idx)
    val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,
                                              shuffle=True, num_workers=n_threads, pin_memory=True)

    validate_tcn(model, val_data, val_data_loader)
    #     if ( ep + 1 ) % 5 == 0:
    #         torch.save(tcn_net.state_dict(), "tcn_model.pwf")
    # torch.save(tcn_net.state_dict(), "tcn_model.pwf")


