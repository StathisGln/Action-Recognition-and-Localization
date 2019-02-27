import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, ToTensor, Resize)
from temporal_transforms import LoopPadding
from jhmdb_dataset import Video

from resnet_3D import resnet34
from roi_align_3d.modules.roi_align  import RoIAlignAvg
from tcn import TCN

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


def train_tcn(clips, target, sample_duration, n_frames, max_dim):

    indexes = range(0, (n_frames - sample_duration  ), int(sample_duration/2))
    features = torch.zeros(1,512,len(indexes)).type_as(clips)
    rois = torch.zeros(max_dim, 7).type_as(clips)
    for i in indexes:

        lim = min(i+sample_duration, (n_frames))
        vid_indices = torch.arange(i,lim).long()
        rois[:,1:] = gt_tubes_r[:,int(i*2/sample_duration),:6]
        vid_seg = clips[:,:,vid_indices]

        outputs = model(vid_seg)

        pooled_feat = roi_align(outputs,rois)

        fc7 = top_part(pooled_feat)
        fc7 = fc7.mean(4)
        fc7 = fc7.mean(3)
        fc7 = fc7.mean(2)

        features[0,:,int(i*2/sample_duration)] = fc7
        
    output = tcn_net(features)
    output = F.softmax(output, 1)
    tcn_loss = F.cross_entropy(output, target.long())
    print(tcn_loss)
    return output, tcn_loss
    


if __name__ == '__main__':
    
    ###################################
    #        JHMDB data inits         #
    ###################################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

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
    # data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
    #                                           shuffle=True, num_workers=n_threads, pin_memory=True)

    n_classes = len(classes)

    ###################################
    #          TCN valiables          #
    ###################################
    
    input_channels = 512
    nhid = 25 ## number of hidden units per levels
    levels = 8
    channel_sizes = [nhid] * levels
    kernel_size = 3
    dropout = 0.05

    roi_align = RoIAlignAvg(7, 7, 16, 1.0/16.0, 1.0)
    tcn_net = TCN(input_channels, n_classes, channel_sizes, kernel_size = kernel_size, dropout=dropout)

    tcn_net = tcn_net.to(device)
    ######################################
    #          Resnet Variables          #
    ######################################

    sample_size = 112
    sample_duration = 16 #len(images)

    batch_size = 16
    n_threads = 1

    # get mean
    mean =  [114.7748, 107.7354, 99.4750]

    # generate model
    last_fc = False
    n_classes = 400
    resnet_shortcut = 'A'
    
    model = resnet34(num_classes=n_classes, shortcut_type=resnet_shortcut,
                     sample_size=sample_size, sample_duration=sample_duration,
                     last_fc=last_fc)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)

    model_data = torch.load('/gpu-data2/sgal/resnet-34-kinetics.pth')
    model.load_state_dict(model_data['state_dict'])

    ######################################
    #          Code starts here          #
    ######################################

    clips, (h, w), gt_tubes_r, gt_rois, n_actions, n_frames = data[161]

    im_info = torch.Tensor([[sample_size, sample_size, n_frames]] ).to(device)
    clips = clips.unsqueeze(0).to(device)
    gt_tubes_r = gt_tubes_r.unsqueeze(0).to(device)
    gt_rois = gt_rois.unsqueeze(0).to(device)
    n_actions = n_actions.unsqueeze(0).to(device)
    
    print('n_actions :',n_actions)
    print('clips :',clips.shape)
    print('gt_tubes :',gt_tubes_r)
    print('gt_tubes.shape :',gt_tubes_r.shape)

    print('im_info :',im_info)
    print('im_info.shape :',im_info.shape)

    print('n_actions :',n_actions)
    print('n_actions.shape :',n_actions.shape)

    top_part = nn.Sequential(model.module.layer4)
    max_dim = 1
    ###################################
    #          Function here          #
    ###################################
    target = gt_tubes_r[:,0,6]
    print('gt_tubes :',gt_tubes_r)
    print('target :',target)
    output, tcn_loss = train_tcn(clips, target, sample_duration, n_frames, max_dim)

    print('tcn_loss :', tcn_loss)
