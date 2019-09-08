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
from box_functions import tube_overlaps

np.random.seed(42)

if __name__ == '__main__':

    # torch.cuda.device_count()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    dataset_folder = '../JHMDB-act-detector-frames'
    split_txt_path =  '../splits'
    boxes_file = '../poses.json'

    n_devs = torch.cuda.device_count()
    sample_size = 112
    # sample_duration = 16  # len(images)
    # sample_duration = 8  # len(images)
    sample_duration = 4  # len(images)

    batch_size = 1
    n_threads = 0

    # # get mean
    # mean =  [103.75581543 104.79421473  91.16894564] # jhmdb
    mean = [103.29825354, 104.63845484,  90.79830328]  # jhmdb from .png
    # mean = [112.07945832, 112.87372333, 106.90993363]  # ucf-101 24 classes

    # generate model
    last_fc = False

    # generate model
    actions = ['__background__','brush_hair', 'clap', 'golf', 'kick_ball', 'pour',
               'push', 'shoot_ball', 'shoot_gun', 'stand', 'throw', 'wave',
               'catch','climb_stairs', 'jump', 'pick', 'pullup', 'run', 'shoot_bow', 'sit',
               'swing_baseball', 'walk' ]

    cls2idx = {actions[i]: i for i in range(0, len(actions))}

    ### get videos id

    spatial_transform = Compose([Scale(sample_size),  # [Resize(sample_size),
                                 ToTensor(),
                                 Normalize(mean, [1, 1, 1])])
    temporal_transform = LoopPadding(sample_duration)

    n_classes = len(actions)

    # Init action_net

    model = ACT_net(actions, sample_duration)
    model.create_architecture()
    model = nn.DataParallel(model)
    model.to(device)

    model_data = torch.load('./action_net_model_jhmdb_4frm_64.pwf')

    # model.load_state_dict(model_data)
    model.eval()

    data = Video(dataset_folder, frames_dur=sample_duration, spatial_transform=spatial_transform,
                 temporal_transform=temporal_transform, json_file = boxes_file,
                 split_txt_path=split_txt_path, mode='train', classes_idx=cls2idx)

    clips, h, w, gt_tubes_r, gt_rois, n_actions, n_frames, im_info = data[14]
    clips2, h2, w2, gt_tubes_r2, gt_rois2, n_actions2, n_frames2, im_info2 = data[15]
    clips_ = clips.unsqueeze(0).to(device)
    gt_tubes_r_ = gt_tubes_r.unsqueeze(0).to(device)
    gt_rois_ = gt_rois.unsqueeze(0).to(device)
    n_actions_ = torch.Tensor(n_actions).to(device)
    im_info_ = im_info.unsqueeze(0).to(device)
    start_fr = torch.zeros((1,1)).to(device)

    clips_2 = clips2.unsqueeze(0).to(device)
    gt_tubes_r_2 = gt_tubes_r2.unsqueeze(0).to(device)
    gt_rois_2 = gt_rois2.unsqueeze(0).to(device)
    n_actions_2 = torch.Tensor(n_actions2).to(device)
    im_info_2 = im_info2.unsqueeze(0).to(device)
    start_fr_2 = torch.zeros((1,1)).to(device)

    # clips = torch.cat((clips_,clips_2))
    # gt_tubes_r_ = torch.cat((gt_tubes_r_, gt_tubes_r_2))
    # gt_rois_ = torch.cat((gt_rois_, gt_rois_2))
    # n_actions_ = torch.cat((n_actions_, n_actions_2))
    # start_fr = torch.cat((start_fr,start_fr_2))

    print('gt_tubes_r_.shape :',gt_tubes_r_.shape)
    print('gt_rois_.shape :',gt_rois_.shape)
    print('n_actions_.shape :',n_actions_.shape)
    print('start_fr.shape :',start_fr.shape)
    print('**********Starts**********')

    inputs = Variable(clips_)
    tubes, _, \
    _,  _, \
    _,\
    _,  _, \
    sgl_rois_bbox_pred, _,  = model(inputs, \
                   im_info_,
                   None, None,
                   None)

    print('**********VGIKE**********')
    print('tubes.shape :',tubes.shape)
    print('tubes :',tubes.cpu().numpy())
    exit(-1)
    print('sgl_rois_bbox_pred.shape :',sgl_rois_bbox_pred.shape)
    print('tubes[0,2] :',tubes[0].cpu().numpy())
    print('sgl_rois_bbox_pred[0,2] :',sgl_rois_bbox_pred[0,10])

    tubes_t = tubes[0,:,1:-1].contiguous()
    gt_rois_t = gt_rois_[0,:,:,:4].contiguous().view(-1,sample_duration*4)

    rois_overlaps = tube_overlaps(tubes_t,gt_rois_t)
    
    gt_max_overlaps_sgl, _ = torch.max(rois_overlaps, 0)
    print('gt_max_overlaps_sgl :',gt_max_overlaps_sgl)
    iou_thresh = 0.5
    gt_max_overlaps_sgl = torch.where(gt_max_overlaps_sgl > iou_thresh, gt_max_overlaps_sgl, torch.zeros_like(gt_max_overlaps_sgl).type_as(gt_max_overlaps_sgl))
    sgl_detected =  gt_max_overlaps_sgl.ne(0).sum()
    n_elems = gt_tubes_r_[0,:,-1].ne(0).sum().item()
    sgl_true_pos = sgl_detected
    sgl_false_neg = n_elems - sgl_detected

    recall = float(sgl_true_pos)  / (float(sgl_true_pos)  + float(sgl_false_neg))

    print(' -----------------------')
    print('| Validation Epoch: {: >3} | '.format(1))
    print('|                       |')
    print('| Proposed Action Tubes |')
    print('|                       |')
    print('| Single frame          |')
    print('|                       |')
    print('| In {: >6} steps    :  |'.format(1))
    print('|                       |')
    print('| True_pos   --> {: >6} |\n| False_neg  --> {: >6} | \n| Recall     --> {: >6.4f} |'.format(
        sgl_true_pos, sgl_false_neg, recall))


    print(' -----------------------')
