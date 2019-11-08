import os
import numpy as np
import glob

from  tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from jhmdb_dataset import  video_names

from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, ToTensor, Resize)
from temporal_transforms import LoopPadding
# from model_par_all import Model
# from model_par_all_v2 import Model
from model_par_all_v3 import Model
from create_video_id import get_vid_dict
from resize_rpn import resize_rpn, resize_tube
import pdb

np.random.seed(42)

if __name__ == '__main__':

    # torch.cuda.device_count()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    dataset_frames = '../JHMDB-act-detector-frames'
    split_txt_path =  '../splits'
    boxes_file = '../poses.json'

    n_devs = torch.cuda.device_count()
    sample_size = 112
    # sample_duration = 16  # len(images)
    sample_duration = 8  # len(images)

    batch_size = 1
    n_threads = 0

    # # get mean
    # mean =  [103.75581543 104.79421473  91.16894564] # jhmdb
    # mean = [103.29825354, 104.63845484,  90.79830328]  # jhmdb from .png
    mean = [103.29825354, 104.63845484,  90.79830328]  # jhmdb from .png

    # generate model
    last_fc = False


    actions = ['__background__','brush_hair', 'clap', 'golf', 'kick_ball', 'pour',
               'push', 'shoot_ball', 'shoot_gun', 'stand', 'throw', 'wave',
               'catch','climb_stairs', 'jump', 'pick', 'pullup', 'run', 'shoot_bow', 'sit',
               'swing_baseball', 'walk' ]


    cls2idx = {actions[i]: i for i in range(0, len(actions))}

    ### get videos id
    vid2idx,vid_names = get_vid_dict(dataset_frames)

    spatial_transform = Compose([Scale(sample_size),  # [Resize(sample_size),
                                 ToTensor(),
                                 Normalize(mean, [1, 1, 1])])
    temporal_transform = LoopPadding(sample_duration)

    n_classes = len(actions)


    # Init action_net
    action_model_path = './action_net_model_8frm_2_avg_jhmdb.pwf'

    model = Model(actions, sample_duration, sample_size)
    model.load_part_model(action_model_path=action_model_path)

    model = nn.DataParallel(model)
    model.to(device)

    vid_name_loader = video_names(dataset_frames, split_txt_path, boxes_file, vid2idx, mode='train', classes_idx=cls2idx)
    data_loader = torch.utils.data.DataLoader(vid_name_loader, batch_size=batch_size,
                                              shuffle=True)

    # vid_id, clips, boxes, n_frames, n_actions, h, w, target =vid_name_loader[14]
    # vid_id_2, clips_2, boxes_2, n_frames_2, n_actions_2, h_2, w_2,target_2 =vid_name_loader[277]
    # vid_id_2, clips_2, boxes_2, n_frames_2, n_actions_2, h_2, w_2,target =vid_name_loader[209]

    vid_id, clips, boxes, n_frames, n_actions, h, w, target =vid_name_loader[1]
    vid_id_2, clips_2, boxes_2, n_frames_2, n_actions_2, h_2, w_2,target_2 =vid_name_loader[0]


    # vid_id = torch.Tensor(vid_id).unsqueeze(0).int()
    # clips = clips.unsqueeze(0).to(device)
    # boxes = torch.from_numpy(boxes).unsqueeze(0).to(device)
    # n_frames = torch.from_numpy(n_frames).unsqueeze(0).to(device)
    # n_actions = torch.from_numpy(n_actions).int().unsqueeze(0).to(device)
    # im_info = torch.Tensor([h,w,clips.size(2)]).unsqueeze(0).to(device)
    # target = torch.Tensor([target]).unsqueeze(0).to(device)

    vid_id = torch.Tensor(vid_id).int()
    clips = clips.to(device)
    boxes = torch.from_numpy(boxes).to(device)
    n_frames = torch.from_numpy(n_frames).to(device)
    n_actions = torch.from_numpy(n_actions).int().to(device)
    im_info = torch.Tensor([h,w,clips.size(2)]).to(device)
    target = torch.Tensor([target]).to(device)

    vid_id_2 = torch.Tensor(vid_id_2).int()
    clips_2 = clips_2.to(device)
    boxes_2 = torch.from_numpy(boxes_2).to(device)
    n_frames_2 = torch.from_numpy(n_frames_2).to(device)
    n_actions_2 = torch.from_numpy(n_actions_2).int().to(device)
    im_info_2 = torch.Tensor([h_2,w_2,clips_2.size(2)]).to(device)
    target_2 = torch.Tensor([target_2]).to(device)

    vid_id = torch.stack([vid_id, vid_id_2])
    clips = torch.stack([clips, clips_2])
    boxes = torch.stack([boxes, boxes_2])
    n_frames = torch.stack([n_frames, n_frames_2])
    n_actions = torch.stack([n_actions, n_actions_2])
    im_info = torch.stack([im_info, im_info_2])
    target = torch.stack([target,target_2])

    print('boxes.shape :',boxes.shape)
    print('clips.shape :',clips.shape)
    print('n_frames.shape :',n_frames.shape)
    print('n_frames.shape :',n_frames)
    print('n_actions.shape :',n_actions.shape)
    print('im_info :',im_info.shape)
    print('target :',target)

    mode = 'train'
    print('**********Starts**********')
    with torch.no_grad():
        # model.eval()
        tubes,  \
        prob_out, cls_loss =  model(n_devs, dataset_frames, \
                                    vid_names, clips, vid_id,  \
                                    boxes, \
                                    mode, cls2idx, n_actions,n_frames, h, w,
                                    target)

    # rois,  bbox_pred, cls_prob, \
    # rpn_loss_cls,  rpn_loss_bbox, \
    # act_loss_cls,  act_loss_bbox, rois_label = model(clips,
    #                                                  im_info,
    #                                                  gt_tubes, None,
    #                                                  n_actions)

    print('**********VGIKE**********')
    print('rois.shape :',tubes.shape)
    print('rois :',tubes)

