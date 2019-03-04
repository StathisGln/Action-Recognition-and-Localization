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

def validate_tcn(model,  val_data, val_data_loader, n_classes):

    ###
    model.eval()
    max_dim = 1
    correct = 0

    for step, data  in enumerate(val_data_loader):

        # if step == 2:
        #     break
        clips,  (h, w), target, gt_tubes, im_info, n_frames = data
            
        clips_t = clips.to(device)
        gt_tubes_t = gt_tubes.to(device)
        im_info_t = im_info.to(device)
        n_frames_t = n_frames.to(device)
        target_t = target.to(device)
        out_prob, tcn_loss = model( clips_t, None, gt_tubes_t, n_frames, max_dim=1)
        output = out_prob.view(-1, n_classes)

        _, cls = torch.max(output,1)

        for i in range(len(target)):
            if cls[i] == target_t[i] :
                correct += 1

    print(' ------------------- ')
    print('|  In {: >6} steps  |'.format(step))
    print('|                   |')
    print('|  Correct : {: >6} |'.format(correct))
    print(' ------------------- ')
    


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

    batch_size = 2
    n_threads = 2

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

    ######################################
    #          TCN initilzation          #
    ######################################

    model = tcn_net(classes, sample_duration, sample_size)

    model.create_architecture()

    if torch.cuda.device_count() > 1:
        print('Using {} GPUs!'.format(torch.cuda.device_count()))

        model = nn.DataParallel(model)

    model_data = torch.load('./tcn_model.pwf')
    model.load_state_dict(model_data)

    model.to(device)

    val_data = Video(dataset_folder, frames_dur=sample_duration, spatial_transform=spatial_transform,
                 temporal_transform=temporal_transform, json_file = boxes_file,
                 split_txt_path=splt_txt_path, mode='val', classes_idx=cls2idx)
    val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,
                                              shuffle=True, num_workers=n_threads, pin_memory=True)

    validate_tcn(model, val_data, val_data_loader, len(classes))


