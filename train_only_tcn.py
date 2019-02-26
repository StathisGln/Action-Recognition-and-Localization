import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, ToTensor, Resize)
from temporal_transforms import LoopPadding
from jhmdb_dataset import Video
from net_utils import adjust_learning_rate

from resnet_3D import resnet34
from roi_align_3d.modules.roi_align  import RoIAlignAvg
from tcn import TCN

def train_tcn(clips, target, sample_duration, n_frames, max_dim):

    if n_frames < 17:
        indexes = [0]
    else:
        indexes = range(0, (n_frames.data - sample_duration  ), int(sample_duration/2))

    features = torch.zeros(1,512,len(indexes)).type_as(clips)
    rois = torch.zeros(max_dim, 7).type_as(clips)
    for i in indexes:

        lim = min(i+sample_duration, (n_frames.item()))
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
    # print('output :',output)
    # print('output.shape :',output.shape)
    tcn_loss = F.cross_entropy(output, target.unsqueeze(0).long())
    # print(tcn_loss)
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
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                              shuffle=True, num_workers=n_threads, pin_memory=True)

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

    top_part = nn.Sequential(model.module.layer4)
    max_dim = 1

    lr = 0.1
    lr_decay_step = 10
    lr_decay_gamma = 0.1

    params = []
    for key, value in dict(tcn_net.named_parameters()).items():
        # print(key, value.requires_grad)
        if value.requires_grad:
            print('key :',key)
            if 'bias' in key:
                params += [{'params':[value],'lr':lr*(True + 1), \
                            'weight_decay': False and 0.0005 or 0}]
            else:
                params += [{'params':[value],'lr':lr, 'weight_decay': 0.0005}]

    lr = lr * 0.1
    optimizer = torch.optim.Adam(params)


    ######################################
    #          Code starts here          #
    ######################################

    epochs = 40
    
    for ep in range(epochs):

        tcn_net.train()
        model.eval()
        loss_temp = 0

        # start = time.time()
        if ep % (lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, lr_decay_gamma)
            lr *= lr_decay_gamma


        print(' ============\n| Epoch {:0>2}/{:0>2} |\n ============'.format(ep+1, epochs))
        for step, data  in enumerate(data_loader):

            # if step == 2:
            #     break

            clips,  (h, w), gt_tubes_r, gt_rois, n_actions, n_frames = data

            # print('gt_tubes_r :',gt_tubes_r)
            # print('gt_tubes_r.shape :',gt_tubes_r.shape)
            clips = clips.to(device)
            gt_tubes_r = gt_tubes_r.to(device)
            gt_rois = gt_rois.to(device)
            n_actions = n_actions.to(device)
            im_info = torch.Tensor([[sample_size, sample_size, n_frames]] ).to(device)    

            ###################################
            #          Function here          #
            ###################################
            target = gt_tubes_r[0,0,6]
            output, tcn_loss = train_tcn(clips, target, sample_duration, n_frames, max_dim)

            loss = tcn_loss.mean()

            loss_temp += loss.item()

            # backw\ard
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Train Epoch: {} \tLoss: {:.6f}\t lr : {:.6f}'.format(
        ep+1,loss_temp/step, lr))

        if ( epoch + 1 ) % 5 == 0:
            torch.save(tcn_net.state_dict(), "tcn_model.pwf".format(epoch+1))
    torch.save(tcn_net.state_dict(), "tcn_model.pwf".format(epoch))

