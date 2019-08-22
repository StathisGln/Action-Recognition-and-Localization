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

from net_utils import adjust_learning_rate

from create_video_id import get_vid_dict
from jhmdb_dataset import  video_names, RNN_JHMDB

from model import Model
from resize_rpn import resize_tube

# torch.backends.cudnn.benchnark=True 


def validate_model(model,  val_data, val_data_loader):

    ###
    max_dim = 1

    correct = 0

    true_pos = torch.zeros(1).long().cuda()
    false_neg = torch.zeros(1).long().cuda()

    true_pos_xy = torch.zeros(1).long().cuda()
    false_neg_xy = torch.zeros(1).long().cuda()

    true_pos_t = torch.zeros(1).long().cuda()
    false_neg_t = torch.zeros(1).long().cuda()

    n_preds = torch.zeros(1).long().to(device)
    preds = torch.zeros(1).long().to(device)
    ## 2 rois : 1450

    for step, data  in enumerate(val_data_loader):

        if step == 2:
            break
        
        vid_id, clips, boxes, n_frames, n_actions, h, w =data
        
        mode = 'test'
        boxes_ = boxes.cuda()
        vid_id_ = vid_id.cuda()
        n_frames_ = n_frames.cuda()
        n_actions_ = n_actions.cuda()

        ## create video tube
        video_tubes = create_video_tube(boxes.type_as(clips_))
        video_tubes_r =  resize_tube(video_tubes.unsqueeze(0), h_,w_,self.sample_size)

        tubes,  bbox_pred, \
        prob_out, rpn_loss_cls, \
        rpn_loss_bbox, act_loss_bbox,  cls_loss =  model(n_devs, dataset_folder, \
                                                         vid_names, vid_id_, spatial_transform, \
                                                         temporal_transform, boxes_, \
                                                         mode, cls2idx, n_actions_,n_frames_)


        # _, cls = torch.max(prob_out,1)

        # if cls == target :
        #     correct += 1

    print(' ------------------- ')
    print('|  In {: >6} steps  |'.format(step))
    print('|                   |')
    print('|  Correct : {: >6} |'.format(correct))
    print(' ------------------- ')

if __name__ == '__main__':
    
    #################################
    #        UCF data inits         #
    #################################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    dataset_frames = '../JHMDB-act-detector-frames'
    split_txt_path =  '../splits'
    boxes_file = '../poses.json'

    sample_size = 112
    sample_duration = 16  # len(images)

    batch_size = 1

    # # # get mean
    mean = [103.29825354, 104.63845484,  90.79830328]  # jhmdb from .png
    # mean = [112.07945832, 112.87372333, 106.90993363]  # ucf-101 24 classes

    # generate model
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


    ##########################################
    #          Model Initialization          #
    ##########################################

    model = Model(actions, sample_duration, sample_size)
    model.load_part_model()
    model.deactivate_action_net_grad()

    # if torch.cuda.device_count() > 1:

    #     print('Using {} GPUs!'.format(torch.cuda.device_count()))
    #     model.act_net = nn.DataParallel(model.act_net)

    # model.to(device)
    # model.act_net.to(device)

    lr = 0.1
    lr_decay_step = 5
    lr_decay_gamma = 0.1

    params = []
    for key, value in dict(model.named_parameters()).items():
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
    # optimizer = optim.SGD(tcn_net.parameters(), lr = lr)

    #######################################
    #          Train starts here          #
    #######################################

    vid_name_loader = video_names(dataset_frames, split_txt_path, boxes_file, vid2idx, mode='train', classes_idx=cls2idx)
    data_loader = torch.utils.data.DataLoader(vid_name_loader, batch_size=batch_size,
                                              shuffle=True)

    epochs = 60
    # epochs = 5

    n_devs = torch.cuda.device_count()
    if torch.cuda.device_count() > 1:

        print('Using {} GPUs!'.format(torch.cuda.device_count()))
        # model.act_net = nn.DataParallel(model.act_net)
        model = nn.DataParallel(model)

    model.to(device)

    for ep in range(epochs):

        model.train()
        loss_temp = 0

        # start = time.time()
        if (ep +1) % (lr_decay_step ) == 0:
            print('time to reduce learning rate ', lr)
            adjust_learning_rate(optimizer, lr_decay_gamma)
            lr *= lr_decay_gamma


        print(' ============\n| Epoch {:0>2}/{:0>2} |\n ============'.format(ep+1, epochs))
        for step, data  in enumerate(data_loader):

            # if step == 2:
            #     break

            # print('step :',step)
            vid_id, clips, boxes, n_frames, n_actions, h, w, target =data
            ###################################
            #          Function here          #
            ###################################

            mode = 'train'
            # boxes_ = boxes.to(device)
            vid_id = vid_id.int()
            clips = clips.to(device)
            boxes = boxes.to(device)
            n_frames = n_frames.to(device)
            n_actions = n_actions.to(device)
            h = h.to(device)
            w = w.to(device)

            tubes,   \
            prob_out, cls_loss =  model(n_devs, dataset_frames, \
                                        vid_names, clips, vid_id, \
                                        boxes, \
                                        mode, cls2idx, n_actions,n_frames, h, w)

            loss =  cls_loss.mean()

            # backw\ard
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_temp += loss.item()

        print('Train Epoch: {} \tLoss: {:.6f}\t lr : {:.6f}'.format(
        ep+1,loss_temp/step, lr))
        

        # if ( ep + 1 ) % 5 == 0: # validation time
        #     val_name_loader = video_names(dataset_frames, spt_path, boxes_file, vid2idx, mode='test')
        #     val_loader = torch.utils.data.DataLoader(val_name_loader, batch_size=batch_size,
        #                                       shuffle=True)

        #     validate_model(model, val_name_loader, val_loader)
        # if ( ep + 1 ) % 5 == 0:
        torch.save(model.state_dict(), "model_linear.pwf")
    # torch.save(model.state_dict(), "model.pwf")


