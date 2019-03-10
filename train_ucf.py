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
from video_dataset import video_names
from model import Model

torch.backends.cudnn.benchnark=True 
def bbox_overlaps_batch_3d(tubes, gt_tubes):

    """
    tubes: (N, 6) ndarray of float
    gt_tubes: (b, K, 5) ndarray of float

    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    batch_size = gt_tubes.size(0)

    if tubes.dim() == 2:

        N = tubes.size(0)
        K = gt_tubes.size(1)

        tubes = tubes[:,1:]
        tubes = tubes.view(1, N, 6)
        tubes = tubes.expand(batch_size, N, 6).contiguous()
        gt_tubes = gt_tubes[:, :, :6].contiguous()

        gt_tubes_x = (gt_tubes[:, :, 3] - gt_tubes[:, :, 0] + 1)
        gt_tubes_y = (gt_tubes[:, :, 4] - gt_tubes[:, :, 1] + 1)
        gt_tubes_t = (gt_tubes[:, :, 5] - gt_tubes[:, :, 2] + 1)

        if batch_size == 1:  # only 1 video in batch:
            gt_tubes_x = gt_tubes_x.unsqueeze(0)
            gt_tubes_y = gt_tubes_y.unsqueeze(0)
            gt_tubes_t = gt_tubes_t.unsqueeze(0)

        gt_tubes_area = (gt_tubes_x * gt_tubes_y * gt_tubes_t)
        gt_tubes_area_xy = gt_tubes_x * gt_tubes_y

        tubes_boxes_x = (tubes[:, :, 3] - tubes[:, :, 0] + 1)
        tubes_boxes_y = (tubes[:, :, 4] - tubes[:, :, 1] + 1)
        tubes_boxes_t = (tubes[:, :, 5] - tubes[:, :, 2] + 1)

        tubes_area = (tubes_boxes_x * tubes_boxes_y *
                        tubes_boxes_t).view(batch_size, N, 1)  # for 1 frame
        tubes_area_xy = (tubes_boxes_x * tubes_boxes_y).view(batch_size, N, 1)  # for 1 frame
        
        gt_area_zero = (gt_tubes_x == 1) & (gt_tubes_y == 1) & (gt_tubes_t == 1)
        tubes_area_zero = (tubes_boxes_x == 1) & (tubes_boxes_y == 1) & (tubes_boxes_t == 1)

        gt_area_zero_xy = (gt_tubes_x == 1) & (gt_tubes_y == 1) 
        tubes_area_zero_xy = (tubes_boxes_x == 1) & (tubes_boxes_y == 1) 

        gt_area_zero_t =  (gt_tubes_t == 1)
        tubes_area_zero_t =  (tubes_boxes_t == 1)

        boxes = tubes.view(batch_size, N, 1, 6)
        boxes = boxes.expand(batch_size, N, K, 6)
        query_boxes = gt_tubes.view(batch_size, 1, K, 6)
        query_boxes = query_boxes.expand(batch_size, N, K, 6)

        iw = (torch.min(boxes[:, :, :, 3], query_boxes[:, :, :, 3]) -
              torch.max(boxes[:, :, :, 0], query_boxes[:, :, :, 0]) + 1)

        iw[iw < 0] = 0

        ih = (torch.min(boxes[:, :, :, 4], query_boxes[:, :, :, 4]) -
              torch.max(boxes[:, :, :, 1], query_boxes[:, :, :, 1]) + 1)
        ih[ih < 0] = 0

        it = (torch.min(boxes[:, :, :, 5], query_boxes[:, :, :, 5]) -
              torch.max(boxes[:, :, :, 2], query_boxes[:, :, :, 2]) + 1)
        it[it < 0] = 0

        ua = tubes_area + gt_tubes_area - (iw * ih * it)
        ua_xy = tubes_area_xy + gt_tubes_area_xy - (iw * ih )
        ua_t = tubes_boxes_t.unsqueeze(2) + gt_tubes_t - it

        # print('ua.shape :',ua.shape)
        # print('ua_xy.shape :',ua_xy.shape)
        # print('tubes_boxes_t.shape :',tubes_boxes_t.shape)
        # print('gt_tubes_t.shape :',gt_tubes_t.shape)
        # print('it :',it.shape)
        # print('ua_t :',ua_t.shape)
        # print('tubes_area.shape :',tubes_area.shape)
        # print('tubes_boxes_t.shape :',tubes_boxes_t.unsqueeze(2)
        #       .shape)
        # print('gt_tubes_area.shape :',gt_tubes_area.shape)
        # print('gt_tubes_t.shape :', gt_tubes_t.shape)
        overlaps = iw * ih * it / ua
        overlaps_xy = iw * ih  / ua_xy
        overlaps_t = it / ua_t

        overlaps.masked_fill_(gt_area_zero.view(
            batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps.masked_fill_(tubes_area_zero.view(
            batch_size, N, 1).expand(batch_size, N, K), -1)

        overlaps_xy.masked_fill_(gt_area_zero.view(
            batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps_xy.masked_fill_(tubes_area_zero.view(
            batch_size, N, 1).expand(batch_size, N, K), -1)

        overlaps_t.masked_fill_(gt_area_zero.view(
            batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps_t.masked_fill_(tubes_area_zero.view(
            batch_size, N, 1).expand(batch_size, N, K), -1)

    else:
        raise ValueError('tubes input dimension is not correct.')

    return overlaps, overlaps_xy, overlaps_t


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
        
        vid_id, boxes, n_frames, n_actions = data
        
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
    
    ###################################
    #        JHMDB data inits         #
    ###################################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    dataset_folder = '/gpu-data2/sgal/UCF-101-frames'
    boxes_file = '/gpu-data2/sgal/pyannot.pkl'
    spt_path = '/gpu-data2/sgal/UCF101_Action_detection_splits/'

    sample_size = 112
    sample_duration = 16  # len(images)

    batch_size = 1

    # # get mean
    mean = [112.07945832, 112.87372333, 106.90993363]  # ucf-101 24 classes

    # generate model
    actions = ['__background__', 'Basketball','BasketballDunk','Biking','CliffDiving','CricketBowling',
               'Diving','Fencing','FloorGymnastics','GolfSwing','HorseRiding','IceDancing',
               'LongJump','PoleVault','RopeClimbing','SalsaSpin','SkateBoarding','Skiing',
               'Skijet','SoccerJuggling','Surfing','TennisSwing','TrampolineJumping',
               'VolleyballSpiking','WalkingWithDog']

    cls2idx = {actions[i]: i for i in range(0, len(actions))}

    ### get videos id
    vid2idx,vid_names = get_vid_dict(dataset_folder)

    spatial_transform = Compose([Scale(sample_size),  # [Resize(sample_size),
                                 ToTensor(),
                                 Normalize(mean, [1, 1, 1])])
    temporal_transform = LoopPadding(sample_duration)

    n_classes = len(actions)


    ##########################################
    #          Model Initialization          #
    ##########################################

    model = Model(actions, sample_duration, sample_size)
    model.create_architecture()

    # if torch.cuda.device_count() > 1:

    #     print('Using {} GPUs!'.format(torch.cuda.device_count()))
    #     model = nn.DataParallel(model)

    model.to(device)

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

    vid_name_loader = video_names(dataset_folder, spt_path, boxes_file, vid2idx, mode='train')
    data_loader = torch.utils.data.DataLoader(vid_name_loader, batch_size=batch_size,
                                              shuffle=True)

    # epochs = 40
    epochs = 5

    n_devs = torch.cuda.device_count()
    if torch.cuda.device_count() > 1:

        print('Using {} GPUs!'.format(torch.cuda.device_count()))
        model.act_net = nn.DataParallel(model.act_net)

    model.act_net = model.act_net.cuda()

    # if torch.cuda.device_count() > 1:

    #     print('Using {} GPUs!'.format(torch.cuda.device_count()))
    #     model.module.act_net = nn.DataParallel(model.module.act_net)

    # model.module.act_net = model.module.act_net.cuda()


    for ep in range(epochs):

        model.train()
        loss_temp = 0

        # start = time.time()
        if (ep +1) % (lr_decay_step ) == 0:
            print('time to reduce learning rate ')
            adjust_learning_rate(optimizer, lr_decay_gamma)
            lr *= lr_decay_gamma


        print(' ============\n| Epoch {:0>2}/{:0>2} |\n ============'.format(ep+1, epochs))
        for step, data  in enumerate(data_loader):

            # if step == 2:
            #     break
            print('step :',step)
            vid_id, boxes, n_frames, n_actions, h, w = data
            
            ###################################
            #          Function here          #
            ###################################

            mode = 'train'
            # boxes_ = boxes.to(device)
            vid_id_ = vid_id.to(device)
            n_frames_ = n_frames.to(device)
            n_actions_ = n_actions.to(device)

            tubes,  bbox_pred, \
            prob_out, rpn_loss_cls, \
            rpn_loss_bbox, act_loss_bbox,  cls_loss =  model(n_devs, dataset_folder, \
                                                             vid_names, vid_id_, spatial_transform, \
                                                             temporal_transform, boxes, \
                                                             mode, cls2idx, n_actions_,n_frames_)

            loss = rpn_loss_cls.mean() + rpn_loss_bbox.mean() + act_loss_bbox.mean() + cls_loss.mean()

            # backw\ard
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_temp += loss.item()

        print('Train Epoch: {} \tLoss: {:.6f}\t lr : {:.6f}'.format(
        ep+1,loss_temp/step, lr))
        
        if ( ep + 1 ) % 5 == 0: # validation time
            val_name_loader = video_names(dataset_folder, spt_path, boxes_file, vid2idx, mode='test')
            val_loader = torch.utils.data.DataLoader(val_name_loader, batch_size=batch_size,
                                              shuffle=True)

            validate_model(model, val_name_loader, val_loader)
        # if ( ep + 1 ) % 5 == 0:
        torch.save(model.state_dict(), "model.pwf")
    # torch.save(model.state_dict(), "model.pwf")


