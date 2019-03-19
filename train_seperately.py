import os
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from ucf_dataset import Video_UCF, video_names

from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, ToTensor, Resize)
from temporal_transforms import LoopPadding

from create_video_id import get_vid_dict
from net_utils import adjust_learning_rate
from resize_rpn import resize_rpn, resize_tube

from model import Model
from action_net import ACT_net
from resize_rpn import resize_boxes
import pdb

np.random.seed(42)


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

        tubes = tubes[:,1:7]
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

        tubes_boxes_x = (tubes[:, :, 3] - tubes[:, :, 0] + 1)
        tubes_boxes_y = (tubes[:, :, 4] - tubes[:, :, 1] + 1)
        tubes_boxes_t = (tubes[:, :, 5] - tubes[:, :, 2] + 1)

        tubes_area = (tubes_boxes_x * tubes_boxes_y *
                        tubes_boxes_t).view(batch_size, N, 1)  # for 1 frame
        gt_area_zero = (gt_tubes_x == 1) & (gt_tubes_y == 1) 
        tubes_area_zero = (tubes_boxes_x == 1) & (tubes_boxes_y == 1)

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
        overlaps = iw * ih * it / ua
        overlaps.masked_fill_(gt_area_zero.view(
            batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps.masked_fill_(tubes_area_zero.view(
            batch_size, N, 1).expand(batch_size, N, K), -1)
    else:
        raise ValueError('tubes input dimension is not correct.')

    return overlaps

def validation(epoch, device, model, dataset_folder, sample_duration, spatial_transform, temporal_transform, boxes_file, splt_txt_path, cls2idx, batch_size, n_threads):

    iou_thresh = 0.5 # Intersection Over Union thresh
    data = Video_UCF(dataset_folder, frames_dur=sample_duration, spatial_transform=spatial_transform,
                     temporal_transform=temporal_transform, json_file = boxes_file,
                     split_txt_path=splt_txt_path, mode='test', classes_idx=cls2idx)
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                              shuffle=True, num_workers=n_threads, pin_memory=True)
    model.eval()
    true_pos = torch.zeros(1).long().to(device)
    false_neg = torch.zeros(1).long().to(device)
    ## 2 rois : 1450
    for step, data  in enumerate(data_loader):

        # if step == 2:
        #     break

        clips, h, w, gt_tubes_r, gt_rois, n_actions, n_frames, im_info = data

        clips_ = clips.to(device)
        gt_tubes_r_ = gt_tubes_r.to(device)
        gt_rois_ = gt_rois.to(device)
        n_actions_ = n_actions.to(device)
        im_info_ = im_info.to(device)
        start_fr = torch.zeros(clips_.size(0)).to(device)

        inputs = Variable(clips_)
        tubes,  bbox_pred, _, \
        _,  _, _, _, _, _  = model(inputs,
                                   im_info_,
                                   gt_tubes_r_, gt_rois_,
                                   start_fr)

        for i in range(tubes.size(0)):
            overlaps = bbox_overlaps_batch_3d(tubes[i].squeeze(0), gt_tubes_r_[i,:n_actions[i]].unsqueeze(0)) # check one video each time
            gt_max_overlaps, _ = torch.max(overlaps, 1)
            gt_max_overlaps = torch.where(gt_max_overlaps > iou_thresh, gt_max_overlaps, torch.zeros_like(gt_max_overlaps).type_as(gt_max_overlaps))
            detected =  gt_max_overlaps.ne(0).sum()
            n_elements = gt_max_overlaps.nelement()
            true_pos += detected
            false_neg += n_elements - detected

    recall = true_pos.float() / (true_pos.float() + false_neg.float())
    print('recall :',recall)
    print(' -----------------------')
    print('| Validation Epoch: {: >3} | '.format(epoch+1))
    print('|                       |')
    print('| Proposed Action Tubes |')
    print('|                       |')
    print('| In {: >6} steps    :  |\n| True_pos   --> {: >6} |\n| False_neg  --> {: >6} | \n| Recall     --> {: >6.4f} |'.format(
        step+1, true_pos.cpu().tolist()[0], false_neg.cpu().tolist()[0], recall.cpu().tolist()[0]))
    print(' -----------------------')
        
def training(epoch, device, model, dataset_folder, sample_duration, spatial_transform, temporal_transform, boxes_file, splt_txt_path, cls2idx, batch_size, n_threads, lr,):

    data = Video_UCF(dataset_folder, frames_dur=sample_duration, spatial_transform=spatial_transform,
                 temporal_transform=temporal_transform, json_file = boxes_file,
                 split_txt_path=splt_txt_path, mode='train', classes_idx=cls2idx)
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size*32,
                                              shuffle=True, num_workers=32, pin_memory=True)
    model.train()
    loss_temp = 0
    
    ## 2 rois : 1450
    for step, data  in enumerate(data_loader):

        # if step == 2:
        #     break

        clips, h, w, gt_tubes_r, gt_rois, n_actions, n_frames, im_info = data
        clips_ = clips.to(device)
        gt_tubes_r_ = gt_tubes_r.to(device)
        gt_rois_ = gt_rois.to(device)
        n_actions_ = n_actions.to(device)
        im_info_ = im_info.to(device)
        start_fr = torch.zeros(clips_.size(0)).to(device)
        
        inputs = Variable(clips_)
        tubes,  bbox_pred, _, \
        rpn_loss_cls,  rpn_loss_bbox, \
        act_loss_bbox, rpn_loss_cls_16,\
        rpn_loss_bbox_16, rois_label, \
        sgl_rois_bbox_pred, sgl_rois_bbox_loss = model(inputs,
                                                      im_info_,
                                                      gt_tubes_r_, gt_rois_,
                                                      start_fr)

        loss = rpn_loss_cls.mean() + rpn_loss_bbox.mean() + act_loss_bbox.mean() + rpn_loss_cls_16.mean() \
               + rpn_loss_bbox_16.mean() + sgl_rois_bbox_loss.mean()

        loss_temp += loss.item()

        # backw\ard
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Train Epoch: {} \tLoss: {:.6f}\t lr : {:.6f}'.format(
        epoch+1,loss_temp/(step+1), lr))

    return model, loss_temp

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    dataset_folder = '/gpu-data2/sgal/UCF-101-pickle'
    dataset_frames = '/gpu-data2/sgal/UCF-101-frames'
    boxes_file = '/gpu-data2/sgal/pyannot.pkl'
    split_txt_path = '/gpu-data2/sgal/UCF101_Action_detection_splits/'

    sample_size = 112
    sample_duration = 16  # len(images)

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
    vid2idx,vid_names = get_vid_dict(dataset_frames)

    spatial_transform = Compose([Scale(sample_size),  # [Resize(sample_size),
                                 ToTensor(),
                                 Normalize(mean, [1, 1, 1])])
    temporal_transform = LoopPadding(sample_duration)

    n_classes = len(actions)


    # ########################################
    # #          Part 1 - train TPN          #
    # ########################################

    # # Init action_net
    # act_model = ACT_net(actions, sample_duration)
    # act_model.create_architecture()
    # if torch.cuda.device_count() > 1:
    #     print('Using {} GPUs!'.format(torch.cuda.device_count()))

    #     act_model = nn.DataParallel(act_model)

    # act_model.to(device)

    # lr = 0.1
    # lr_decay_step = 8
    # lr_decay_gamma = 0.1
    

    # params = []
    # for key, value in dict(act_model.named_parameters()).items():
    #     # print(key, value.requires_grad)
    #     if value.requires_grad:
    #         print('key :',key)
    #         if 'bias' in key:
    #             params += [{'params':[value],'lr':lr*(True + 1), \
    #                         'weight_decay': False and 0.0005 or 0}]
    #         else:
    #             params += [{'params':[value],'lr':lr, 'weight_decay': 0.0005}]

    # lr = lr * 0.1
    # optimizer = torch.optim.Adam(params)

    # epochs = 40
    # n_devs = torch.cuda.device_count()
    # for epoch in range(epochs):
    #     print(' ============\n| Epoch {:0>2}/{:0>2} |\n ============'.format(epoch+1, epochs))

    #     if epoch % (lr_decay_step + 1) == 0:
    #         adjust_learning_rate(optimizer, lr_decay_gamma)
    #         lr *= lr_decay_gamma


    #     act_model, loss = training(epoch, device, act_model, dataset_folder, sample_duration, spatial_transform, temporal_transform, boxes_file, split_txt_path, cls2idx, n_devs, 0, lr)

    #     if (epoch + 1) % (5) == 0:
    #         validation(epoch, device, act_model, dataset_folder, sample_duration, spatial_transform, temporal_transform, boxes_file, split_txt_path, cls2idx, n_devs, 0)


    #     if ( epoch + 1 ) % 5 == 0:
    #         torch.save(act_model.state_dict(), "action_net_model.pwf".format(epoch+1))
    # torch.save(act_model.state_dict(), "action_net_model.pwf".format(epoch))

    ###########################################
    #          Part 2 - train Linear          #
    ###########################################
    
    # first initialize model
    n_devs = torch.cuda.device_count()
    model = Model(actions, sample_duration, sample_size)
    model.load_part_model()
    model.deactivate_action_net_grad()
    
    if torch.cuda.device_count() > 1:

        print('Using {} GPUs!'.format(torch.cuda.device_count()))
        # model.act_net = nn.DataParallel(model.act_net)
        model = nn.DataParallel(model)

    # model.act_net = model.act_net.to(device)
    model = model.to(device)
    # init data_loaders
    
    vid_name_loader = video_names(dataset_frames, split_txt_path, boxes_file, vid2idx, mode='train')
    data_loader = torch.utils.data.DataLoader(vid_name_loader, batch_size=n_devs, num_workers=8*n_devs, pin_memory=True,
                                              shuffle=True)    # reset learning rate

    lr = 0.1
    lr_decay_step = 5
    lr_decay_gamma = 0.1

    # reset learning rate

    params = []
    for key, value in dict(model.module.act_rnn.named_parameters()).items():
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

    ##########################
    
    epochs = 40
    # epochs = 1

    for ep in range(epochs):

        model.train()
        loss_temp = 0

        print(' ============\n| Epoch {:0>2}/{:0>2} |\n ============'.format(ep+1, epochs))

        # model.train()
        # model.act_net.eval()
        
        if ep % (lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, lr_decay_gamma)
            lr *= lr_decay_gamma

        for step, data  in enumerate(data_loader):

            # if step == 2:
            #     break

            # print('step :',step)
            vid_id, clips, boxes, n_frames, n_actions, h, w = data
            mode = 'train'

            vid_id = vid_id.to(device)
            n_frames = n_frames.to(device)
            n_actions = n_actions.to(device)
            h = h.to(device)
            w = w.to(device)

            tubes,  bbox_pred, \
            prob_out, rpn_loss_cls, \
            rpn_loss_bbox, act_loss_bbox,  cls_loss =  model(n_devs, dataset_folder, \
                                                             vid_names, clips, vid_id,  \
                                                             boxes, \
                                                             mode, cls2idx, n_actions,n_frames, h, w)

            loss = cls_loss.mean()

            # backw\ard
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_temp += loss.item()

        print('Train Epoch: {} \tLoss: {:.6f}\t lr : {:.6f}'.format(
        ep+1,loss_temp/step, lr))
        if ( ep + 1 ) % 5 == 0:
            torch.save(model.linear.state_dict(), "act_rnn.pwf")
    torch.save(model.linear.state_dict(), "act_rnn.pwf")

    # ###########################################
    # #          Part 3 - train Linear          #
    # ###########################################
    
    # # first initialize model and load weights
    # # act_net_path = './action_net_model.pwf'
    # # linear_path = './linear.pwf'

    # torch.backends.cudnn.benchmark = False
    # model = Model(actions, sample_duration, sample_size)
    # # model.load_part_model(action_model_path=act_net_path, linear_path = linear_path)
    # model.load_part_model()

    # n_devs = torch.cuda.device_count()
    # # model.load_part_model(action_model_path=None, linear_path = None)
    # # if torch.cuda.device_count() > 1:

    # #     print('Using {} GPUs!'.format(torch.cuda.device_count()))
    # #     model = nn.DataParallel(model)

    # # model = model.to(device)
    # if torch.cuda.device_count() > 1:

    #     print('Using {} GPUs!'.format(torch.cuda.device_count()))
    #     model.act_net = nn.DataParallel(model.act_net)
    #     # model = nn.DataParallel(model)

    # model.act_net = model.act_net.to(device)
    # model = model.to(device)

    # # init data_loaders
    
    # # vid_name_loader = video_names(dataset_frames, split_txt_path, boxes_file, vid2idx, mode='train')
    # vid_name_loader = video_names(dataset_folder, split_txt_path, boxes_file, vid2idx, mode='train', sample_size=sample_size)
    # data_loader = torch.utils.data.DataLoader(vid_name_loader, batch_size=1,num_workers=4,pin_memory=True,
    #                                           shuffle=True)    # reset learning rate
    #                                           # shuffle=False)    # reset learning rate

    # lr = 0.1
    # lr_decay_step = 5
    # lr_decay_gamma = 0.1

    # # reset learning rate

    # params = []
    # for key, value in dict(model.named_parameters()).items():
    #     # print(key, value.requires_grad)
    #     if value.requires_grad:
    #         print('key :',key)
    #         if 'bias' in key:
    #             params += [{'params':[value],'lr':lr*(True + 1), \
    #                         'weight_decay': False and 0.0005 or 0}]
    #         else:
    #             params += [{'params':[value],'lr':lr, 'weight_decay': 0.0005}]

    # lr = lr * 0.1
    # optimizer = torch.optim.Adam(params)

    # ##########################

    
    # # epochs = 40
    # epochs = 1

    # for ep in range(epochs):

    #     model.train()
    #     loss_temp = 0

    #     print(' ============\n| Epoch {:0>2}/{:0>2} |\n ============'.format(ep+1, epochs))

    #     if ep % (lr_decay_step + 1) == 0:
    #         adjust_learning_rate(optimizer, lr_decay_gamma)
    #         lr *= lr_decay_gamma

    #     for step, data  in enumerate(data_loader):

    #         # if step == 2:
    #         #     break

    #         print('! step :',step)
    #         vid_id, clips, boxes, n_frames, n_actions, h, w = data
    #         # print('clips.type() :',clips.type())
    #         # print('clips.type() :',clips.shape)
    #         # print('vid_id.shape :',vid_id.shape)
    #         # print('boxes.shape :',boxes.shape)
    #         # print('n_frames.shape :',n_frames.shape)
    #         # print('n_actions.shape :',n_actions.shape)
    #         # print('h :',h)
    #         # print('w :',w)

    # #         boxes = preprocess_boxes(boxes, h, w, sample_size).to(device)
    #         mode = 'train'

    #         vid_id = vid_id.to(device)
    #         n_frames = n_frames.to(device)
    #         n_actions = n_actions.to(device)
    #         h = h.to(device)
    #         w = w.to(device)
    #         # print('boxes.type() :',boxes.type())
    #         # print('----------Before----------')
    #         tubes,  bbox_pred, \
    #         prob_out, rpn_loss_cls, \
    #         rpn_loss_bbox, act_loss_bbox,  cls_loss =  model(n_devs, dataset_folder, \
    #                                                          vid_names, clips, vid_id,  \
    #                                                          boxes, \
    #                                                          mode, cls2idx, n_actions,n_frames, h, w)

    #         loss = cls_loss.mean()

    #         # backw\ard
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

            
    #         loss_temp += loss.item()

    #     print('Train Epoch: {} \tLoss: {:.6f}\t lr : {:.6f}'.format(
    #     ep+1,loss_temp/(step+1), lr))
    #     if ( ep + 1 ) % 5 == 0:
    #         torch.save(model.state_dict(), "model.pwf")
    # torch.save(model.state_dict(), "model.pwf")


