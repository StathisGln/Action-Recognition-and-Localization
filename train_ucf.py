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
    print("Device being used:", device)

    dataset_folder = '/gpu-data2/sgal/UCF-101-frames'
    boxes_file = '/gpu-data2/sgal/pyannot.pkl'

    sample_size = 112
    sample_duration = 16  # len(images)

    batch_size = 2

    # # get mean
    mean = [112.07945832, 112.87372333, 106.90993363]  # ucf-101 24 classes

    # generate model
    actions = ['__background__', 'Basketball','BasketballDunk','Biking','CliffDiving','CricketBowling',
               'Diving','Fencing','FloorGymnastics','GolfSwing','HorseRiding','IceDancing',
               'LongJump','PoleVault','RopeClimbing','SalsaSpin','SkateBoarding','Skiing',
               'Skijet','SoccerJuggling','Surfing','TennisSwing','TrampolineJumping',
               'VolleyballSpiking','WalkingWithDog']

    cls2idx = {actions[i]: i for i in range(0, len(actions))}

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

    if torch.cuda.device_count() > 1:

        print('Using {} GPUs!'.format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

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

    vid_names = video_names(dataset_folder, boxes_file)
    data_loader = torch.utils.data.DataLoader(vid_names, batch_size=batch_size,
                                              shuffle=True)

    epochs = 40
    
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

            if step == 2:
                break

            vid_path,  boxes, n_frames, n_actions, = data
            
            ###################################
            #          Function here          #
            ###################################

            mode = 'train'
            print('vid_path', vid_path)
            print('boxes.shape :',boxes.shape)
            tubes,  bbox_pred, \
            prob_out, rpn_loss_cls, \
            rpn_loss_bbox, act_loss_bbox,  cls_loss =  model(device, dataset_folder, \
                                                             vid_id, spatial_transform, \
                                                             temporal_transform, boxes, \
                                                             mode, cls2idx, n_actions,)

            out_prob, tcn_loss = model( clips, target, gt_tubes_r, n_frames, max_dim=1)
            loss = tcn_loss.mean()

            # backw\ard
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_temp += loss.item()

        print('Train Epoch: {} \tLoss: {:.6f}\t lr : {:.6f}'.format(
        ep+1,loss_temp/step, lr))
        
    #     if ( ep + 1 ) % 5 == 0: # validation time
    #         val_data = Video(dataset_folder, frames_dur=sample_duration, spatial_transform=spatial_transform,
    #                      temporal_transform=temporal_transform, json_file = boxes_file,
    #                      split_txt_path=splt_txt_path, mode='val', classes_idx=cls2idx)
    #         val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,
    #                                                   shuffle=True, num_workers=n_threads, pin_memory=True)

    #         validate_tcn(model, val_data, val_data_loader)
    #     if ( ep + 1 ) % 5 == 0:
    #         torch.save(model.state_dict(), "model.pwf")
    # torch.save(model.state_dict(), "model.pwf")


