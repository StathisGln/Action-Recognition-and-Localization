import os
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

# from ucf_dataset import Video_UCF, video_names
from jhmdb_dataset import Video


from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, ToTensor, Resize)
from temporal_transforms import LoopPadding

from create_video_id import get_vid_dict
from net_utils import adjust_learning_rate
from resize_rpn import resize_rpn, resize_tube

from model import Model
from action_net import ACT_net
from resize_rpn import resize_boxes
import argparse
from bbox_transform import bbox_transform_inv, bbox_overlaps_batch, clip_boxes
np.random.seed(42)

def rois_overlaps(tubes, gt_tubes):

    sample_duration = tubes.size(1)
    overl = []
    for i in range(sample_duration):
        overl.append(bbox_overlaps(tubes[:,i], gt_tubes[:,i,:4]))

    overl = torch.stack(overl)
    non_zero = overl.ne(-1.0).sum(0).float()
    sums = overl.clamp_(min=0).sum(0)
    overlaps = sums/non_zero
    idx = gt_tubes[:,:,:4].contiguous().view(gt_tubes.size(0),sample_duration*4).eq(0).all(dim=1)
    if idx.nelement() != 0:
        overlaps.masked_fill_(idx.view(
             1,idx.size(0),).expand( overlaps.size(0),idx.size(0)), -1)
    return overlaps


    
def bbox_overlaps(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 4) ndarray of float

    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.size(0)
    K = gt_boxes.size(0)

    gt_boxes_x = (gt_boxes[ :, 2] - gt_boxes[ :, 0] + 1)
    gt_boxes_y = (gt_boxes[ :, 3] - gt_boxes[ :, 1] + 1)

    gt_boxes_area = (gt_boxes_x *
                     gt_boxes_y).view(1,K)

    anchors_boxes_x = (anchors[ :, 2] - anchors[ :, 0] + 1)
    anchors_boxes_y = (anchors[ :, 3] - anchors[ :, 1] + 1)

    anchors_area = (anchors_boxes_x *
                    anchors_boxes_y).view(N, 1)

    gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
    anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

    boxes = anchors.view(N, 1, 4).expand(N, K, 4)
    query_boxes = gt_boxes.view(1, K, 4).expand(N, K, 4)

    iw = (torch.min(boxes[:, :, 2], query_boxes[:, :, 2]) -
          torch.max(boxes[:, :, 0], query_boxes[:, :, 0]) + 1)
    iw[iw < 0] = 0

    ih = (torch.min(boxes[:, :, 3], query_boxes[:, :, 3]) -
          torch.max(boxes[:, :, 1], query_boxes[:, :, 1]) + 1)
    ih[ih < 0] = 0

    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    anchors_area_zero = anchors_area_zero.view(N, 1).expand( N, K)
    gt_area_zero = gt_area_zero.view( 1, K).expand(N, K)

    zero_area = (anchors_area_zero == 1) & (gt_area_zero == 1)

    overlaps.masked_fill_(zero_area, -1)

    return overlaps

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

    data = Video(dataset_folder, frames_dur=sample_duration, spatial_transform=spatial_transform,
                     temporal_transform=temporal_transform, json_file = boxes_file,
                     split_txt_path=splt_txt_path, mode='test', classes_idx=cls2idx)
    # data = Video_UCF(dataset_folder, frames_dur=sample_duration, spatial_transform=spatial_transform,
    #                  temporal_transform=temporal_transform, json_file = boxes_file,
    #                  split_txt_path=splt_txt_path, mode='test', classes_idx=cls2idx)

    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                              shuffle=True, num_workers=n_threads, pin_memory=True)
    model.eval()
    true_pos = torch.zeros(1).long().to(device)
    false_neg = torch.zeros(1).long().to(device)

    sgl_true_pos = torch.zeros(1).long().to(device)
    sgl_false_neg = torch.zeros(1).long().to(device)

    sgl_follow_tp = torch.zeros(1).long().to(device)
    sgl_follow_fn = torch.zeros(1).long().to(device)

    for step, data  in enumerate(data_loader):

        # if step == 2:
        #     break
        # print('step :',step)
        clips, h, w, gt_tubes_r, gt_rois, n_actions, n_frames, im_info = data

        clips_ = clips.to(device)
        gt_tubes_r_ = gt_tubes_r.to(device)
        gt_rois_ = gt_rois.to(device)
        n_actions_ = n_actions.to(device)
        im_info_ = im_info.to(device)
        start_fr = torch.zeros(clips_.size(0)).to(device)

        inputs = Variable(clips_)
        tubes,  _, \
        _, _, _, _, _,sgl_rois_bbox_pred, _  = model(inputs,
                                                     im_info_,
                                                     gt_tubes_r_, gt_rois_,
                                                     start_fr)

        tubes_rois = tubes[:,:,[1,2,4,5]].unsqueeze(2).expand(tubes.size(0),tubes.size(1),sample_duration, 4).permute(0,2,1,3).contiguous()
        tubes_rois = bbox_transform_inv(tubes_rois.view(-1, tubes_rois.size(2),4).contiguous(),
                                        sgl_rois_bbox_pred.view(-1, tubes_rois.size(2),4).contiguous(),
                                        tubes_rois.size(0)*tubes_rois.size(1))

        tubes_rois = clip_boxes(tubes_rois,im_info.unsqueeze(1).expand(tubes.size(0),sample_duration,3).contiguous().view(-1,3), tubes_rois.size(0))
        
        tubes_rois = tubes_rois.view(tubes.size(0), -1, tubes.size(1), 4).contiguous().permute(0,2,1,3)

        rois = torch.zeros(tubes.size(0), tubes.size(1), sample_duration,4).type_as(tubes)

        for i in range(tubes.size(0)):

            start_fr = tubes[i,:,3].round().int()
            end_fr = tubes[i,:,6].round().int()
            for j in range(tubes.size(1)):

                rois[i,j,start_fr[j]:end_fr[j]+1] = tubes_rois[i,j,start_fr[j]:end_fr[j]+1]

        for i in range(tubes.size(0)):
            # # get overlaps in tubes

            overlaps_rois = rois_overlaps(rois[i], gt_rois_[i].type_as(rois))
            overlaps_max,_ =  torch.max(overlaps_rois, 0)
            overlaps_max = torch.where(overlaps_max > iou_thresh, overlaps_max, torch.zeros_like(overlaps_max).type_as(overlaps_max))
            detected = overlaps_max.ne(0).sum()
            n_elements = overlaps_max.nelement()
            sgl_true_pos += detected
            sgl_false_neg += n_elements - detected
            
            # check one video each time
            overlaps = bbox_overlaps_batch_3d(tubes[i].squeeze(0), gt_tubes_r_[i,:n_actions[i]].unsqueeze(0).type_as(tubes))

            gt_max_overlaps, argmax = torch.max(overlaps, 1)
            gt_max_overlaps = torch.where(gt_max_overlaps > iou_thresh, gt_max_overlaps, torch.zeros_like(gt_max_overlaps).type_as(gt_max_overlaps))
            detected =  gt_max_overlaps.ne(0).sum()
            n_elements = gt_max_overlaps.nelement()
            true_pos += detected
            false_neg += n_elements - detected

            tubes_over_ind = overlaps.view(overlaps.size(1)).gt(iou_thresh).nonzero().view(-1)

            if tubes_over_ind.nelement() > 0:

                overlaps_rois = overlaps_rois[tubes_over_ind]
                n_elements = overlaps_rois.nelement()
                
                overlaps_max,_ =  torch.max(overlaps_rois, 1)
                overlaps_max = torch.where(overlaps_max > iou_thresh, overlaps_max, torch.zeros_like(overlaps_max).type_as(overlaps_max))

                detected = overlaps_max.ne(0).sum()

                sgl_follow_tp += detected
                sgl_follow_fn += n_elements - detected

    recall = true_pos.float() / (true_pos.float() + false_neg.float())
    sgl_recall = sgl_true_pos.float() / (sgl_true_pos.float() + sgl_false_neg.float())
    sgl_follow_recall = sgl_follow_tp.float() / (sgl_follow_tp.float() + sgl_follow_fn.float())
    print(' -----------------------')
    print('| Validation Epoch: {: >3} | '.format(epoch+1))
    print('|                       |')
    print('| Proposed Action Tubes |')
    print('|                       |')
    print('| In {: >6} steps    :  |\n| True_pos   --> {: >6} |\n| False_neg  --> {: >6} | \n| Recall     --> {: >6.4f} |'.format(
        step+1, true_pos.cpu().tolist()[0], false_neg.cpu().tolist()[0], recall.cpu().tolist()[0]))
    print('|                       |')
    print('| Single Rois           |')
    print('|                       |')
    print('| In {: >6} steps    :  |\n| True_pos   --> {: >6} |\n| False_neg  --> {: >6} | \n| Recall     --> {: >6.4f} |'.format(
        step+1, sgl_true_pos.cpu().tolist()[0], sgl_false_neg.cpu().tolist()[0], sgl_recall.cpu().tolist()[0]))
    print('|                       |')
    print('| Single Rois follow up |')
    print('|                       |')
    print('| In {: >6} steps    :  |\n| True_pos   --> {: >6} |\n| False_neg  --> {: >6} | \n| Recall     --> {: >6.4f} |'.format(
        step+1, sgl_follow_tp.cpu().tolist()[0], sgl_follow_fn.cpu().tolist()[0], sgl_follow_recall.cpu().tolist()[0]))

    print(' -----------------------')

    fp = open('validation.txt','a')
    fp.write(' -----------------------\n')
    fp.write('| Validation Epoch: {: >3} |\n'.format(epoch+1))
    fp.write('|                       |\n')
    fp.write('| Proposed Action Tubes |\n')
    fp.write('|                       |\n')
    fp.write('| In {: >6} steps    :  |\n| True_pos   --> {: >6} |\n| False_neg  --> {: >6} | \n| Recall     --> {: >6.4f} |\n'.format(
        step+1, true_pos.cpu().tolist()[0], false_neg.cpu().tolist()[0],recall.cpu().tolist()[0]))
    fp.write('|                       |\n')
    fp.write('| Single Rois           |\n')
    fp.write('|                       |\n')
    fp.write('| In {: >6} steps    :  |\n| True_pos   --> {: >6} |\n| False_neg  --> {: >6} | \n| Recall     --> {: >6.4f} |\n'.format(
        step+1, sgl_true_pos.cpu().tolist()[0], sgl_false_neg.cpu().tolist()[0], sgl_recall.cpu().tolist()[0]))
    fp.write('|                       |\n')
    fp.write('| Single Rois follow up |\n')
    fp.write('|                       |\n')
    fp.write('| In {: >6} steps    :  |\n| True_pos   --> {: >6} |\n| False_neg  --> {: >6} | \n| Recall     --> {: >6.4f} |\n'.format(
        step+1, sgl_follow_tp.cpu().tolist()[0], sgl_follow_fn.cpu().tolist()[0], sgl_follow_recall.cpu().tolist()[0]))


    fp.write(' -----------------------\n')

def training(epoch, device, model, dataset_folder, sample_duration, spatial_transform, temporal_transform, boxes_file, splt_txt_path, cls2idx, batch_size, n_threads, lr, mode = 1):

    data = Video(dataset_folder, frames_dur=sample_duration, spatial_transform=spatial_transform,
                 temporal_transform=temporal_transform, json_file = boxes_file,
                 split_txt_path=splt_txt_path, mode='train', classes_idx=cls2idx)
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size*16,
                                              shuffle=True, num_workers=32, pin_memory=True)
    # data_loader = torch.utils.data.DataLoader(data, batch_size=2,
    #                                           shuffle=True, num_workers=0, pin_memory=True)

    model.train()
    loss_temp = 0
    
    ## 2 rois : 1450
    for step, data  in enumerate(data_loader):

        # if step == 2:
        #     break

        clips, h, w, gt_tubes_r, gt_rois, n_actions, n_frames, im_info = data
        clips_ = clips.to(device)
        gt_tubes_r_ = gt_tubes_r.float().to(device)
        gt_rois_ = gt_rois.float().to(device)
        n_actions_ = n_actions.to(device)
        im_info_ = im_info.to(device)
        start_fr = torch.zeros(clips_.size(0)).to(device)
        
        inputs = Variable(clips_)
        tubes,  _, \
        rpn_loss_cls,  rpn_loss_bbox, \
        rpn_loss_cls_16,\
        rpn_loss_bbox_16,  rois_label, \
        sgl_rois_bbox_pred, sgl_rois_bbox_loss,  = model(inputs, \
        # actioness_score, actioness_loss
                                                im_info_,
                                                gt_tubes_r_, gt_rois_,
                                                start_fr)
        if mode == 1:
            loss = rpn_loss_cls.mean() + rpn_loss_bbox.mean() + act_loss_bbox.mean() + rpn_loss_cls_16.mean() \
                   + rpn_loss_bbox_16.mean() +  act_loss_bbox_16.mean() 
        # elif mode == 2:
        #     loss = actioness_loss.mean()
        elif mode == 3:
            loss = sgl_rois_bbox_loss.mean()
        # elif mode == 4:
        #     loss = rpn_loss_cls.mean() + rpn_loss_bbox.mean() 
        elif mode == 4:
            loss = rpn_loss_cls.mean() + rpn_loss_bbox.mean()  + rpn_loss_cls_16.mean() \
                   + rpn_loss_bbox_16.mean()  +  sgl_rois_bbox_loss.mean()




        # elif mode == 4:
        #     loss = rpn_loss_cls.mean() + rpn_loss_bbox.mean()   \

            # loss = rpn_loss_cls.mean() + rpn_loss_bbox.mean()  + rpn_loss_cls_16.mean() \
            #        + rpn_loss_bbox_16.mean()  + sgl_rois_bbox_loss.mean()


        loss_temp += loss.item()

        # backw\ard
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Train Epoch: {} \tLoss: {:.6f}\t lr : {:.6f}'.format(
        epoch+1,loss_temp/(step+1), lr))

    return model, loss_temp

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train action_net, regression layer and RNN')

    parser.add_argument('--demo', '-d', help='Run just 2 steps for test if everything works fine', action='store_true')
    # parser.add_argument('--n_1_1', help='Run only part 1.1, training action net only', action='store_true')
    # parser.add_argument('--n_1_2', help='Run only part 1.2, training only regression layer', action='store_true')
    # parser.add_argument('--n_2', help='Run only part 2, train only RNN', action='store_true')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    dataset_folder = '../JHMDB-act-detector-frames'
    split_txt_path =  '../splits'
    boxes_file = '../poses.json'

    sample_size = 112
    sample_duration = 16  # len(images)

    # # get mean
    mean = [112.07945832, 112.87372333, 106.90993363]  # ucf-101 24 classes


    # generate model
    actions = ['__background__','brush_hair', 'clap', 'golf', 'kick_ball', 'pour',
               'push', 'shoot_ball', 'shoot_gun', 'stand', 'throw', 'wave',
               'catch','climb_stairs', 'jump', 'pick', 'pullup', 'run', 'shoot_bow', 'sit',
               'swing_baseball', 'walk' ]


    cls2idx = {actions[i]: i for i in range(0, len(actions))}

    ### get videos id
    vid2idx,vid_names = get_vid_dict(dataset_folder)

    spatial_transform = Compose([Scale(sample_size),  # [Resize(sample_size),
                                 ToTensor(),
                                 Normalize(mean, [1, 1, 1])])
    temporal_transform = LoopPadding(sample_duration)

    n_classes = len(actions)


    #######################################################
    #          Part 1-1 - train nTPN - without reg         #
    #######################################################

    print(' -----------------------------------------------------')
    print('|          Part 1-1 - train TPN - without reg         |')
    print(' -----------------------------------------------------')

    # Init action_net
    act_model = ACT_net(actions, sample_duration)
    act_model.create_architecture()
    if torch.cuda.device_count() > 1:
        print('Using {} GPUs!'.format(torch.cuda.device_count()))

    act_model = nn.DataParallel(act_model)

    act_model.to(device)

    lr = 0.1
    lr_decay_step = 5
    lr_decay_gamma = 0.1
    
    params = []

    # for p in act_model.module.reg_layer.parameters() : p.requires_grad=False

    for key, value in dict(act_model.named_parameters()).items():
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

    epochs = 40
    # epochs = 2

    file = open('training_loss.txt','w')
    n_devs = torch.cuda.device_count()
    for epoch in range(epochs):
        print(' ============\n| Epoch {:0>2}/{:0>2} |\n ============'.format(epoch+1, epochs))

        if epoch % (lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, lr_decay_gamma)
            lr *= lr_decay_gamma
        act_model, loss = training(epoch, device, act_model, dataset_folder, sample_duration, spatial_transform, temporal_transform, boxes_file, split_txt_path, cls2idx, n_devs*4, 0, lr, mode=4)

        file.write('epoch :'+str(epoch)+' loss :'+str(loss)+'\n')
        if (epoch + 1) % (5) == 0:
            validation(epoch, device, act_model, dataset_folder, sample_duration, spatial_transform, temporal_transform, boxes_file, split_txt_path, cls2idx, n_devs, 0)

        if ( epoch + 1 ) % 5 == 0:
            torch.save(act_model.state_dict(), "action_net_model_2scores_mIoU.pwf".format(epoch+1))
    torch.save(act_model.state_dict(), "action_net_model_2scores_mIoU.pwf")

