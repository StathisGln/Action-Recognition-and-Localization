import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from lib.dataloaders.ucf_dataset import Video_Dataset_small_clip

from lib.utils.spatial_transforms import (
    Compose, Normalize, Scale, ToTensor)
from lib.utils.temporal_transforms import LoopPadding

from lib.utils.create_video_id import get_vid_dict

from lib.models.action_net import ACT_net
import argparse

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
    data = Video_Dataset_small_clip(dataset_folder, frames_dur=sample_duration, spatial_transform=spatial_transform,
                                    temporal_transform=temporal_transform, bboxes_file= boxes_file,
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
        _,  _, _, _, _, _, _,sgl_rois_bbox_pred, _  = model(inputs,
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
        
def training(epoch, device, model, dataset_folder, sample_duration, spatial_transform, temporal_transform, boxes_file, splt_txt_path, cls2idx, batch_size, n_threads, lr, mode = 1):

    data = Video_Dataset_small_clip(dataset_folder, frames_dur=sample_duration, spatial_transform=spatial_transform,
                                    temporal_transform=temporal_transform, bboxes_file= boxes_file,
                                    split_txt_path=splt_txt_path, mode='train', classes_idx=cls2idx)
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size*16,
                                              shuffle=True, num_workers=32, pin_memory=True)
    # data_loader = torch.utils.data.DataLoader(data, batch_size=2,
    #                                           shuffle=True, num_workers=0, pin_memory=True)

    model.train()
    
    ## 2 rois : 1450
    for step, data  in enumerate(data_loader):

        if step == 2:
            exit(-1)
            break

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
        rpn_loss_bbox_16, act_loss_bbox_16, rois_label, \
        sgl_rois_bbox_pred, sgl_rois_bbox_loss,  = model(inputs, \
        # actioness_score, actioness_loss
                                                im_info_,
                                                gt_tubes_r_, gt_rois_,
                                                start_fr)

    return model, -1

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train action_net, regression layer and RNN')

    parser.add_argument('--demo', '-d', help='Run just 2 steps for test if everything works fine', action='store_true')
    # parser.add_argument('--n_1_1', help='Run only part 1.1, training action net only', action='store_true')
    # parser.add_argument('--n_1_2', help='Run only part 1.2, training only regression layer', action='store_true')
    # parser.add_argument('--n_2', help='Run only part 2, train only RNN', action='store_true')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

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

    # for p in act_model.module.reg_layer.parameters() : p.requires_grad=False

    epochs = 5

    n_devs = torch.cuda.device_count()
    for epoch in range(epochs):
        print(' ============\n| Epoch {:0>2}/{:0>2} |\n ============'.format(epoch+1, epochs))

        act_model, loss = training(epoch, device, act_model, dataset_frames, sample_duration, spatial_transform, temporal_transform, boxes_file, split_txt_path, cls2idx, n_devs, 0, 0.1, mode=4)




