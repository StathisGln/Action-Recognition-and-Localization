import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lib.dataloaders.ucf_dataset import Video_Dataset

from lib.utils.spatial_transforms import (
    Compose, Normalize, Scale, ToTensor)
from lib.utils.temporal_transforms import LoopPadding
from lib.models.action_net import ACT_net
from lib.utils.box_functions import tube_transform_inv, clip_boxes, tube_overlaps

np.random.seed(42)


def validation(epoch, device, model, dataset_folder, sample_duration, spatial_transform, temporal_transform, boxes_file, splt_txt_path, cls2idx, batch_size, n_threads):

    iou_thresh = 0.5 # Intersection Over Union thresh
    # iou_thresh = 0.1 # Intersection Over Union thresh
    data = Video_Dataset(dataset_folder, frames_dur=sample_duration, spatial_transform=spatial_transform,
                         temporal_transform=temporal_transform, bboxes_file= boxes_file,
                         split_txt_path=splt_txt_path, mode='test', classes_idx=cls2idx)
    data_loader = torch.utils.data.DataLoader(data, batch_size=16,
                                              shuffle=True, num_workers=0, pin_memory=True)
    # data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size*4,
    #                                           shuffle=True, num_workers=0, pin_memory=True)

    model.eval()

    sgl_true_pos = 0
    sgl_false_neg = 0

    ## 2 rois : 1450
    tubes_sum = 0
    for step, data  in enumerate(data_loader):

        # if step == 10:
        #     break
        print('step :',step)

        clips, h, w, gt_tubes_r, gt_rois, n_actions, n_frames, im_info = data
        clips_   = clips.to(device)
        gt_tubes_r_ = gt_tubes_r.to(device)
        gt_rois_ = gt_rois.to(device)
        n_actions_ = n_actions.to(device)
        im_info_ = im_info.to(device)
        start_fr = torch.zeros(clips_.size(0)).to(device)
        # for i in range(2):
        #     print('gt_rois :',gt_rois[i,:n_actions[i]])
        tubes, _, _, _, _, _, _, \
        sgl_rois_bbox_pred, _  = model(clips,
                                       im_info,
                                       None, None,
                                       None)
        tubes_ = tubes.contiguous()
        n_tubes = len(tubes)

        tubes = tubes.view(-1, sample_duration*4+2)

        tubes[:,1:-1] = tube_transform_inv(tubes[:,1:-1],\
                                           sgl_rois_bbox_pred.view(-1,sample_duration*4),(1.0,1.0,1.0,1.0))
        tubes = tubes.view(n_tubes,-1, sample_duration*4+2)
        tubes[:,:,1:-1] = clip_boxes(tubes[:,:,1:-1], im_info, tubes.size(0))

        
        # print('tubes[0]:',tubes.shape)
        # exit(-1)
        # print('tubes.cpu().numpy() :',tubes.cpu().numpy())
        # exit(-1)
        # print('gt_rois_[:,0] :',gt_rois_[:,0])

        for i in range(tubes.size(0)): # how many frames we have
            
            tubes_t = tubes[i,:,1:-1].contiguous()
            gt_rois_t = gt_rois_[i,:,:,:4].contiguous().view(-1,sample_duration*4)

            rois_overlaps = tube_overlaps(tubes_t,gt_rois_t)
            # rois_overlaps = Tube_Overlaps()(tubes_t,gt_rois_t)

            gt_max_overlaps_sgl, max_indices = torch.max(rois_overlaps, 0)

            non_empty_indices =  gt_rois_t.ne(0).any(dim=1).nonzero().view(-1)
            n_elems = non_empty_indices.nelement()            
            # print('non_empty_indices :',non_empty_indices)
            # if gt_tubes_r[i,0,5] - gt_tubes_r[i,0,2 ] < 12 and gt_tubes_r[i,0,5] - gt_tubes_r[i,0,2 ] > 0:
            #     print('tubes_t.cpu().numpy() :',tubes_t[:5].detach().cpu().numpy())
            #     print('sgl_rois_bbox_pred.cpu().numpy() :',sgl_rois_bbox_pred[i,:5].detach().cpu().numpy())
            #     print('tubes_.detach.cpu().numpy() :',tubes_[i,:5].detach().cpu().numpy())
            #     print('gt_rubes_r[i] :',gt_tubes_r[i])
            #     exit(-1)

            if gt_max_overlaps_sgl[0] > 0.5 and gt_rois_t[0,-4:].sum()==0:
                print('max_indices :',max_indices, max_indices.shape, gt_max_overlaps_sgl )
                print('tubes_t[max_indices[0]] :',tubes_t[max_indices[0]])
                print('gt_rois_t[0] :',gt_rois_t[0])

            gt_max_overlaps_sgl = torch.where(gt_max_overlaps_sgl > iou_thresh, gt_max_overlaps_sgl, torch.zeros_like(gt_max_overlaps_sgl).type_as(gt_max_overlaps_sgl))

            sgl_detected =  gt_max_overlaps_sgl[non_empty_indices].ne(0).sum()

            sgl_true_pos += sgl_detected
            sgl_false_neg += n_elems - sgl_detected

        # if step == 0:
        #     break
        #     # exit(-1)


    recall = float(sgl_true_pos)  / (float(sgl_true_pos)  + float(sgl_false_neg))

    print(' -----------------------')
    print('| Validation Epoch: {: >3} | '.format(epoch+1))
    print('|                       |')
    print('| Proposed Action Tubes |')
    print('|                       |')
    print('| Single frame          |')
    print('|                       |')
    print('| In {: >6} steps    :  |'.format(step))
    print('|                       |')
    print('| True_pos   --> {: >6} |\n| False_neg  --> {: >6} | \n| Recall     --> {: >6.4f} |'.format(
        sgl_true_pos, sgl_false_neg, recall))


    print(' -----------------------')
        
if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    dataset_folder = '/gpu-data2/sgal/UCF-101-frames'
    boxes_file = '/gpu-data2/sgal/pyannot.pkl'
    split_txt_path = '/gpu-data2/sgal/UCF101_Action_detection_splits/'

    sample_size = 112
    # sample_duration = 8 # len(images)
    sample_duration = 16 # len(images)

    batch_size = 1
    n_threads = 0

    # # get mean
    # mean =  [103.75581543 104.79421473  91.16894564] # jhmdb
    # mean = [103.29825354, 104.63845484,  90.79830328]  # jhmdb from .png
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

    # Init action_net
    model = ACT_net(actions, sample_duration)
    model.create_architecture()
    model = nn.DataParallel(model)
    model.to(device)

    # model_data = torch.load('./actio_net_model_both.pwf')
    # model_data = torch.load('./action_net_model_both_without_avg.pwf')
    # model_data = torch.load('./action_net_model_16frm_64.pwf')
    # model_data = torch.load('./action_net_model_both_sgl_frm.pwf')
    model_data = torch.load('./action_net_model_both.pwf')
    # 
    # model_data = torch.load('./action_net_model_part1_1_8frm.pwf')
    model.load_state_dict(model_data)

    # model_data = torch.load('./region_net_8frm.pwf')
    # model.module.act_rpn.load_state_dict(model_data)

    model.eval()

    validation(0, device, model, dataset_folder, sample_duration, spatial_transform, temporal_transform, boxes_file, split_txt_path, cls2idx, 4, n_threads)
