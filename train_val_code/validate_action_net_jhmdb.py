import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lib.dataloaders.jhmdb_dataset import Video

from lib.utils.spatial_transforms import (
    Compose, Normalize, Scale, ToTensor)
from lib.utils.temporal_transforms import LoopPadding
from lib.models.action_net import ACT_net
from lib.utils.box_functions import tube_overlaps

np.random.seed(42)


def validation(epoch, device, model, dataset_folder, sample_duration, spatial_transform, temporal_transform, boxes_file, splt_txt_path, cls2idx, batch_size, n_threads):

    iou_thresh = 0.5 # Intersection Over Union thresh
    # iou_thresh = 0.1 # Intersection Over Union thresh
    data = Video(dataset_folder, frames_dur=sample_duration, spatial_transform=spatial_transform,
                 temporal_transform=temporal_transform, json_file = boxes_file,
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

        # if step == 1:
        #     break
        print('step :',step)

        clips, h, w, gt_tubes_r, gt_rois, n_actions, n_frames, im_info = data
        clips_   = clips.to(device)
        gt_tubes_r_ = gt_tubes_r.to(device)
        gt_rois_ = gt_rois.float().to(device)
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

        # tubes = tubes.view(-1, sample_duration*4+2)
        # tubes[:,1:-1] = tube_transform_inv(tubes[:,1:-1],\
        #                                    sgl_rois_bbox_pred.view(-1,sample_duration*4),(1.0,1.0,1.0,1.0))
        # tubes = tubes.view(n_tubes,-1, sample_duration*4+2)
        # tubes[:,:,1:-1] = clip_boxes(tubes[:,:,1:-1], im_info, tubes.size(0))

        # print('tubes[0]:',tubes[0])
        # print('tubes[0]:',tubes.shape)
        # exit(-1)
        # print('tubes.cpu().numpy() :',tubes.cpu().numpy())
        # exit(-1)
        # print('gt_rois_[:,0] :',gt_rois_[:,0])

        for i in range(tubes.size(0)): # how many frames we have
            
            tubes_t = tubes[i,:,1:-1].contiguous()
            gt_rois_t = gt_rois_[i,:,:,:4].contiguous().view(-1,sample_duration*4)

            rois_overlaps = tube_overlaps(tubes_t,gt_rois_t)
            gt_max_overlaps_sgl, max_indices = torch.max(rois_overlaps, 0)
            non_empty_indices =  gt_rois_t.ne(0).any(dim=1).nonzero().view(-1)
            n_elems = non_empty_indices.nelement()            

            gt_max_overlaps_sgl_ = torch.where(gt_max_overlaps_sgl > iou_thresh, gt_max_overlaps_sgl, torch.zeros_like(gt_max_overlaps_sgl).type_as(gt_max_overlaps_sgl))

            sgl_detected =  gt_max_overlaps_sgl_[non_empty_indices].ne(0).sum()

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

    dataset_folder = '../JHMDB-act-detector-frames'
    split_txt_path =  '../splits'
    boxes_file = '../poses.json'

    sample_size = 112
    # sample_duration = 8 # len(images)
    sample_duration = 16 # len(images)

    batch_size = 1
    n_threads = 0

    # # get mean
    mean = [103.29825354, 104.63845484,  90.79830328]  # jhmdb from .png

    # generate model
    actions = ['__background__','brush_hair', 'clap', 'golf', 'kick_ball', 'pour',
               'push', 'shoot_ball', 'shoot_gun', 'stand', 'throw', 'wave',
               'catch','climb_stairs', 'jump', 'pick', 'pullup', 'run', 'shoot_bow', 'sit',
               'swing_baseball', 'walk' ]


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
    # model_data = torch.load('./action_net_model_jhmdb_16frm_64.pwf')

    # model_data = torch.load('./action_net_model_both_jhmdb.pwf')
    model_data = torch.load('./action_net_model_16frm_avg_jhmdb.pwf')
    # action_model_path = './action_net_model_jhmdb_16frm_64.pwf'

    model.load_state_dict(model_data)

    # model_data = torch.load('./region_net_8frm.pwf')
    # model.module.act_rpn.load_state_dict(model_data)

    model.eval()


    validation(0, device, model, dataset_folder, sample_duration, spatial_transform, temporal_transform, boxes_file, split_txt_path, cls2idx, 4, n_threads)
