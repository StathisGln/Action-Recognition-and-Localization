import numpy as np

import torch

from lib.dataloaders.ucf_dataset import Video_UCF

from lib.utils.spatial_transforms import (
    Compose, Normalize, Scale, ToTensor)
from lib.utils.temporal_transforms import LoopPadding
from region_net import _RPN

np.random.seed(42)

if __name__ == '__main__':

    # torch.cuda.device_count()
    device = torch.device('cpu')

    dataset_frames = '/gpu-data2/sgal/UCF-101-frames'
    boxes_file = '/gpu-data2/sgal/pyannot.pkl'
    split_txt_path = '/gpu-data2/sgal/UCF101_Action_detection_splits/'

    # dataset_frames = '../UCF-101-frames'
    # boxes_file = '../pyannot.pkl'
    # split_txt_path = '../UCF101_Action_detection_splits/'

    sample_size = 112
    sample_duration = 16  # len(images)
    # sample_duration = 8  # len(images)


    batch_size = 1
    n_threads = 0

    # # get mean
    # mean =  [103.75581543 104.79421473  91.16894564] # jhmdb
    # mean = [103.29825354, 104.63845484,  90.79830328]  # jhmdb from .png
    mean = [112.07945832, 112.87372333, 106.90993363]  # ucf-101 24 classes

    # generate model
    last_fc = False

    actions = ['__background__', 'Basketball','BasketballDunk','Biking','CliffDiving','CricketBowling',
               'Diving','Fencing','FloorGymnastics','GolfSwing','HorseRiding','IceDancing',
               'LongJump','PoleVault','RopeClimbing','SalsaSpin','SkateBoarding','Skiing',
               'Skijet','SoccerJuggling','Surfing','TennisSwing','TrampolineJumping',
               'VolleyballSpiking','WalkingWithDog']

    cls2idx = {actions[i]: i for i in range(0, len(actions))}

    ### get videos id

    spatial_transform = Compose([Scale(sample_size),  # [Resize(sample_size),
                                 ToTensor(),
                                 Normalize(mean, [1, 1, 1])])
    temporal_transform = LoopPadding(sample_duration)

    n_classes = len(actions)

    # Init region_net
    model = _RPN(128,sample_duration)
    # model = ACT_net(actions, sample_duration)
    # model.create_architecture()

    model.to(device)

    data = Video_UCF(dataset_frames, frames_dur=sample_duration, spatial_transform=spatial_transform,
                 temporal_transform=temporal_transform, json_file = boxes_file,
                 split_txt_path=split_txt_path, mode='train', classes_idx=cls2idx)

    clips, h, w, gt_tubes_r, gt_rois, n_actions, n_frames, im_info = data[14]
    clips2, h2, w2, gt_tubes_r2, gt_rois2, n_actions2, n_frames2, im_info2 = data[15]
    clips_ = clips.unsqueeze(0).to(device)
    gt_tubes_r_ = gt_tubes_r.unsqueeze(0).to(device)
    gt_rois_ = gt_rois.unsqueeze(0).to(device)
    n_actions_ = torch.from_numpy(n_actions).to(device)
    im_info_ = im_info.unsqueeze(0).to(device)
    start_fr = torch.zeros((1,1)).to(device)

    clips_2 = clips2.unsqueeze(0).to(device)
    gt_tubes_r_2 = gt_tubes_r2.unsqueeze(0).to(device)
    gt_rois_2 = gt_rois2.unsqueeze(0).to(device)
    n_actions_2 = torch.from_numpy(n_actions2).to(device)
    im_info_2 = im_info2.unsqueeze(0).to(device)
    start_fr_2 = torch.zeros((1,1)).to(device)

    clips = torch.cat((clips_,clips_2))
    gt_tubes_r_ = torch.cat((gt_tubes_r_, gt_tubes_r_2))
    gt_rois_ = torch.cat((gt_rois_, gt_rois_2))
    n_actions_ = torch.cat((n_actions_, n_actions_2))
    start_fr = torch.cat((start_fr,start_fr_2))

    print('gt_tubes_r_.shape :',gt_tubes_r_.shape)
    print('gt_rois_.shape :',gt_rois_.shape)
    print('n_actions_.shape :',n_actions_.shape)
    print('start_fr.shape :',start_fr.shape)
    print('**********Starts**********')
    # base_feat = torch.rand([1, 128, 16, 14, 14])
    base_feat = torch.rand([2, 128, 16, 14, 14])
    # ret = model(base_feat,torch.Tensor([[112,112,16]]),gt_tubes_r_, gt_rois_)                       
    ret = model(base_feat,torch.Tensor([[112,112,16],[112,112,16]]),gt_tubes_r_, gt_rois_)                       
    print('**********VGIKE**********')
    # print('rois.shape :',rois.shape)
    # print('rois :',rois)

