import json
import os
import torch
"""
This file is not currently being used. It has been left just for debugging purpose
"""
##TODO finish this file

from lib.utils.spatial_transforms import (
    Compose, Normalize, Scale, ToTensor)
from lib.utils.temporal_transforms import LoopPadding
from lib.models.region_net import _RPN

if __name__ == '__main__':

    from config.dataset_config import cfg as dataset_cfg, set_dataset

    # torch.cuda.device_count()
    print('++++++++++++++++++++++++++++')
    dataset = 'KTH'
    set_dataset(dataset)
    # print('Running for UCF101 Dataset')
    # from lib.dataloaders.ucf_dataset import Video_Dataset

    print('Running for KTH Dataset')
    from lib.dataloaders.kth_dataset import  Video_Dataset_small_clip
    print('++++++++++++++++++++++++++++')

    # torch.cuda.device_count()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    dataset_folder = '/gpu-data/sgal/UCF-101-frames'
    boxes_file = './pyannot.pkl'

    ## TODO fix paths
    data_root_dir = '../../thesis_code/sgal/'
    data_root_dir2  = './'

    model_path = os.path.abspath(os.path.join(data_root_dir, 'resnet-34-kinetics.pth'))
    dataset_frames = os.path.abspath(os.path.join(data_root_dir2,dataset_cfg.dataset.dataset_frames_folder))

    boxes_file = os.path.abspath(os.path.join(data_root_dir2,dataset_cfg.dataset.boxes_file))
    split_txt_path = os.path.abspath(os.path.join(data_root_dir2, dataset_cfg.dataset.split_txt_path))

    n_devs = torch.cuda.device_count()
    sample_size = 112
    sample_duration = 16  # len(images)
    # sample_duration = 8  # len(images)
    # sample_duration = 4  # len(images)

    batch_size = 1
    n_threads = 0

    # # get mean
    # mean =  [103.75581543 104.79421473  91.16894564] # jhmdb
    # mean = [103.29825354, 104.63845484,  90.79830328]  # jhmdb from .png
    mean = (0.5,0.5,0.5)
    std  = (0.5,0.5,0.5)


    actions = dataset_cfg.dataset.classes
    cls2idx = {actions[i]: i for i in range(0, len(actions))}

    ## Init model
    rpn_model = _RPN(256).cuda()
    rpn_model = nn.DataParallel(rpn_model)
    rpn_model.to(device)

    ### get videos id
    spatial_transform = Compose([Scale(sample_size),  # [Resize(sample_size),
                                 ToTensor(),
                                 Normalize(mean, std)])
    temporal_transform = LoopPadding(sample_duration)


    data = Video_Dataset_small_clip(dataset_frames, frames_dur=sample_duration, spatial_transform=spatial_transform,
                                    temporal_transform=temporal_transform, bboxes_file= boxes_file,
                                    split_txt_path=split_txt_path, mode='train', classes_idx=cls2idx)

    clips, h, w, gt_tubes_r, gt_rois, n_actions, n_frames, im_info = data[14].values()
    clips2, h2, w2, gt_tubes_r2, gt_rois2, n_actions2, n_frames2, im_info2 = data[15].values()

    clips_ = clips.unsqueeze(0).to(device)
    gt_tubes_r_ = gt_tubes_r.unsqueeze(0).to(device)
    gt_rois_ = gt_rois.unsqueeze(0).to(device)

    clips,  (h, w), gt_tubes, gt_rois = data[1451]

    # print('clips.shape :',clips.shape)
    clips = clips.unsqueeze(0)

    gt_tubes = gt_tubes.unsqueeze(0)
    gt_rois = gt_rois.unsqueeze(0)
    print('gt_tubes.shape :',gt_tubes.shape)
    print('gt_rois.shape :',gt_rois.shape)

    clis = clips.cuda()
    gt_tubes = gt_tubes.cuda()
    gt_rois = gt_rois.cuda()

    with open('./outputs.json', 'r') as fp:
        data = json.load( fp)
        outputs = torch.Tensor(data).cuda()

    rois, rpn_loss_cls, rpn_loss_box = rpn_model(outputs,
                                                 torch.Tensor(
                                                     [[h, w, sample_duration]] * gt_tubes.size(1)).cuda(),
                                                 gt_tubes.cuda(), gt_rois, len(gt_tubes))
