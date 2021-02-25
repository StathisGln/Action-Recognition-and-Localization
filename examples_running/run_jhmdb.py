import numpy as np

import torch
import torch.nn as nn

from lib.dataloaders.simple_dataset import Video
from lib.utils.spatial_transforms import (
    Compose, Normalize, Scale, ToTensor)
from lib.utils.temporal_transforms import LoopPadding
from lib.models.model import Model

np.random.seed(42)

if __name__ == '__main__':

    # torch.cuda.device_count()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    dataset_folder = '../JHMDB-act-detector-frames'
    split_txt_path =  '../splits'
    boxes_file = '../poses.json'

    sample_size = 112
    sample_duration = 16  # len(images)

    batch_size = 1
    n_threads = 0

    # # get mean
    mean = [103.29825354, 104.63845484,  90.79830328]  # jhmdb from .png

    # generate model
    classes = ['__background__','brush_hair', 'clap', 'golf', 'kick_ball', 'pour',
               'push', 'shoot_ball', 'shoot_gun', 'stand', 'throw', 'wave',
               'catch','climb_stairs', 'jump', 'pick', 'pullup', 'run', 'shoot_bow', 'sit',
               'swing_baseball', 'walk' ]


    cls2idx = {classes[i]: i for i in range(0, len(classes))}

    spatial_transform = Compose([Scale(sample_size),  # [Resize(sample_size),
                                 ToTensor(),
                                 Normalize(mean, [1, 1, 1])])
    temporal_transform = LoopPadding(sample_duration)

    data = Video(dataset_folder, frames_dur=sample_duration, spatial_transform=spatial_transform,
                 temporal_transform=temporal_transform, json_file = boxes_file,
                 split_txt_path=split_txt_path, mode='train', classes_idx=cls2idx)
    # data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
    #                                           shuffle=True, num_workers=n_threads, pin_memory=True)

    n_classes = len(classes)

    # Init action_net
    model = Model(classes, sample_duration, sample_size)
    model.create_architecture()
    if torch.cuda.device_count() > 1:
        print('Using {} GPUs!'.format(torch.cuda.device_count()))

        model = nn.DataParallel(model)

    model.to(device)


    clips, (h,w), target, gt_tubes, im_info, n_frames = data[24]
    # clips, (h,w), target, gt_tubes, im_info, n_frames = data[144]

    # clips = torch.stack((clips,clips),dim=0).to(device) 

    # clips = torch.stack((clips,clips),dim=0).to(device)
    # gt_tubes = torch.stack((gt_tubes_r,gt_tubes2_r),dim=0).to(device)
    # n_actions = torch.Tensor((n_actions,n_actions2)).to(device)
    # im_info = torch.Tensor([[sample_size, sample_size, sample_duration]] * gt_tubes.size(1)).to(device)

    # clips = torch.stack((clips,clips),dim=0).to(device)
    # gt_tubes = torch.stack((gt_tubes_r,gt_tubes2_r),dim=0).to(device)
    # n_actions = torch.Tensor((n_actions,n_actions2)).to(device)
    clips_t = clips.unsqueeze(0).to(device)
    target_t = torch.Tensor([target]).unsqueeze(0).to(device)
    gt_tubes_t = torch.from_numpy(gt_tubes).float().unsqueeze(0).to(device)
    im_info_t = im_info.unsqueeze(0).to(device)
    n_frames_t = torch.Tensor([n_frames]).long().unsqueeze(0).to(device)
    num_boxes = torch.Tensor([[1],[1],[1]]).unsqueeze(0).to(device)

    print('clips :',clips_t.shape)
    print('clips.type() :',clips.type())

    print('target_t :',target_t)
    print('target_t.type() :',target_t.type())
    print('target_t :',target_t.shape)

    print('gt_tubes :',gt_tubes_t)
    print('gt_tubes.type() :',gt_tubes_t.type())
    print('gt_tubes.shape :',gt_tubes_t.shape)

    print('im_info :',im_info_t)
    print('im_info.type() :',im_info.type())
    print('im_info.shape :',im_info_t.shape)

    print('n_frames_t :',n_frames_t)
    print('n_frames_t.type() :',n_frames_t.type())
    print('n_frames_t.shape :',n_frames_t.shape)

    print('num_boxes :', num_boxes)
    print('num_boxes.shape :', num_boxes.shape)

    print('**********Start**********')
    ret = model(clips_t, target_t,
                im_info_t,
                gt_tubes_t, None,
                n_frames_t, num_boxes, max_dim=1, phase=2)

    print('**********VGIKE**********')
    # print('rois.shape :',rois.shape)
    # print('rois :',rois)

