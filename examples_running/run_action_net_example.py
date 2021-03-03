import os
import sys
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

### if file is inside "exaples_running" folder, add folder path to sys
dir_path = os.path.dirname(os.path.realpath(__file__))
if os.path.basename(dir_path) == 'examples_running':
    path_2_append = os.path.dirname(dir_path)
else:
    path_2_append = dir_path
print('--------------------\n'+f'APPENDING TO PATH : \n{path_2_append}\n'+'--------------------\n')
sys.path.append(path_2_append)


from lib.utils.spatial_transforms import (
    Compose, Normalize, Scale, ToTensor)
from lib.utils.temporal_transforms import LoopPadding
from lib.models.action_net import ACT_net

np.random.seed(42)

if __name__ == '__main__':

    from config.dataset_config import cfg as dataset_cfg, set_dataset

    # torch.cuda.device_count()
    print('++++++++++++++++++++++++++++')
    dataset = 'KTH'
    set_dataset(dataset)
    # print('Running for UCF101 Dataset')
    # from lib.dataloaders.ucf_dataset import Video_Dataset

    print('Running for KTH Dataset')
    from lib.dataloaders.kth_dataset import  Video_Dataset

    print('++++++++++++++++++++++++++++')



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    # data_root_dir = os.getcwd()
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
    # mean = [112.07945832, 112.87372333, 106.90993363]  # ucf-101 24 classes
    mean = (0.5,0.5,0.5)
    std  = (0.5,0.5,0.5)
    # generate model
    last_fc = False

    actions = dataset_cfg.dataset.classes

    cls2idx = {actions[i]: i for i in range(0, len(actions))}

    ### get videos id

    spatial_transform = Compose([Scale(sample_size),  # [Resize(sample_size),
                                 ToTensor(),
                                 Normalize(mean, std)])
    temporal_transform = LoopPadding(sample_duration)

    n_classes = len(actions)

    # Init action_net

    model = ACT_net(actions, sample_duration)
    model.create_architecture(model_path)
    model = nn.DataParallel(model)
    model.to(device)

    data = Video_Dataset(dataset_frames, frames_dur=sample_duration, spatial_transform=spatial_transform,
                         temporal_transform=temporal_transform, bboxes_file= boxes_file,
                         split_txt_path=split_txt_path, mode='train', classes_idx=cls2idx)

    # data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
    #                                           shuffle=False, num_workers=n_threads, pin_memory=True)

    clips, h, w, gt_tubes_r, gt_rois, n_actions, n_frames, im_info = data[14].values()
    clips2, h2, w2, gt_tubes_r2, gt_rois2, n_actions2, n_frames2, im_info2 = data[15].values()

    clips_ = clips.unsqueeze(0).to(device)
    gt_tubes_r_ = gt_tubes_r.unsqueeze(0).to(device)
    gt_rois_ = gt_rois.unsqueeze(0).to(device)
    # print('gt_rois_[0,0] :',gt_rois_[0,0,:,:4].shape)
    # print(torch.Tensor([43., 59., 55., 80., 43., 59., 55., 80., 44., 60., 56., 81., 44., 60.,
    #                            56., 81., 32.,  32.,  21.,  21.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
    #                            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
    #                            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                               # 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]).shape)
    # gt_rois_[0,0,:,:4] =  torch.Tensor([43., 59., 55., 80., 43., 59., 55., 80., 44., 60., 56., 81., 44., 60.,
    #                            56., 81., 32.,  32.,  21.,  21.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
    #                            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
    #                            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
    #                                     0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]).view(16,4)

    n_actions_ = torch.from_numpy(n_actions).to(device)
    im_info_ = im_info.unsqueeze(0).to(device)
    start_fr = torch.zeros((1,1)).to(device)

    clips_2 = clips2.unsqueeze(0).to(device)
    gt_tubes_r_2 = gt_tubes_r2.unsqueeze(0).to(device)
    gt_rois_2 = gt_rois2.unsqueeze(0).to(device)
    n_actions_2 = torch.from_numpy(n_actions2).to(device)
    im_info_2 = im_info2.unsqueeze(0).to(device)
    start_fr_2 = torch.zeros((1,1)).to(device)

    clips_ = torch.cat((clips_,clips_2,clips_,clips_2))
    gt_tubes_r_ = torch.cat((gt_tubes_r_, gt_tubes_r_2,gt_tubes_r_, gt_tubes_r_2))
    gt_rois_ = torch.cat((gt_rois_, gt_rois_2,gt_rois_, gt_rois_2))
    n_actions_ = torch.cat((n_actions_, n_actions_2,n_actions_, n_actions_2))
    start_fr = torch.cat((start_fr,start_fr_2,start_fr,start_fr_2))
    im_info_ = torch.Tensor([[112,112,16],[112,112,16],[112,112,16],[112,112,16]]).to(device)

    # clips_ = torch.cat((clips_,clips_2,clips_,clips_2))
    # gt_tubes_r_ = torch.cat((gt_tubes_r_, gt_tubes_r_2,gt_tubes_r_, gt_tubes_r_2))
    # gt_rois_ = torch.cat((gt_rois_, gt_rois_2,gt_rois_, gt_rois_2))
    # n_actions_ = torch.cat((n_actions_, n_actions_2,n_actions_, n_actions_2))
    # start_fr = torch.cat((start_fr,start_fr_2,start_fr,start_fr_2))
    # im_info_ = torch.Tensor([[112,112,16],[112,112,16],[112,112,16],[112,112,16]]).to(device)

    print('clips.shape :',clips.shape)
    print('clips[0].shape :',clips[0].shape)
    img = clips[:,0].permute(1,2,0).numpy().copy()
    print('img.shape :',img.shape)
    print('gt_rois.shape :',gt_rois.shape)
    print('gt_rois[0,0]:',gt_rois[0,0])
    # import cv2
    # cv2.rectangle(img,(int(gt_rois[0,0,0]),int(gt_rois[0,0,1])),(int(gt_rois[0,0,2]),int(gt_rois[0,0,3])),(0,255,0),3)
    # cv2.imwrite('test_img.jpg', img)

    print('gt_tubes_r_.shape :',gt_tubes_r_.shape)
    print('gt_rois_.shape :',gt_rois_.shape)
    print('n_actions_.shape :',n_actions_.shape)
    print('start_fr.shape :',start_fr.shape)
    print('im_info_.shape :',im_info_.shape)
    print('**********Starts**********')
    # exit(-1)
    inputs = Variable(clips_)
    rois, feats, \
    rpn_loss_cls,  rpn_loss_bbox, \
    rpn_loss_cls_16,\
    rpn_loss_bbox_16,  rois_label, \
    sgl_rois_bbox_pred, sgl_rois_bbox_loss,  = model(inputs, \
                                                im_info_,
                                                gt_tubes_r_, gt_rois_,
                                                start_fr)

    print('**********VGIKE**********')
    print('feats.shape :',feats.shape)
    print('rois.shape :',rois.shape)
    print('sgl_rois_bbox_pred.shape :',sgl_rois_bbox_pred.shape)

