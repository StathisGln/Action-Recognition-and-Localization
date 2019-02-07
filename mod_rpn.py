import os
import numpy as np
import glob
import json

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from resnet_3D import resnet34
from video_dataset import Video
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, ToTensor, Resize)
from temporal_transforms import LoopPadding
from region_net import _RPN
from resize_rpn import resize_rpn, resize_tube
import cv2

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    dataset_folder = '/gpu-data/sgal/UCF-101-frames'
    boxes_file = './pyannot.pkl'
    # boxes_file = '/gpu-data/sgal/UCF-bboxes.json'
    # dataset_folder = '../UCF-101-frames'
    # boxes_file = '../UCF-101-frames/UCF-bboxes.json'

    sample_size = 112
    sample_duration = 16  # len(images)
    batch_size = 1
    n_threads = 0

    # # get mean
    # mean =  [103.75581543 104.79421473  91.16894564] # jhmdb
    mean = [103.29825354, 104.63845484,  90.79830328]  # jhmdb from .png

    # generate model
    last_fc = False

    actions = ['Basketball','BasketballDunk','Biking','CliffDiving','CricketBowling',
               'Diving','Fencing','FloorGymnastics','GolfSwing','HorseRiding','IceDancing',
               'LongJump','PoleVault','RopeClimbing','SalsaSpin','SkateBoarding','Skiing',
               'Skijet','SoccerJuggling','Surfing','TennisSwing','TrampolineJumping',
               'VolleyballSpiking','WalkingWithDog']

    cls2idx = {actions[i]: i for i in range(0, len(actions))}

    scale_size = [sample_size,sample_size]
    spatial_transform = Compose([Scale(sample_size),  # [Resize(sample_size),
                                 ToTensor(),
                                 Normalize(mean, [1, 1, 1])])
    temporal_transform = LoopPadding(sample_duration)

    data = Video(dataset_folder, frames_dur=sample_duration, spatial_transform=spatial_transform,
                 temporal_transform=temporal_transform, json_file=boxes_file,
                 mode='test', classes_idx=cls2idx, scale_size = scale_size)
    # data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
    #                                           shuffle=True, num_workers=n_threads, pin_memory=True)


    # for i in range(700,850):
    #     k = data[i]
    #     print('abs path :',k[4], ' i :',i)
    
    
    # clips,  (h, w), gt_tubes, final_rois = data[906]
    # clips,  (h, w), gt_tubes, final_rois = data[905]
    clips, (h,w), gt_tubes, gt_rois, path,frame_indices = data[1024]
    print('clips.shape :',clips.size())
    clips = clips.permute(1,2,3,0)
    
    # target_h = h
    # target_w = w
    # scale = w > h
    # if scale:
    #     target_h = int(np.round(sample_size * float(w) / float(w)))
    # else:
    #     target_w = int(np.round(sample_size * float(h) / float(h)))

    # top = int(max(0, np.round((sample_size - target_h) / 2)))
    # left = int(max(0, np.round((sample_size - target_w) / 2)))

    # # top = int(max(0, np.round((h - sample_size) / 2)))
    # # left = int(max(0, np.round((w - sample_size) / 2)))
    # # bottom = height - top - target_h
    # # right = width - left - target_w

    # print('top {}, left {}'.format(top,left))
    # print('w {}, h {}, sample {}'.format(w, h, sample_size ))
    # gt_rois[:,:,0] = (( gt_rois[:,:,0]-left ) * sample_size/w).clamp_(min=0)
    # gt_rois[:,:,1] = (( gt_rois[:,:,1]-top  ) * sample_size/h).clamp_(min=0)
    # gt_rois[:,:,2] = (( gt_rois[:,:,2]-left ) * sample_size/w).clamp_(min=0)
    # gt_rois[:,:,3] = (( gt_rois[:,:,3]-top  ) * sample_size/h).clamp_(min=0)

    # gt_rois_r[:,:,0] = gt_rois_r[:,:,0]-left
    # gt_rois_r[:,:,1] = gt_rois_r[:,:,1]-top
    # gt_rois_r[:,:,2] = gt_rois_r[:,:,2]-left
    # gt_rois_r[:,:,3] = gt_rois_r[:,:,3]-top

    gt_rois_r = resize_rpn(gt_rois, h,w, sample_size)
    gt_tubes_r = resize_tube(gt_tubes, h,w, sample_size)
    print('gt_tubes_r.shape :', gt_tubes_r.shape)
    for i in range(len(frame_indices)):
        t = clips[i].cpu().numpy()
        img_tmp = t.copy()
        # if t.all():
        #     print('Image {} not found '.format(os.path.join(path, 'image_{:0>5}.jpg'.format(frame_indices[i]))))
        #     break
        # for j in range(10):
        #     cv2.rectangle(img_tmp,(int(gt_rois[i,j,0]),int(rois[i,j,1])),(int(rois[i,j,2]),int(rois[i,j,3])), (255,0,0),3)
        # # print('out : ./out/{:0>3}.jpg'.format(i))
        # cv2.imwrite('./out_frames/action_{:0>3}.jpg'.format(i), img_tmp)
        # img_tmp = img.copy()
        print('rois.shape :',gt_rois.shape)
        cv2.rectangle(img_tmp,(int(gt_rois_r[0,i,0]),int(gt_rois_r[0,i,1])),(int(gt_rois_r[0,i,2]),int(gt_rois_r[0,i,3])), (255,0,0),5)
        cv2.imwrite('./out_frames/rescaled_rois_{:0>3}.jpg'.format(i), img_tmp)
        img_tmp = t.copy()
        print('gt_tubes :',gt_tubes)
        cv2.rectangle(img_tmp,(int(gt_tubes_r[0,0]),int(gt_tubes_r[0,1])),(int(gt_tubes_r[0,3]),int(gt_tubes_r[0,4])), (0,255,0),3)
        cv2.imwrite('./out_frames/rescaled_tubes_{:0>3}.jpg'.format(i), img_tmp)


    # print('path :',path)
    # print('clips.shape :',clips.shape)
    # clips = clips.unsqueeze(0)
    # gt_tubes = gt_tubes.unsqueeze(0)
    # print('gt_rois.shape :',gt_rois.shape)
    # print('gt_rois :', gt_rois)

    # # print('h :', h, ' w :', w)
    # # print('gt_tubes :', gt_tubes)
    # # print('final_rois :', final_rois)
    # # print('type final_rois: ', type(final_rois))

    # # n_classes = len(classes)
    # n_classes = len(actions)
    # resnet_shortcut = 'A'

    # model = resnet34(num_classes=400, shortcut_type=resnet_shortcut,
    #                  sample_size=sample_size, sample_duration=sample_duration,
    #                  last_fc=last_fc)
    # model = model.cuda()
    # model = nn.DataParallel(model, device_ids=None)

    # model_data = torch.load('../temporal_localization/resnet-34-kinetics.pth')
    # model.load_state_dict(model_data['state_dict'])
    # model.eval()

    # lr = 0.001
    # rpn_model = _RPN(256).cuda()
    # rpn_data = torch.load('./rpn_model_pre_004.pwf')
    # rpn_model.load_state_dict(rpn_data)
    # rpn_model.eval()
    # # rpn_model = _RPN(512).cuda()

    # inputs = Variable(clips).cuda()
    # outputs = model(inputs)
    # outputs_list = outputs.tolist()

    # with open('./outputs.json', 'w') as fp:
    #     json.dump(outputs_list, fp)
    
    # rois, rpn_loss_cls, rpn_loss_box = rpn_model(outputs,
    #                                              torch.Tensor(
    #                                                  [[h, w]] * gt_tubes.size(1)).cuda(),
    #                                              None,None,None)
    # print('h %d w %d ' % (h,w))
    # rois[:,[0,2]] =rois[:,[0,2]].clamp_(min=0, max=w)
    # rois[:,[1,3]] =rois[:,[1,3]].clamp_(min=0, max=h)
    # print('rois.shape :',rois.shape)
    # rois = rois[:,:-1]
    # print('rois.shape :',rois.shape)

    # rois = rois.view(300,16,-1).permute(1,0,2).cpu().numpy()
    # colors = [ (255,0,0), (0,255,0), (0,0,255)]
    # print('rois.shape :',rois.shape)
    # for i in range(len(frame_indices)):
    #     img = cv2.imread(os.path.join(path, 'image_{:0>5}.jpg'.format(frame_indices[i])))
    #     img_tmp = img.copy()
    #     if img.all():
    #         print('Image {} not found '.format(os.path.join(path, 'image_{:0>5}.jpg'.format(frame_indices[i]))))
    #         break
    #     for j in range(10):
    #         cv2.rectangle(img_tmp,(int(rois[i,j,0]),int(rois[i,j,1])),(int(rois[i,j,2]),int(rois[i,j,3])), (255,0,0),3)
    #     # print('out : ./out/{:0>3}.jpg'.format(i))
    #     cv2.imwrite('./out_frames/action_{:0>3}.jpg'.format(i), img_tmp)
    #     img_tmp = img.copy()
    #     print('rois.shape :',gt_rois.shape)
    #     cv2.rectangle(img_tmp,(int(gt_rois[0,i,0]),int(gt_rois[0,i,1])),(int(gt_rois[0,i,2]),int(gt_rois[0,i,3])), (255,0,0),5)
    #     # cv2.imwrite('./out_frames/gt_rois_{:0>3}.jpg'.format(i), img_tmp)
    #     # img_tmp = img.copy()
    #     # print('gt_tubes :',gt_tubes)
    #     cv2.rectangle(img_tmp,(int(gt_tubes[0,0,0]),int(gt_tubes[0,0,1])),(int(gt_tubes[0,0,3]),int(gt_tubes[0,0,4])), (0,255,0),3)
    #     cv2.imwrite('./out_frames/gt_both_{:0>3}.jpg'.format(i), img_tmp)

