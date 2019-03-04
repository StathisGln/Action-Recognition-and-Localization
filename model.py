import os
import numpy as np
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from action_net import ACT_net
from tcn import TCN

from create_tubes_from_boxes import create_tube
from connect_tubes import connect_tubes

from video_dataset import single_video

from config import cfg

class Model(nn.Module):
    """ 
    action localizatio network which contains:
    -ACT_net : a network for proposing action tubes for 16 frames
    -TCN net : a dilation network which classifies the input tubes
    """
    def __init__(self, actions, sample_duration, sample_size):
        super(Model, self).__init__()

        self.classes = actions
        self.n_classes = len(actions)

        self.act_net = ACT_net(actions)

        ## general options
        self.sample_duration = sample_duration
        self.sample_size = sample_size
        self.step = int(self.sample_duration/2)

        # For now a linear classifier only
        self.linear = nn.Linear(512, self.n_classes)

    def forward(self, device, dataset_folder, vid_path, spatial_transform, temporal_transform, boxes_file, mode, cls2idx):
        '''
        TODO describe procedure
        '''

        ## define a dataloader for the whole video
        batch_size = 8

        num_images = 1
        rois_per_image = int(cfg.TRAIN.BATCH_SIZE / num_images)

        data = single_video(dataset_folder, vid_path, self.sample_duration, self.sample_size, spatial_transform=spatial_transform,
                        temporal_transform=temporal_transform, json_file=boxes_file,
                        mode=mode, classes_idx=cls2idx)

        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                              shuffle=False)
        n_clips = data.__len__()
        max_sim_actions = data.__max_sim_actions__()

        features = torch.zeros(n_clips, rois_per_image, 512, self.sample_duration).to(device)
        p_tubes = torch.zeros(n_clips, rois_per_image,  8).to(device) # all the proposed rois

        f_tubes = []
        for step, dt in enumerate(data_loader):
            if step == 1:
                return -1
            clips,  (h, w),  gt_tubes, gt_rois, im_info, n_acts = dt
            clips_ = clips.to(device)
            gt_tubes_ = gt_tubes.to(device)
            im_info_ = im_info.to(device)
            n_acts_ = n_acts.to(device)

            # print('tubes :',gt_tubes_)
            # print('gt_rois :',gt_rois)
            tubes,  bbox_pred, pooled_feat, \
            rpn_loss_cls,  rpn_loss_bbox, \
            act_loss_cls,  act_loss_bbox, rois_label = self.act_net(clips_,
                                                                    im_info_,
                                                                    gt_tubes_,
                                                                    None, n_acts_
                                                                    )
            pooled_f = pooled_feat.view(-1,rois_per_image,512,16)

            idx_s = step * batch_size 
            idx_e = step * batch_size + batch_size

            features[idx_s:idx_e] = pooled_f
            p_tubes[idx_s:idx_e] = tubes

            print('----------Out TPN----------')
            # print('f_tubes.type() :',f_tubes.type())
            print('p_tubes.type() :',p_tubes.type())
            print('tubes.type() :',tubes.type())
            print('----------Connect TUBEs----------')
            tubes = connect_tubes(f_tubes,tubes, p_tubes, pooled_f, rois_label, step)
        return 0

        # for i in indexes:

        #     lim = min(i+self.sample_duration, (n_frames))
        #     vid_indices = torch.arange(i,lim).long()

        #     vid_seg = input_video[:,:,vid_indices]
        #     gt_rois_seg = gt_rois[:,:,vid_indices]
        #     ## TODO remove that and just filter gt_tubes
        #     gt_tubes_seg = create_tube(gt_rois_seg, im_info, 16)
        #     # print('gt_tubes_seg.shape :',gt_tubes_seg.shape)
        #     # print('gt_tubes_seg :',gt_tubes_seg)
        #     # print('gt_tubes :',gt_tubes)
        #     ## run ACT_net
        #     rois,  bbox_pred, rois_feat, \
        #     rpn_loss_cls,  rpn_loss_bbox, \
        #     act_loss_cls,  act_loss_bbox, rois_label = self.act_net(vid_seg,
        #                                                             im_info,
        #                                                             gt_tubes_seg,
        #                                                             gt_rois_seg,
        #                                                             num_boxes)
        #     # print('rois :', rois)
        #     tubes, pooled_feats = connect_tubes(tubes,rois, pooled_feats, rois_feat, rois_label, i)

        # ###################################
        # #           Time for TCN          #
        # ###################################

        # # classification probability
        # cls_prob = torch.zeros((len(tubes),self.n_classes)).type_as(input_video)
        # # classification loss
        # cls_loss = torch.zeros(len(tubes)).type_as(input_video)
        # max_dim = -1
        # target = torch.zeros(len(tubes)).type_as(input_video)
        # for i in range(len(tubes)):
        #     # get tubes to tensor
        #     tubes_t = torch.Tensor(tubes[i]).type_as(input_video)
        #     if (len(tubes[i]) > max_dim):
        #         max_dim = len(tubes[i])
        #     feat = torch.zeros(len(tubes[i]),512,16).type_as(input_video)
        #     feat = Variable(feat)

        #     if self.training:
        #         target[i] = tubes_t[0,7].long()

        #     for j in range(len(pooled_feats[i])):
        #         feat[j] = pooled_feats[i][j]

        #     feat = feat.permute(1,0,2).mean(2).unsqueeze(0)
        #     tmp_prob = self.tcn_net(feat)
        #     cls_prob[i] = F.softmax(tmp_prob,1)

        # if self.training:
        #     target = torch.ceil(target)
        #     cls_loss = F.cross_entropy(cls_prob, target.long())

        # if self.training:
        #     return rois, bbox_pred,  cls_prob, rpn_loss_cls, rpn_loss_bbox, act_loss_bbox, cls_loss
        # else:
        #     return tubes, bbox_pred, cls_prob

    def create_architecture(self):

        self.act_net.create_architecture()
