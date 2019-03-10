import os
import numpy as np
import glob
from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from action_net import ACT_net
from tcn import TCN

from create_tubes_from_boxes import create_video_tube, create_tube_from_tubes
from connect_tubes import connect_tubes, get_gt_tubes_feats_label, get_tubes_feats_label
from resize_rpn import resize_boxes, resize_tube

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

        self.act_net = ACT_net(actions,sample_duration)

        ## general options
        self.sample_duration = sample_duration
        self.sample_size = sample_size
        self.step = int(self.sample_duration/2)

        # For now a linear classifier only
        self.linear = nn.Linear(512, self.n_classes).cuda()

    def forward(self,n_devs, dataset_folder, vid_names, vid_id, spatial_transform, temporal_transform, boxes, mode, cls2idx, num_actions, num_frames):
        '''
        TODO describe procedure
        '''

        ## define a dataloader for the whole video
        batch_size = n_devs # 

        num_images = 1
        rois_per_image = int(cfg.TRAIN.BATCH_SIZE / num_images) if self.training else 10
        # print('rois_per_image :',rois_per_image)
        boxes = boxes[:,:num_actions, :num_frames].squeeze(0)

        data = single_video(dataset_folder, vid_names, vid_id, frames_dur= self.sample_duration, sample_size =self.sample_size,
                            spatial_transform=spatial_transform, temporal_transform=temporal_transform, boxes=boxes,
                            mode=mode, classes_idx=cls2idx)

        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                                  shuffle=False)
        n_clips = data.__len__()
        max_sim_actions = data.__max_sim_actions__()
        features = torch.zeros(n_clips, rois_per_image, 512, self.sample_duration).cuda()
        p_tubes = torch.zeros(n_clips, rois_per_image,  8).cuda() # all the proposed rois

        f_tubes = []

        if self.training:
            
            f_gt_tubes = torch.zeros(n_clips,num_actions,7).cuda() # gt_tubes
            f_gt_feats = torch.zeros(n_clips,num_actions,7).cuda() # gt_tubes' feat
            tubes_labels = torch.zeros(n_clips,rois_per_image).cuda()  # tubes rois
            loops = int(np.ceil(n_clips / batch_size))

            rpn_loss_cls_  = torch.zeros(loops).cuda() 
            rpn_loss_bbox_ = torch.zeros(loops).cuda()
            act_loss_bbox_ = torch.zeros(loops).cuda()

        for step, dt in enumerate(data_loader):

            # if step == 1:
            #     break
            clips,  (h, w),  gt_tubes, _, im_info, n_acts, start_fr = dt
            clips_ = clips.cuda()
            h_ = h.cuda()
            w_ = w.cuda()
            gt_tubes_ = gt_tubes.type_as(clips_).cuda()
            im_info_ = im_info.cuda()
            n_acts_ = n_acts.cuda()
            start_fr_ = start_fr.cuda()

            tubes,  bbox_pred, pooled_feat, \
            rpn_loss_cls,  rpn_loss_bbox, \
            act_loss_bbox, rois_label = self.act_net(clips,
                                                     im_info,
                                                     gt_tubes.float(),
                                                     None, n_acts,
                                                     start_fr)
            
            # print('tubes.shape :',tubes.shape)
            # print('rpn_loss_cls :',rpn_loss_cls)
            # print('rpn_loss_cls :',rpn_loss_cls.mean())
            # print('rpn_loss_cls.shape :',rpn_loss_cls.shape)
            # print('act_loss_bbox :',act_loss_bbox)
            # print('act_loss_bbox :',act_loss_bbox.mean())
            # print('act_loss_bbox.shape :',act_loss_bbox.shape)

            # print('pooled_feat.shape :',pooled_feat.shape)
            pooled_f = pooled_feat.view(-1,rois_per_image,512,self.sample_duration)
            indexes_ = (torch.arange(0, tubes.size(0))*int(self.sample_duration/2) + start_fr[0]).unsqueeze(1)
            indexes_ = indexes_.expand(tubes.size(0),tubes.size(1)).type_as(tubes)

            tubes[:,:,3] = tubes[:,:,3] + indexes_
            tubes[:,:,6] = tubes[:,:,6] + indexes_

            idx_s = step * batch_size 
            idx_e = step * batch_size + batch_size
            # print('idx_s :',idx_s)
            # print('idx_e :',idx_e)
            # print('pooled_f.shape :',pooled_f.shape)
            # print('features.shape :',features.shape)
            features[idx_s:idx_e] = pooled_f
            p_tubes[idx_s:idx_e] = tubes

            if self.training:

                idx_s_ = step * n_devs 
                idx_e_ = step * n_devs + n_devs
                f_gt_tubes[idx_s:idx_e] = gt_tubes
                tubes_labels[idx_s:idx_e] = rois_label.squeeze(-1)

                rpn_loss_cls_[step] = rpn_loss_cls.mean().unsqueeze(0)
                rpn_loss_bbox_[step] = rpn_loss_bbox.mean().unsqueeze(0)
                act_loss_bbox_[step] = act_loss_bbox.mean().unsqueeze(0)

            # print('----------Out TPN----------')
            # # print('p_tubes.type() :',p_tubes.type())
            # # print('tubes.type() :',tubes.type())
            # print('----------Connect TUBEs----------')

            f_tubes = connect_tubes(f_tubes,tubes, p_tubes, pooled_f, rois_label, step*batch_size)
            # print('----------End Tubes----------')

        ###############################################
        #          Choose Tubes for RCNN\TCN          #
        ###############################################

        torch.cuda.synchronize()
        ## TODO choose tubes layer 
        # print('rpn_loss_cls_ :',rpn_loss_cls_)
        # print('rpn_loss_bbox_ :',rpn_loss_bbox_)
        # print('act_loss_bbox_ :',act_loss_bbox_)

        if self.training:

            f_rpn_loss_cls = rpn_loss_cls_.mean()
            f_rpn_loss_bbox = rpn_loss_bbox_.mean()
            f_act_loss_bbox = act_loss_bbox_.mean()

            ## first get video tube
            video_tubes = create_video_tube(boxes.type_as(clips_))
            video_tubes_r =  resize_tube(video_tubes.unsqueeze(0), h_,w_,self.sample_size)
            
            # get gt tubes and feats
            # print('f_gt_tubes :',f_gt_tubes)
            gt_tubes_feats,gt_tubes_list = get_gt_tubes_feats_label(f_tubes, p_tubes, features, tubes_labels, f_gt_tubes)
            bg_tubes = get_tubes_feats_label(f_tubes, p_tubes, features, tubes_labels, video_tubes_r)

            gt_tubes_list = [x for x in gt_tubes_list if x != []]
            gt_lbl = torch.zeros(len(gt_tubes_list)).type_as(f_gt_tubes)
        
            for i in torch.arange(len(gt_tubes_list)).long().cuda():
                gt_lbl[i] = f_gt_tubes[gt_tubes_list[i][0][0],i,6]
            bg_lbl = torch.zeros((len(bg_tubes))).type_as(f_gt_tubes)
            
            ## concate fb, bg tubes
            f_tubes = gt_tubes_list + bg_tubes
            target_lbl = torch.cat((gt_lbl,bg_lbl),0)

        ##############################################

        # if (len(f_tubes) <2) :
        #     print(f_tubes)
        max_seq = reduce(lambda x, y: y if len(y) > len(x) else x, f_tubes)
        max_length = len(max_seq)
        # print('max_seq :',max_seq)
        # print('max_length :',max_length)

        ## calculate input rois
        ## TODO create tensor
        f_feat_mean = torch.zeros(len(f_tubes),512).cuda() #.to(device)
        final_video_tubes = torch.zeros(len(f_tubes),6).cuda()
        
        for i in range(len(f_tubes)):

            seq = f_tubes[i]

            tmp_tube = torch.Tensor(len(seq),6)
            feats = torch.Tensor(len(seq),512)
            for j in range(len(seq)):
                # print('features[seq[j]].mean(1).shape :',features[seq[j]].mean(1).shape)
                feats[j] = features[seq[j]].mean(1)
                tmp_tube[j] = p_tubes[seq[j]][1:7]
            final_video_tubes[i] = create_tube_from_tubes(tmp_tube)
            # print(feats)
            # print('torch.mean(feats,1) :',torch.mean(feats,0).shape)
            # print('torch.mean(feats,1).unsqueeze(0).shape :',torch.mean(feats,0).unsqueeze(0).shape)
            f_feat_mean[i,:] = torch.mean(feats,0).unsqueeze(0)

        # ### get gt_tubes
        # if self.training:
            
        # ######################################
        # #           Time for Linear          #
        # ######################################

        ## TODO : to add TCN or RNN

        cls_loss = 0
        prob_out = self.linear(f_feat_mean)
        # print('prob_out.shape :',prob_out.shape)
        # # classification probability

        if self.training:
            cls_loss = F.cross_entropy(prob_out, target_lbl.long())

        if self.training:
            return tubes, bbox_pred,  prob_out, f_rpn_loss_cls, f_rpn_loss_bbox, f_act_loss_bbox, cls_loss, 
        else:
            return final_video_tubes, bbox_pred, prob_out

    def create_architecture(self):

        self.act_net.create_architecture()

    def load_part_model(self, action_model_path=None, linear_path=None):


        if action_model_path != None:
            
            act_net = ACT_net(self.actions,self.sample_duration)
            act_net = nn.DataParallel(act_net, device_ids=None)
            act_data = torch.load(action_model_path)
            act_net.load(act_data)
            self.act_net = act_net

        if linear_path != None:

            # linear = nn.Linear(512, self.n_classes).cuda()
            linear = nn.Linear(256, self.n_classes).cuda()
            linear_data = torch.load(linear_path)
            linear.load_state_dict(linear_data)
            self.linear = linear

