import os
import numpy as np
import glob
from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from conf import conf
from action_net import ACT_net
from act_rnn import Act_RNN
from all_scores.calc import Calculator
from translate_rois.calc import Translate_Calculator

from create_tubes_from_boxes import create_video_tube, create_tube_from_tubes, create_tube_with_frames
from connect_tubes import connect_tubes, get_gt_tubes_feats_label, get_tubes_feats_label
from resize_rpn import resize_boxes, resize_tube

from ucf_dataset import single_video

from bbox_transform import bbox_overlaps_connect
from collections import OrderedDict
from box_functions import bbox_transform, tube_transform_inv, clip_boxes, tube_overlaps
from imdetect import scoreTubes
from nms_3d_whole_video.py_nms import py_cpu_nms_tubes
import time

class Model(nn.Module):
    """ 
    action localizatio network which contains:
    -ACT_net : a network for proposing action tubes for 16 frames
    -TCN net : a dilation network which classifies the input tubes
    """
    def __init__(self, actions, sample_duration, sample_size, ):
        super(Model, self).__init__()

        self.classes = actions
        self.n_classes = len(actions)

        ## general options
        self.sample_duration = sample_duration
        self.sample_size = sample_size

        self.step = sample_duration
        # self.p_feat_size = 64 # 128 # 256 # 512
        # self.p_feat_size = 128 # 256 # 512
        self.p_feat_size = 256 # 512
        self.pooling_time = 2
        self.POOLING_SIZE = 7
        # self.POOLING_SIZE = 4

        # For connection 
        self.max_num_tubes = conf.MAX_NUMBER_TUBES
        self.connection_thresh = conf.CONNECTION_THRESH
        self.update_thresh = conf.UPDATE_THRESH

        self.n_ret_tubes = conf.ALL_SCORES_THRESH

        print('sample_duration :',sample_duration, ' self.step :', self.step)
        print('self.n_ret_tubes :', self.n_ret_tubes)
        self.calc = Calculator(self.max_num_tubes, self.update_thresh, self.connection_thresh, self.n_ret_tubes)
        self.translator = Translate_Calculator(self.sample_duration, self.step)
        # for nms
        self.post_nms_topN = 50
        
    def forward(self,n_devs, dataset_folder, vid_names, clips, vid_id, boxes, mode, cls2idx, num_actions, num_frames, h_, w_, target):
        '''
        TODO describe procedure
        '''
        # start = time.time()
        clips = clips.squeeze(0)
        ret_n_frames = clips.size(0)
        clips = clips[:num_frames]
        if self.training:

            # boxes = boxes.squeeze(0).permute(1,0,2).cpu()
            boxes = boxes.squeeze(0).permute(1,0,2)
            boxes = boxes[:num_frames,:num_actions].clamp_(min=0)

        batch_size = 4 # 
        num_images = 1
        rois_per_image = int(conf.TRAIN.BATCH_SIZE / num_images) if self.training else conf.TEST.RPN_POST_NMS_TOP_N

        data = single_video(dataset_folder,h_,w_, vid_names, vid_id, frames_dur= self.sample_duration, sample_size =self.sample_size,step=self.step,
                            classes_idx=cls2idx, n_frames=num_frames)

        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, pin_memory=False,# num_workers=num_workers, pin_memory=True,
                                                  # shuffle=False, num_workers=8)
                                                  shuffle=False)

        n_clips = data.__len__()
        max_clips = 5

        features = torch.zeros(max_clips, rois_per_image, self.p_feat_size, self.sample_duration,self.POOLING_SIZE, self.POOLING_SIZE).type_as(clips)
        p_tubes = torch.zeros(max_clips, rois_per_image,  self.sample_duration*4).type_as(clips) # all the proposed tube-rois
        # print('features.device :',features.device)
        # print('p_tubes.device :',p_tubes.device)

        actioness_score = torch.zeros(max_clips, rois_per_image).type_as(clips)
        overlaps_scores = torch.zeros(max_clips, rois_per_image, rois_per_image).type_as(clips)
        # print('actioness_score.device :',actioness_score.device)
        # print('overlaps_scores.device :',overlaps_scores.device)


        pos_shape = [rois_per_image for i in range(n_clips)] + [n_clips,2]

        if self.training:
            
            f_gt_tubes = torch.zeros(n_clips,num_actions,self.sample_duration*4) # gt_tubes
            tubes_labels = torch.zeros(n_clips,rois_per_image)  # tubes rois
            loops = int(np.ceil(n_clips / batch_size))
            labels = torch.zeros(num_actions)

            for i in range(num_actions):

                idx = boxes[:,i,4].nonzero().view(-1)
                labels[i] = boxes[idx[0],i,4]

        ## Init connect thresh
        self.calc.thresh = self.connection_thresh

        init_time = time.time()

        for step, dt in enumerate(data_loader):

            # if step == 1:
            #     break
            # print('\tstep :',step)

            frame_indices, im_info, start_fr = dt
            clips_ = clips[frame_indices].cuda()

            if self.training:
                boxes_ = boxes[frame_indices].cuda()
                box_ = boxes_.permute(0,2,1,3).float().contiguous()[:,:,:,:-1]
            else:
                box_ = None
                
            im_info = im_info.cuda()
            start_fr = start_fr.cuda()
            
            with torch.no_grad():

                tubes, pooled_feat, \
                rpn_loss_cls,  rpn_loss_bbox, \
                _,_, rois_label, \
                sgl_rois_bbox_pred, sgl_rois_bbox_loss = self.act_net(clips_.permute(0,2,1,3,4),
                                                            im_info,
                                                            None,
                                                            box_,
                                                            start_fr)

            pooled_feat = pooled_feat.view(-1,rois_per_image,self.p_feat_size,self.sample_duration, self.POOLING_SIZE, self.POOLING_SIZE)

            # # regression
            n_tubes = len(tubes)

            if not self.training:
                tubes = tubes.view(-1, self.sample_duration*4+2)
                tubes[:,1:-1] = tube_transform_inv(tubes[:,1:-1],\
                                               sgl_rois_bbox_pred.view(-1,self.sample_duration*4),(1.0,1.0,1.0,1.0))
                tubes = tubes.view(n_tubes,rois_per_image, self.sample_duration*4+2)
                tubes[:,:,1:-1] = clip_boxes(tubes[:,:,1:-1], im_info, tubes.size(0))

            idx_s = step * batch_size 
            idx_e = min(step * batch_size + batch_size, n_clips)

            # TODO uncomment
            features[idx_s:idx_e] = pooled_feat
            p_tubes[idx_s:idx_e,] = tubes[:,:,1:-1]
            actioness_score[idx_s:idx_e] = tubes[:,:,-1]

            if self.training:

                box = boxes_.permute(0,2,1,3).contiguous()[:,:,:,:-2]
                box = box.contiguous().view(box.size(0),box.size(1),-1)

                f_gt_tubes[idx_s:idx_e] = box


            # connection algo
            for i in range(idx_s, idx_e):
                if i == 0:

                    continue
                
                # calculate overlaps
                overlaps_ = tube_overlaps(p_tubes[i-1,:,-1*4:],p_tubes[i,:,:1*4]).type_as(p_tubes)  #
                overlaps_scores[i-1] = overlaps_

        proposals_time = time.time()
        print('proposals_time :',proposals_time-init_time)
        
        # calculate 500 best tubes
        final_combinations, conn_scores = self.calc(torch.Tensor([n_clips]),torch.Tensor([rois_per_image]),
                                actioness_score, overlaps_scores)
        final_combinations = final_combinations.view(-1,n_clips,2)

        # print('connect :',time.time()-proposals_time)
        final_tubes= self.translator(torch.Tensor([n_clips]), torch.Tensor([rois_per_image]),torch.Tensor([num_frames]),\
                                                       p_tubes, final_combinations.int().contiguous())
        print('final_tubes :',final_tubes.shape)
        translate_time = time.time()
        print('translate_time :',translate_time-proposals_time)
        # ##############################################
        # ## NMS time

        # # print('final_tubes.shape :',final_tubes.shape)
        # # print('conn_scores.shape :',conn_scores.shape)
        # fin_tubes =final_tubes.view(-1,num_frames*4).contiguous()
        # # # print('final_tubes.shape :',fin_tubes.shape)
        # # # print('torch.cat([fin_tubes,conn_scores.unsqueeze(1)], dim=1).shape :',\
        # # #       torch.cat([fin_tubes,conn_scores.unsqueeze(1)], dim=1).shape)
        # # keep_idx_i = nms_gpu.nms_gpu(torch.cat([fin_tubes,conn_scores.unsqueeze(1)], dim=1), 0.7)
        # keep_idx_i = py_cpu_nms_tubes(torch.cat([fin_tubes,conn_scores.unsqueeze(1)], dim=1).cpu().numpy(), 0.7)
        # keep_idx_i = torch.Tensor(keep_idx_i).cuda().long().view(-1)

        # if self.post_nms_topN > 0:
        #     keep_idx_i = keep_idx_i[:self.post_nms_topN]

        # final_tubes = final_tubes[keep_idx_i]
        # final_combinations = final_combinations[keep_idx_i]
        # ##############################################

        prob_out = torch.zeros( self.n_ret_tubes,self.n_classes).type_as(clips)

        for i in range(final_combinations.size(0)):

            t = final_combinations[i].ne(-1).all(dim=1).nonzero().view(-1)
            if t.numel() < 1:
                continue
            indices = final_combinations[i,t].long()

            feat_rnn = features[indices[:,0],indices[:,1]].mean(0).unsqueeze(0)
            # feat_rnn = features[indices[:,0],indices[:,1]].max(0)[0].unsqueeze(0)            

            ret = self.act_rnn(feat_rnn.view(feat_rnn.size(0),-1))
            prob_out[i] = self.act_rnn(feat_rnn.view(feat_rnn.size(0),-1))

        # ##########################################
        # #           Time for Linear Loss         #
        # ##########################################

        if self.training:

            return None, None,  cls_loss, 

        else:

            prob_out = F.softmax(prob_out,1)
            # init padding tubes because of multi-GPU system
            if final_tubes.size(0) > conf.ALL_SCORES_THRESH:
                _, indices = torch.sort(conn_scores)
                final_tubes = final_tubes[indices[:conf.ALL_SCORES_THRESH]].contiguous()
                prob_out = prob_out[indices[:conf.ALL_SCORES_THRESH]].contiguous()

            ret_tubes = torch.zeros(1,conf.ALL_SCORES_THRESH, ret_n_frames,4).type_as(final_tubes).float() -1
            ret_prob_out = torch.zeros(1,conf.ALL_SCORES_THRESH,self.n_classes).type_as(final_tubes).float() - 1

            ret_tubes[0,:final_tubes.size(0),:num_frames] = final_tubes
            ret_prob_out[0,:prob_out.size(0)] = prob_out

            print('rest :',time.time() - translate_time)
            return ret_tubes, ret_prob_out, torch.Tensor([final_tubes.size(0)]).cuda()
        

    def temporal_pool(self, features):

        n_filters = features.size(1)
        x_axis = features.size(3)
        y_axis = features.size(4)
        
        indexes = torch.linspace(0, features.size(0), self.pooling_time+1).int()
        ret_feats = torch.zeros(self.pooling_time, n_filters,\
                                self.sample_duration,x_axis, y_axis)

        if features.size(0) < self.pooling_time:

            ret_feats[:features.size(0)] = features
            return ret_feats
    
        for i in range(self.pooling_time):
            
            t = features[indexes[i]:indexes[i+1]].permute(1,0,2,3,4)
            t = t.view(t.size(0),t.size(1),t.size(2),-1)
            t = F.max_pool3d(t,kernel_size=(indexes[i+1]-indexes[i],1,1)).squeeze()

            ret_feats[i] = t.view(t.size(0),t.size(1), features.size(3),features.size(4))

        return ret_feats

    def deactivate_action_net_grad(self):

        for p in self.act_net.parameters() : p.requires_grad=False
        # self.act_net.eval()
        # for key, value in dict(self.named_parameters()).items():
        #     print(key, value.requires_grad)

    def load_part_model(self, action_model_path=None, rnn_path=None):

        # load action net
        if action_model_path != None:
            
            act_data = torch.load(action_model_path)

            ## to remove module
            new_state_dict = OrderedDict()
            for k, v in act_data.items():
                # if k.find('module') != -1 :
                name = k[7:] # remove `module.`
                new_state_dict[name] = v

            act_net = ACT_net(self.classes,self.sample_duration)

            act_net.create_architecture()
            act_net.load_state_dict(new_state_dict)
            self.act_net = act_net

        else:
            self.act_net = ACT_net(self.classes,self.sample_duration)
            self.act_net.create_architecture()
            
        # load lstm
        if rnn_path != None:

            act_rnn = nn.Sequential(
                nn.Linear(self.p_feat_size*self.sample_duration*self.POOLING_SIZE*self.POOLING_SIZE, self.n_classes),
                # nn.ReLU(True),
                # nn.Dropout(0.8),
                # nn.Linear(256,self.n_classes),
                # nn.Linear(n_inputs, n_outputs),
                # nn.ReLU(True),
                # nn.Dropout(0.8),
                # nn.Linear(256,self.n_classes),

            )

            act_rnn_data = torch.load(rnn_path)
            print('act_rnn_data :',act_rnn_data.keys())
            act_rnn.load_state_dict(act_rnn_data)
            self.act_rnn = act_rnn

        else:

            self.act_rnn = nn.Sequential(
                nn.Linear(self.p_feat_size*self.sample_duration*self.POOLING_SIZE*self.POOLING_SIZE, self.n_classes),
                # nn.ReLU(True),
                # nn.Dropout(0.8),
                # nn.Linear(256,self.n_classes),
                # nn.Linear(n_inputs, n_outputs),
                # nn.ReLU(True),
                # nn.Dropout(0.8),
                # nn.Linear(256,self.n_classes),

            )

        
