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
# from action_net_withoutbn import ACT_net

from rest_model import RestNet

# from calc_score_v2.calc import Calculator
from calc_score_v3.calc import Calculator
from translate_rois.calc import Translate_Calculator

from create_tubes_from_boxes import create_video_tube, create_tube_from_tubes, create_tube_with_frames
from connect_tubes import connect_tubes, get_gt_tubes_feats_label, get_tubes_feats_label
from resize_rpn import resize_boxes, resize_tube

from ucf_dataset import single_video

from bbox_transform import bbox_overlaps_connect
from collections import OrderedDict
from box_functions import bbox_transform, tube_transform_inv, clip_boxes, tube_overlaps
# from nms_3d_whole_video.nms_gpu import nms_gpu

from nms_tubes.py_nms import py_cpu_nms_tubes
# from soft_nms_3d.py_nms import py_cpu_softnms 
import time 

class Model(nn.Module):
    """ 
    action localization network which contains:
    -ACT_net : a network for proposing action tubes for 16 frames
    -TCN net : a dilation network which classifies the input tubes
    """
    def __init__(self, actions, sample_duration, sample_size, ):
        super(Model, self).__init__()

        self.classes = actions
        self.n_classes = len(actions)

        # self.act_net = ACT_net(actions,sample_duration)

        ## general options
        self.sample_duration = sample_duration
        self.sample_size = sample_size
        self.step = sample_duration

        # self.p_feat_size = 64 # 128 # 256 # 512
        self.p_feat_size  = 256 # 512


        self.POOLING_SIZE = 7
        self.n_ret_tubes = conf.CALC_THRESH
        self.pooling_time = conf.POOLING_TIME
        print('self.pooling_time :',self.pooling_time)
        # For connection 
        self.max_num_tubes = conf.MAX_NUMBER_TUBES
        self.connection_thresh = conf.CONNECTION_THRESH
        self.update_thresh = conf.UPDATE_THRESH
        self.final_scores_update = conf.FINAL_SCORES_UPDATE
        self.final_scores_keep = conf.FINAL_SCORES_KEEP

        self.pre_nms_tubes = conf.MODEL_PRE_NMS_TUBES
        self.post_nms_tubes = conf.MODEL_POST_NMS_TUBES
        self.final_scores_max_num = conf.FINAL_SCORES_MAX_NUM

        print('self.final_scores_max_num :',self.final_scores_max_num)
        print('self.pre_nms_tubes :',self.pre_nms_tubes)
        print('self.post_nms_tubes :',self.post_nms_tubes)
        print('self.max_num_tubes :',self.max_num_tubes)
        print('self.update_thresh :',self.update_thresh)
        print('self.connection_thresh :',self.connection_thresh)
        print('self.final_scores_update :',self.final_scores_update )
        print('self.final_scores_keep :',self.final_scores_keep)
        # self.calc = Calculator(self.max_num_tubes, self.update_thresh, self.connection_thresh, self.final_scores_update, self.final_scores_keep)
        self.calc = Calculator(self.max_num_tubes, self.update_thresh, self.connection_thresh, self.final_scores_update, self.pre_nms_tubes)
        self.translator = Translate_Calculator(self.sample_duration, self.step)


    def forward(self,n_devs, dataset_folder, vid_names, clips, vid_id, boxes, mode, cls2idx, num_actions, num_frames, h_, w_, target):
        '''
        TODO describe procedure
        '''

        batch_size = clips.size(0)
        ret_n_frames = clips.size(1)

        max_n_frames = torch.max(num_frames).long()

        clips = clips[:,:max_n_frames]

        if self.training:

            max_n_actions = torch.max(num_actions).long()
            boxes = boxes.permute(0,2,1,3).cpu()
            boxes = boxes[:,:max_n_frames,:max_n_actions].clamp_(min=0)
            self.step = self.sample_duration-1


        # data_loader_batch_size = 16 # 
        data_loader_batch_size = 8 # 
        num_images = 1
        print('vid_names[vid] :',vid_names[vid_id[0]], vid_id.device, num_frames)

        rois_per_image = int(conf.TRAIN.BATCH_SIZE / num_images) if self.training else int(conf.TEST.RPN_POST_NMS_TOP_N)
        data = single_video(dataset_folder,h_,w_, vid_names, vid_id, frames_dur= self.sample_duration, \
                            sample_size =self.sample_size,step=self.step,
                            classes_idx=cls2idx, n_frames=max_n_frames.item())

        data_loader = torch.utils.data.DataLoader(data, batch_size=data_loader_batch_size, pin_memory=False,
                                                  shuffle=False)
        n_clips = data.__len__()


        # features = torch.zeros(batch_size, n_clips, rois_per_image, self.p_feat_size,\
        #                        self.sample_duration,self.POOLING_SIZE, self.POOLING_SIZE).type_as(clips)
        features = torch.zeros(batch_size, n_clips, rois_per_image, self.p_feat_size,\
                               self.POOLING_SIZE, self.POOLING_SIZE).type_as(clips)


        p_tubes = torch.zeros(batch_size, n_clips, rois_per_image,  self.sample_duration*4).type_as(clips) # all the proposed tube-rois
        actioness_score = torch.zeros(batch_size, n_clips, rois_per_image).type_as(clips)
        overlaps_scores = torch.zeros(batch_size, n_clips, rois_per_image, rois_per_image).type_as(clips)

        if self.training:
            
            f_gt_tubes = torch.zeros(batch_size, n_clips,max_n_actions,self.sample_duration*4) # gt_tubes

            # gt_limits = torch.zeros(n_actions[i].int(),2)
            # t = torch.arange(n_frames[i].int().item()).unsqueeze(0).expand(n_actions[i].int().item(),n_frames[i].int().item()).type_as(box)
            # z = box.eq(0).all(dim=2)


            # gt_limits[:,0] = torch.min(t.masked_fill_(z,1000),dim=1)[0]
            # gt_limits[:,1] = torch.max(t.masked_fill_(z,-1),dim=1)[0]+1

            ## TODO change to parallel code
            # for i in range(batch_size):
            #     for j in range(num_actions[i]):
            #         idx = boxes[i,:,j,4].nonzero().view(-1)
            #         labels[i,j] = boxes[i,idx[0],j,4]

        ## Init connect thresh
        self.calc.thresh = self.connection_thresh

        init_time = time.time()
        for step, dt in enumerate(data_loader):

            # if step == 1:
            #     break
            # print('\tstep :',step)

            frame_indices, im_info, start_fr = dt

            im_info = im_info.unsqueeze(0).expand(batch_size, im_info.size(0),3).contiguous().view(-1,3)
            clips_ = clips[:,frame_indices].cuda()
            clips_ = clips_.contiguous().view(-1,self.sample_duration,3,self.sample_size,self.sample_size)

            if self.training:
                boxes_ = boxes[:,frame_indices].cuda()

                boxes_ = boxes_.contiguous().view(-1,self.sample_duration,max_n_actions,6)
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
            # print('pooled_feat.shape :',pooled_feat.shape)
            pooled_feat = pooled_feat.view(batch_size,-1,rois_per_image,self.p_feat_size,self.sample_duration,\
                                           self.POOLING_SIZE, self.POOLING_SIZE)

            # # regression
            n_tubes = tubes.size(0)
            if not self.training:

                tubes = tubes.view(-1, self.sample_duration*4+2)
                tubes[:,1:-1] = tube_transform_inv(tubes[:,1:-1],\
                                               sgl_rois_bbox_pred.view(-1,self.sample_duration*4),(1.0,1.0,1.0,1.0))
                tubes = tubes.view(n_tubes,-1, self.sample_duration*4+2)
                tubes[:,:,1:-1] = clip_boxes(tubes[:,:,1:-1], im_info, tubes.size(0))

            tubes = tubes.view(batch_size,-1,rois_per_image, self.sample_duration*4+2)

            idx_s = step * data_loader_batch_size 
            idx_e = min(step * data_loader_batch_size + tubes.size(1), n_clips)


            # # TODO uncomment
            features[:,idx_s:idx_e] = pooled_feat.max(4)[0]
            p_tubes[:,idx_s:idx_e,] = tubes[...,1:-1].contiguous()
            actioness_score[:,idx_s:idx_e] = tubes[...,-1].contiguous()

            # TODO
            if self.training:


                box = boxes_.permute(0,2,1,3).contiguous()[...,:-2]
                box = box.contiguous().view(batch_size,-1,box.size(1),self.sample_duration*4)

                f_gt_tubes[:,idx_s:idx_e] = box

        # # connection algo
        final_combinations = torch.zeros(batch_size, self.post_nms_tubes,n_clips,2).type_as(clips)
        final_comb_scores  = torch.zeros(batch_size, self.post_nms_tubes).type_as(clips)
        final_tubes = torch.zeros(batch_size, self.post_nms_tubes, max_n_frames, 4).type_as(clips)
        keep_indices_batch = torch.zeros(batch_size).type_as(clips).long()

        for b in range(batch_size):
            ## Init connect thresh
            thresh = self.connection_thresh

            clips_per_video = len(range(1,num_frames[b], self.step))
            
            for i in range(clips_per_video):

                if i == 0:

                    # Init tensors for connecting
                    offset = torch.arange(0,rois_per_image).int().cuda()
                    ones_t = torch.ones(rois_per_image).int().cuda()
                    zeros_t = torch.zeros(rois_per_image,n_clips,2).int().cuda()-1

                    pos = torch.zeros(rois_per_image,n_clips,2).int().cuda() -1 # initial pos
                    pos[:,0,0] = 0
                    pos[:,0,1] = offset.contiguous()                                # contains the current tubes to be connected
                    pos_indices = torch.zeros(rois_per_image).int().cuda()          # contains the pos of the last element of the previous tensor
                    actioness_scr = actioness_score[b,0].float().cuda()               # actioness sum of active tubes
                    overlaps_scr = torch.zeros(rois_per_image).float().cuda()       # overlaps  sum of active tubes
                    final_scores = torch.Tensor().float().cuda()                    # final scores
                    final_poss   = torch.Tensor().int().cuda()                      # final tubes

                    continue

                # calculate overlaps
                # overlaps_ = tube_overlaps(p_tubes[i-1,:,6*4:],p_tubes[i,:,:2*4]).type_as(p_tubes)  #
                overlaps_ = tube_overlaps(p_tubes[b,i-1,:,-1*4:],p_tubes[b,i,:,:1*4]).type_as(p_tubes)  #

                pos, pos_indices, \
                f_scores, actioness_scr, \
                overlaps_scr,thresh = self.calc(torch.Tensor([n_clips]),torch.Tensor([rois_per_image]),torch.Tensor([pos.size(0)]),
                                                pos, pos_indices, actioness_scr, overlaps_scr,
                                                overlaps_, actioness_score[b,i].contiguous(), torch.Tensor([i]), torch.Tensor([thresh]).type_as(clips))

                if pos.size(0) > self.update_thresh:

                    final_scores, final_poss, pos , pos_indices, \
                    actioness_scr, overlaps_scr,  f_scores, thresh = self.calc.update_scores(final_scores,final_poss, f_scores, pos, pos_indices, actioness_scr, overlaps_scr, thresh)


                if final_scores.size(0) > self.final_scores_max_num:

                    final_scores, final_poss  = self.calc.update_final_scores(final_scores,final_poss)

                if f_scores.dim() == 0:
                    f_scores = f_scores.unsqueeze(0)
                    pos = pos.unsqueeze(0)
                    pos_indices = pos_indices.unsqueeze(0)
                    actioness_scr = actioness_scr.unsqueeze(0)
                    overlaps_scr = overlaps_scr.unsqueeze(0)

                if final_scores.dim() == 0:
                    final_scores = final_scores.unsqueeze(0)
                    final_poss = final_poss.unsqueeze(0)

                try:
                    final_scores = torch.cat((final_scores, f_scores))
                except:
                    print('final_scores :',final_scores)
                    print('final_scores.shape :',final_scores.shape)
                    print('final_scores.dim() :',final_scores.dim())
                    print('f_scores :',f_scores)
                    print('f_scores.shape :',f_scores.shape)
                    print('f_scores.dim() :',f_scores.dim())
                    exit(-1)

                try:
                    final_poss = torch.cat((final_poss, pos))                    
                except:
                    print('final_poss :',final_poss)
                    print('final_poss.shape :',final_poss.shape)
                    print('final_poss.dim() :',final_poss.dim())
                    print('pos :',pos)
                    print('pos.shape :',pos.shape)
                    print('pos.dim() :',pos.dim())
                    exit(-1)

                # # add new tubes
                pos= torch.cat((pos,zeros_t))
                pos[-rois_per_image:,0,0] = ones_t * i
                pos[-rois_per_image:,0,1] = offset

                pos_indices   = torch.cat((pos_indices,torch.zeros((rois_per_image)).type_as(pos_indices)))
                actioness_scr = torch.cat((actioness_scr, actioness_score[b,i]))
                overlaps_scr  = torch.cat((overlaps_scr, torch.zeros((rois_per_image)).type_as(overlaps_scr)))

            ## add only last layers
            indices = actioness_score[b,-1].ge(thresh).nonzero().view(-1)
            
            if indices.nelement() > 0:
                zeros_t[:,0,0] = idx_e-1
                zeros_t[:,0,1] = offset
                final_poss = torch.cat([final_poss, zeros_t[indices]])
                final_scores = torch.cat([final_scores, actioness_score[b,-1,indices]])

            if pos.size(0) > self.update_thresh:

                final_scores, final_poss, pos , pos_indices, \
                actioness_scr, overlaps_scr,  f_scores,thresh = self.calc.update_scores(final_scores,final_poss, f_scores, pos,\
                                                                                        pos_indices, actioness_scr, overlaps_scr, thresh)

            before_nms = time.time()
            # ######################################################################################################
            # # pick best scoring tubes
            # print('final_scores.size(0) :',final_scores.size(0))

            # k_epil = min(self.post_nms_tubes,final_scores.size(0))
            # _, indices = torch.topk(final_scores,k_epil)

            # print('k_epil :',k_epil)

            # final_combinations[b,:k_epil] = final_poss[indices]
            # final_comb_scores[b,:k_epil]  = final_scores[indices]
            # final_tubes[b,:,:num_frames[b]]= self.translator(torch.Tensor([n_clips]), torch.Tensor([rois_per_image]),torch.Tensor([num_frames[b]]),\
            #                                                  p_tubes[b].contiguous(), final_combinations[b].int().contiguous())

            # ######################################################################################################
            ######################################################################################################

            # pick best scoring tubes

            if final_scores.size(0) > self.pre_nms_tubes:
                
                _, indices = torch.topk(final_scores,self.pre_nms_tubes)


                final_poss = final_poss[indices]
                final_scores = final_scores[indices]

            ### translate tubes
            tubes_trsl = self.translator(torch.Tensor([n_clips]), torch.Tensor([rois_per_image]),torch.Tensor([num_frames[b]]),\
                                         p_tubes[b].contiguous(), final_poss.int().contiguous())

            ### NMS

            fin_tubes =tubes_trsl.view(-1,num_frames*4).contiguous()

            # keep_idx_i = py_cpu_nms_tubes(torch.cat([fin_tubes,final_scores.unsqueeze(1)], dim=1), 0.7)
            # keep_idx_i = py_cpu_nms_tubes(torch.cat([fin_tubes,final_scores.unsqueeze(1)], dim=1), 0.8)
            keep_idx_i = py_cpu_nms_tubes(torch.cat([fin_tubes,final_scores.unsqueeze(1)], dim=1), 0.9)
            keep_idx_i = torch.Tensor(keep_idx_i).cuda().long().view(-1)

            # keep_idx_i = py_cpu_softnms(fin_tubes,final_scores, Nt=0.7)
            # keep_idx_i = py_cpu_softnms(tubes_trsl,final_scores, Nt=0.7, method=1)
            # print('keep_idx_i :',keep_idx_i)
            print('keep_idx_i.type() :',keep_idx_i.type())
            print('keep_idx_i.shape :',keep_idx_i.shape)



            if self.post_nms_tubes > 0:

                keep_idx_i = keep_idx_i[:self.post_nms_tubes]


            keep_indices_batch[b] = keep_idx_i.numel()
            final_combinations[b,:keep_idx_i.numel()] = final_poss[keep_idx_i]
            final_comb_scores[b,:keep_idx_i.numel()]  = final_scores[keep_idx_i]
            final_tubes[b,:keep_idx_i.numel(),:num_frames[b]] = tubes_trsl[keep_idx_i]
            
            after_nms = time.time()
            print('nms+translate :',after_nms-before_nms)

        ######################################################################################################


        # trans_time = time.time()
        # print('trans_time :',trans_time-conn_time)

        ### PRiNT COMBINATIONS
        # for i in range(self.n_ret_tubes):
        #     for j in range(final_combinations.size(1)):
        #         if final_combinations[i,j,0]==-1:
        #             break
        #         print('[',final_combinations[i,j,0].item(),',',final_combinations[i,j,1].item(),'|',end=' ')
        #     print()

        ###################################################
        #          Choose gth Tubes for RCNN\TCN          #
        ###################################################
        tubes_labels = None

        if self.training:

            tubes_per_video = 36
            tubes_labels = torch.zeros(batch_size, tubes_per_video).type_as(final_tubes)
            picked_tubes = torch.zeros(batch_size, tubes_per_video, max_n_frames, 4).type_as(final_tubes)
            picked_pos   = torch.zeros(batch_size, tubes_per_video, n_clips,2).type_as(final_tubes) -1

            fg_tubes_per_video = 12 

            boxes_ = boxes.permute(0,2,1,3).contiguous()
            boxes_ = boxes_[:,:,:,:4].contiguous().view(batch_size, max_n_actions,-1)
            
            f_tubes_train = []

            for b in range(batch_size):

                final_combinations_ = final_combinations[b]
                final_tubes_ = final_tubes[b]

                # print('overlaps.shape :',overlaps.shape)
                # max_overlaps,_ = torch.max(overlaps,1)
                # max_overlaps = max_overlaps.clamp_(min=0)

                # gt_max_overlaps,_ = torch.max(overlaps, 0)

                gt_tube_tensor = torch.zeros(num_actions[b],n_clips,2) -1                    
                gt_tubes_list = [[] for i in range(num_actions[b])]
                p_indx = torch.zeros(num_actions[b]).int()

                for i in range(n_clips):

                    overlaps = tube_overlaps(p_tubes[b,i], f_gt_tubes[b,i].type_as(p_tubes))
                    max_overlaps, argmax_overlaps = torch.max(overlaps, 0)

                    for j in range(num_actions[b]):
                        if max_overlaps[j] > 0.9: 
                            gt_tubes_list[j].append([i,j])
                            gt_tube_tensor[j,p_indx[j],0] = i
                            gt_tube_tensor[j,p_indx[j],1] = j
                            p_indx[j] += 1

                # f_tubes = gt_tubes_list + f_tubes
                final_tubes_ = torch.cat([ boxes_[b,:num_actions[b]].view(num_actions[b],-1,4).type_as(final_tubes),final_tubes_])
                final_combinations_ = torch.cat([gt_tube_tensor.type_as(final_combinations_),final_combinations_])

                # evaluate again overlaps
                overlaps = tube_overlaps(final_tubes_.view(-1,max_n_frames*4), boxes_[b,:num_actions[b]].view(-1,max_n_frames*4).type_as(final_tubes))
                max_overlaps,_ = torch.max(overlaps,1)
                max_overlaps = max_overlaps.clamp_(min=0)
                gt_max_overlaps,_ = torch.max(overlaps, 0)

                ## TODO change numbers
                fg_tubes_indices = max_overlaps.ge(0.7).nonzero().view(-1)
                fg_num_tubes = fg_tubes_indices.numel()

                bg_tubes_indices = torch.nonzero((max_overlaps >= 0.1 ) &
                                                 (max_overlaps <  0.3 )).view(-1)
                bg_num_tubes = bg_tubes_indices.numel()

                if fg_num_tubes > 0 and bg_num_tubes > 0:

                    fg_tubes_per_this_video = min(fg_tubes_per_video, fg_num_tubes)
                    rand_num = torch.from_numpy(np.random.permutation(fg_num_tubes)).type_as(final_tubes).long()
                    fg_tubes_indices = fg_tubes_indices[rand_num[:fg_tubes_per_this_video]]

                    # sampling bg
                    bg_tubes_per_this_video = tubes_per_video - fg_tubes_per_this_video

                    rand_num = np.floor(np.random.rand(bg_tubes_per_this_video) * bg_num_tubes)
                    rand_num = torch.from_numpy(rand_num).type_as(final_tubes).long()
                    bg_tubes_indices = bg_tubes_indices[rand_num]

                elif fg_num_tubes > 0 and bg_num_tubes == 0:

                    rand_num = np.floor(np.random.rand(tubes_per_video) * fg_num_tubes)
                    rand_num =torch.from_numpy(rand_num).type_as(final_tubes).long()

                    fg_tubes_indices = fg_tubes_indices[rand_num]
                    fg_tubes_per_this_video = tubes_per_video
                    bg_tubes_per_this_video = 0

                elif fg_num_tubes == 0 and bg_num_tubes > 0:

                    print("NO FG tubes found... problemm...")
                    print('max_overlaps :',max_overlaps)
                    print('final_combinations :',final_combinations)
                    rand_num = np.floor(np.random.rand(tubes_per_video) * bg_num_tubes)
                    rand_num = torch.from_numpy(rand_num).type_as(final_tubes).long()

                    bg_tubes_indices = bg_tubes_indices[rand_num]
                    bg_tubes_per_this_video = tubes_per_video
                    fg_tubes_per_this_video = 0

                else:
                    print('max_overlaps  :',max_overlaps)
                    print('final_tubes :',final_tubes)
                    print('pos :',pos[:20])
                    print('scores :',conn_scores)
                    exit(-1)

                keep_inds = torch.cat([fg_tubes_indices, bg_tubes_indices], 0)

                tubes_labels[b,:fg_tubes_per_this_video] = target[b]

                picked_pos[b] = final_combinations_[keep_inds]
                picked_tubes[b] = final_tubes_[keep_inds]

                final_combinations = picked_pos
                final_tubes = picked_tubes

        ##############################################

        # ##########################################
        # #           Time for Linear Loss         #
        # ##########################################


        prob_out = torch.zeros(batch_size, self.post_nms_tubes,self.n_classes).type_as(clips)


        for b in range(batch_size):

            n_filters = features.size(-3)
            x_axis = features.size(-2)
            y_axis = features.size(-1)

            f_features = features.new(keep_indices_batch[b], self.pooling_time, n_filters, \
                                      x_axis, y_axis).zero_().type_as(features)

            for i in range(keep_indices_batch[b]):
                t = final_combinations[b,i].ne(-1).all(dim=1).nonzero().view(-1)
                if t.numel() < 1:
                    continue
                indices = final_combinations[b,i,t].long()
                f_features[i]  = self.temporal_pool(features[b,indices[:,0],indices[:,1]])

            f_features = f_features.permute(0,2,1,3,4)
            f_features = self.act_rnn_top(f_features).mean(4).mean(3)
            prob_out[b,:keep_indices_batch[b]] = self.cls(f_features.view(f_features.size(0),-1))

            
        if self.training:

            return None, None,  cls_loss, 

        else:
            prob_out = F.softmax(prob_out,2)
            # print('final_tubes.shape :',final_tubes.shape)
            # print('prob_out.shape :',prob_out.shape)
            
            # init padding tubes because of multi-GPU system
            # if final_tubes.size(0) > conf.MODEL_POST_NMS_TUBES:
            #     _, indices = torch.sort(conn_scores)
            #     final_tubes = final_tubes[:,indices[:conf.ALL_SCORES_THRESH]].contiguous()
            #     prob_out = prob_out[:,indices[:conf.ALL_SCORES_THRESH]].contiguous()

            # ret_tubes = torch.zeros(batch_size,conf.MODEL_POST_NMS_TUBES, ret_n_frames,4).type_as(final_tubes).float() -1
            # ret_prob_out = torch.zeros(batch_size,conf.MODEL_POST_NMS_TUBES,self.n_classes).type_as(final_tubes).float() - 1

            # ret_tubes[:,:final_tubes.size(1),:num_frames] = final_tubes
            # print('ret_prob_out.shape :',ret_prob_out.shape)
            # ret_prob_out[:,:prob_out.size(1)] = prob_out

            # return ret_tubes, ret_prob_out, torch.Tensor([final_tubes.size(0)]).cuda()
            return final_tubes, prob_out, keep_indices_batch

    def temporal_pool(self, features):

        n_filters = features.size(1)
        x_axis = features.size(2)
        y_axis = features.size(3)
        
        indexes = torch.linspace(0, features.size(0), self.pooling_time+1).int()
        ret_feats = torch.zeros(self.pooling_time, n_filters,\
                                x_axis, y_axis)

        if features.size(0) < self.pooling_time:

            ret_feats[:features.size(0)] = features
            return ret_feats
    
        for i in range(self.pooling_time):
            
            t = features[indexes[i]:indexes[i+1]].permute(1,0,2,3)
            t = t.view(t.size(0),t.size(1),t.size(2),-1)
            t = F.max_pool3d(t,kernel_size=(indexes[i+1]-indexes[i],1,1)).squeeze()
            ret_feats[i] = t.view(t.size(0), features.size(2),features.size(3))

        return ret_feats

    def deactivate_action_net_grad(self):

        for p in self.act_net.parameters() : p.requires_grad=False

    def load_part_model(self, action_model_path=None, rnn_path=None):

        
        # load action net
        if action_model_path != None:
            
            act_data = torch.load(action_model_path)
            # act_data = torch.load('./action_net_model.pwf')


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

            act_rnn = RestNet(self.n_classes, self.sample_duration, pooling_time=20)
            act_rnn.create_architecture()
            act_rnn_data = torch.load(rnn_path)
            print('act_rnn_data :',act_rnn_data.keys())
            act_rnn.load_state_dict(act_rnn_data)
            self.act_rnn_top = act_rnn.top_net
            self.cls = act_rnn.cls

        else:

            act_rnn = RestNet(self.n_classes, self.sample_duration, pooling_time=20)
            act_rnn.create_architecture()
            self.act_rnn_top = act_rnn.top_net
            self.cls = act_rnn.cls

