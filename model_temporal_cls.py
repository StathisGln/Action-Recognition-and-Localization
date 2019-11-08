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
from resnet_3D import resnet34_orig
# from action_net_withoutbn import ACT_net

from act_rnn import Act_RNN
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
from mAP_temp_function import calculate_mAP

from nms_tubes.py_nms import py_cpu_nms_tubes, py_cpu_temp_nms
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
        self.restNet = RestNet(self.n_classes, self.sample_duration, self.pooling_time)        


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

                tubes, _, \
                rpn_loss_cls,  rpn_loss_bbox, \
                _,_, rois_label, \
                sgl_rois_bbox_pred, sgl_rois_bbox_loss = self.act_net(clips_.permute(0,2,1,3,4),
                                                            im_info,
                                                            None,
                                                            box_,
                                                            start_fr)
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

            keep_idx_i = py_cpu_nms_tubes(torch.cat([fin_tubes,final_scores.unsqueeze(1)], dim=1), 0.7)
            # keep_idx_i = py_cpu_nms_tubes(torch.cat([fin_tubes,final_scores.unsqueeze(1)], dim=1), 0.8)
            # keep_idx_i = py_cpu_nms_tubes(torch.cat([fin_tubes,final_scores.unsqueeze(1)], dim=1), 0.9)
            keep_idx_i = torch.Tensor(keep_idx_i).cuda().long().view(-1)

            # keep_idx_i = py_cpu_softnms(fin_tubes,final_scores, Nt=0.7)
            # keep_idx_i = py_cpu_softnms(tubes_trsl,final_scores, Nt=0.7, method=1)
            # print('keep_idx_i :',keep_idx_i)
            # print('keep_idx_i.type() :',keep_idx_i.type())
            print('keep_idx_i.shape :',keep_idx_i.shape)

            if self.post_nms_tubes > 0:

                keep_idx_i = keep_idx_i[:self.post_nms_tubes]

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

        ##############################################

        # if mode == 'noclass':

        #     ret_final_tubes = torch.zeros(batch_size,self.post_nms_tubes,ret_n_frames,4).type_as(final_tubes)
        #     ret_final_tubes[:,:,:max_n_frames] = final_tubes
        #     return ret_final_tubes, torch.Tensor([0]).cuda(), torch.Tensor([0]).cuda()



        # ##########################################
        # #           Time for Linear Loss         #
        # ##########################################

        # confidence_scores = torch.zeros(batch_size,self.post_nms_tubes, self.n_classes).type_as(final_tubes)
        confidence_scores = torch.zeros(batch_size,self.post_nms_tubes).type_as(final_tubes)
        cls_pred = torch.zeros(batch_size,self.post_nms_tubes).type_as(final_tubes)

        for b in range(batch_size):

            N = final_tubes.size(1)
            f_tubes_limits = torch.zeros(N,2).type_as(final_tubes)

            t_ = torch.arange(num_frames[b].int().item()).unsqueeze(0).expand(N,num_frames[b].int()).type_as(final_tubes)
            zero_frames = final_tubes[b].view(N,-1,4).eq(0).all(dim=2)

            f_tubes_limits[:,0] = torch.min(t_.masked_fill_(zero_frames, 10000),dim=1)[0]
            f_tubes_limits[:,1] = torch.max(t_.masked_fill_(zero_frames, -1),dim=1)[0]

            f_tubes_limits[f_tubes_limits.eq(-1)] = 0
            f_tubes_limits[f_tubes_limits.eq(10000)] = 0
            f_scores_limits = final_comb_scores[b].contiguous()

            nonzero_dur_tubes = f_tubes_limits.ne(0).any(dim=1).nonzero().view(-1)
            f_tubes_limits = f_tubes_limits[nonzero_dur_tubes]
            f_scores_limits = f_scores_limits[nonzero_dur_tubes]

            # use nms to remove all double tubes
            # keep_idx_i = py_cpu_temp_nms(torch.cat([f_tubes_limits,f_scores_limits.unsqueeze(1)], dim=1), 0.999)
            # keep_idx_i = py_cpu_temp_nms(torch.cat([f_tubes_limits,f_scores_limits.unsqueeze(1)], dim=1), 0.3)
            keep_idx_i = py_cpu_temp_nms(torch.cat([f_tubes_limits,f_scores_limits.unsqueeze(1)], dim=1), 0.2)
            keep_idx_i = torch.Tensor(keep_idx_i).cuda().long().view(-1)

            f_tubes_limits = f_tubes_limits[keep_idx_i]
            f_scores_limits = f_scores_limits[keep_idx_i]

            # find all starting indexes
            step = self.sample_duration
            n_temp_tubes = f_tubes_limits.size(0)
            add_indices = torch.arange(0,self.sample_duration)

            max_num_videos = 4
            for i in range(n_temp_tubes):

                indices_vid = torch.arange(f_tubes_limits[i,0], f_tubes_limits[i,1], step)
                indices_vid = indices_vid.unsqueeze(1).expand(indices_vid.size(0),self.sample_duration) \
                              + add_indices.type_as(indices_vid)
                indices_vid = indices_vid.clamp_(min=0, max=num_frames[b,0]-1).long()

                if indices_vid.size(0) <= max_num_videos:

                    clips_ = clips[b,indices_vid].cuda()

                    with torch.no_grad():

                        ret = self.resnet(clips_.permute(0,2,1,3,4))
                        ret = F.softmax(ret,dim=1)

                else:

                    ret = torch.zeros(indices_vid.size(0), self.n_classes).type_as(f_tubes_limits)
                    for j in range(0,indices_vid.size(0),max_num_videos):
                        
                        clips_ = clips[b,indices_vid[j:j+max_num_videos]].cuda()

                        with torch.no_grad():

                            ret[j:j+max_num_videos] = self.resnet(clips_.permute(0,2,1,3,4))
                            ret[j:j+max_num_videos] = F.softmax(ret[j:j+max_num_videos],dim=1)

                ret_score = ret.mean(dim=0)
                max_score, max_indices = torch.max(ret_score, dim=0)
                # confidence_scores[b,i] = ret_score

                confidence_scores[b,i] = max_score
                cls_pred[b,i] = max_indices

        if self.training:
            return None, None,  cls_loss, 
        else:

            ret_tubes = torch.zeros(batch_size, self.post_nms_tubes, 2).type_as(final_tubes) 
            ret_tubes[:,:n_temp_tubes] = f_tubes_limits
            
            return ret_tubes, confidence_scores, cls_pred
        

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

        # load ResNet
        resnet_shortcut = 'A'
        sample_size = 112
        print('self.sample_duration :',self.sample_duration)

        model = resnet34_orig(num_classes=self.n_classes, shortcut_type=resnet_shortcut,
                              sample_size=sample_size, sample_duration=self.sample_duration,
                              last_fc=True)
        model = nn.DataParallel(model, device_ids=None)
        self.model_path = '../ucf_101.pth'
        # self.model_path = '../resnext-101-kinetics.pth'

        print("Loading pretrained weights from %s" %(self.model_path))
        model_data = torch.load(self.model_path)
        model.load_state_dict(model_data['state_dict'])

        self.resnet = model
