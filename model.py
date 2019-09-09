import os
import numpy as np
import glob
from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from conf import conf
# from action_net import ACT_net
from action_prog import Act_Prog
from act_rnn import Act_RNN
from new_calc.calc import Calculator

from create_tubes_from_boxes import create_video_tube, create_tube_from_tubes, create_tube_with_frames
from connect_tubes import connect_tubes, get_gt_tubes_feats_label, get_tubes_feats_label
from resize_rpn import resize_boxes, resize_tube

from ucf_dataset import single_video

from bbox_transform import bbox_overlaps_connect
from collections import OrderedDict
from box_functions import bbox_transform, tube_transform_inv, clip_boxes, tube_overlaps


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
        self.max_con_tubes = 2
        # self.act_net = ACT_net(actions,sample_duration)

        ## general options
        self.sample_duration = sample_duration
        self.sample_size = sample_size

        # self.step = int(self.sample_duration/2)
        # self.step = int(12)
        self.step = sample_duration-1
        self.p_feat_size = 64 # 128 # 256 # 512
        
        self.POOLING_SIZE = 7
        self.MAX_NUM_TUBES = 50
        # For connection 
        self.max_num_tubes = conf.MAX_NUMBER_TUBES
        self.connection_thresh = conf.CONNECTION_THRESH
        self.update_thresh = conf.UPDATE_THRESH
        self.calc = Calculator(self.max_num_tubes, self.update_thresh, self.connection_thresh)
        


    def forward(self,n_devs, dataset_folder, vid_names, clips, vid_id, boxes, mode, cls2idx, num_actions, num_frames, h_, w_):
        '''
        TODO describe procedure
        '''

        # print('boxes.shape :',boxes.shape)

        ## define a dataloader for the whole video
        # print('----------Inside----------')
        # print('num_frames :',num_frames)
        # print('clips.shape :',clips.shape)

        clips = clips.squeeze(0)
        ret_n_frames = clips.size(0)
        clips = clips[:num_frames]
        
        # print('num_frames :',num_frames)
        # print('clips.shape :',clips.shape)
        # exit(-1)
        if self.training:

            boxes = boxes.squeeze(0).permute(1,0,2).cpu()
            boxes = boxes[:num_frames,:num_actions].clamp_(min=0)

            act_s = torch.zeros(num_actions)
            act_e = torch.zeros(num_actions)
            for i in range(num_actions):
                fr =  boxes[:,i,4].nonzero()
                act_s[i] = fr[0]
                act_e[i] = fr[-1]


        batch_size = 4 # 
        # batch_size = 2 # 
        # batch_size = 16 # 

        num_images = 1
        rois_per_image = int(conf.TRAIN.BATCH_SIZE / num_images) if self.training else 150

        data = single_video(dataset_folder,h_,w_, vid_names, vid_id, frames_dur= self.sample_duration, sample_size =self.sample_size,
                            classes_idx=cls2idx, n_frames=num_frames, step=self.step)

        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, pin_memory=False,# num_workers=num_workers, pin_memory=True,
                                                  # shuffle=False, num_workers=8)
                                                  shuffle=False)

        n_clips = data.__len__()

        features = torch.zeros(n_clips, rois_per_image, self.p_feat_size, self.sample_duration, self.POOLING_SIZE, self.POOLING_SIZE).type_as(clips)
        p_tubes = torch.zeros(n_clips, rois_per_image,  self.sample_duration*4).type_as(clips) # all the proposed tube-rois
        tube_rate = torch.zeros(n_clips, rois_per_image, self.n_classes).type_as(clips) # all the proposed tube-rois
        tube_prog = torch.zeros(n_clips, rois_per_image, self.n_classes).type_as(clips) # all the proposed tube-rois

        actioness_score = torch.zeros(n_clips, rois_per_image, self.n_classes).type_as(clips) # saving after each loop
        overlaps_scores = torch.zeros(n_clips, rois_per_image, rois_per_image).type_as(clips) 

        # final_tubes = torch.zeros(self.n_classes, self.MAX_NUM_TUBES, n_clips, 2).type_as(clips) - 1 
        # final_scores = torch.zeros(self.n_classes, self.MAX_NUM_TUBES).type_as(clips)
        # final_temporal_scr = torch.zeros(self.n_classes, self.MAX_NUM_TUBES).type_as(clips)

        final_tubes_pos = torch.zeros(self.n_classes, rois_per_image, n_clips, 2).type_as(clips).int() - 1
        final_tube_indices = torch.zeros(self.n_classes, rois_per_image ).type_as(clips).int() - 1
        final_scores = torch.zeros(self.n_classes, rois_per_image).type_as(clips)
        final_temporal_scr = torch.zeros(self.n_classes, rois_per_image).type_as(clips)
        final_temporal_rt = torch.zeros(self.n_classes, rois_per_image).type_as(clips)
        N_up = torch.zeros(self.n_classes, rois_per_image).type_as(clips)
        N_down = torch.zeros(self.n_classes, rois_per_image).type_as(clips)

        f_tubes = []

        
        if self.training:
            
            f_gt_tubes = torch.zeros(n_clips,num_actions,self.sample_duration*4) # gt_tubes
            tubes_labels = torch.zeros(n_clips,rois_per_image)  # tubes rois
            loops = int(np.ceil(n_clips / batch_size))
            labels = torch.zeros(num_actions)

            for i in range(num_actions):
                idx = boxes[:,i,4].nonzero().view(-1)
                # if boxes[idx[0],i,4] < 1:
                #     print('boxes[i,] :',boxes[:,i])
                #     print('boxes[:,i,4].gt(0) :',boxes[:,i,4])
                #     print('boxes[:,i,4].gt(0) :',boxes[:,i,4].gt(0))
                #     print('boxes[:,i,4].gt(0) :',boxes[:,i,4].gt(0).nonzero().view(-1))
                #     print('idx[0] :',idx[0])
                    
                labels[i] = boxes[idx[0],i,4]
        ## Init connect thresh
        self.calc.thresh = self.connection_thresh
        # print('n_clips :',n_clips)
        for step, dt in enumerate(data_loader):

            # if step == 1:
            #     break
            # print('\tstep :',step)

            frame_indices, im_info, start_fr = dt
            clips_ = clips[frame_indices].cuda()

            if self.training:
                boxes_ = boxes[frame_indices].cuda()
                box_ = boxes_.permute(0,2,1,3).float().contiguous()[:,:,:,:-1]
                rate = torch.zeros(batch_size, num_actions)

                fr = frame_indices[:,-1].contiguous().view(frame_indices.size(0),1).expand(frame_indices.size(0),num_actions).float()
                fr = torch.where(fr>act_s, fr, torch.zeros(fr.shape))
                rate = (fr/ act_e.float()).clamp_(max=1)
                
            else:
                box_ = None
                rate = None

            im_info = im_info.cuda()
            start_fr = start_fr.cuda()

            with torch.no_grad():
                tubes, pooled_feat, \
                rpn_loss_cls,  rpn_loss_bbox, \
                rois_rate,_, \
                rois_prog, _, \
                rois_label, \
                sgl_rois_bbox_pred, sgl_rois_bbox_loss = self.act_net(clips_.permute(0,2,1,3,4),
                                                                      im_info,
                                                                      None,
                                                                      box_,
                                                                      start_fr, rate)
                # print('rois_rate.shape :',rois_rate.shape)

            pooled_feat = pooled_feat.view(-1,rois_per_image,self.p_feat_size,self.sample_duration, \
                                           self.POOLING_SIZE,self.POOLING_SIZE).contiguous()
            rois_rate = rois_rate.view(-1, rois_per_image, self.n_classes).contiguous()
            rois_prog = rois_prog.view(-1, rois_per_image, self.n_classes).contiguous()

            n_tubes = len(tubes)
            if not self.training:
                tubes = tubes.view(-1, self.sample_duration*4+2)
                tubes[:,1:-1] = tube_transform_inv(tubes[:,1:-1],\
                                               sgl_rois_bbox_pred.view(-1,self.sample_duration*4),(1.0,1.0,1.0,1.0))
                tubes = tubes.view(n_tubes,rois_per_image, self.sample_duration*4+2)
                tubes[:,:,1:-1] = clip_boxes(tubes[:,:,1:-1], im_info, tubes.size(0))

            indexes_ = (torch.arange(0, tubes.size(0))*int(self.sample_duration/2) + start_fr[0].cpu()).unsqueeze(1)
            indexes_ = indexes_.expand(tubes.size(0),tubes.size(1)).type_as(tubes)

            idx_s = step * batch_size 
            idx_e = min(step * batch_size + batch_size, n_clips)

            features[idx_s:idx_e] = pooled_feat
            p_tubes[idx_s:idx_e,] = tubes[:,:,1:-1]
            tube_rate[idx_s:idx_e,] = rois_rate.clamp(min=0, max=1)
            tube_prog[idx_s:idx_e,] = rois_prog.clamp(min=0, max=1)

            actioness_score[idx_s:idx_e] = rois_prog *tubes[:,:,-1].contiguous().unsqueeze(-1)
            # actioness_score[idx_s:idx_e] = tubes[:,:,-1].contiguous().unsqueeze(-1)

            if self.training:

                box = boxes_.permute(0,2,1,3).contiguous()[:,:,:,:-2]
                box = box.contiguous().view(box.size(0),box.size(1),-1)

                f_gt_tubes[idx_s:idx_e] = box

            # connection algo
            for i in range(idx_s, idx_e):

                if i == 0:

                    for j in range(1,self.n_classes):
                        # Init tensors for connecting

                        offset = torch.arange(0,rois_per_image).int().cuda()
                        ones_t = torch.ones(rois_per_image).int().cuda()
                        zeros_t = torch.zeros(rois_per_image,n_clips,2).int().cuda()-1

                        final_tubes_pos[j,:,0,0] = 0
                        final_tubes_pos[j,:,0,1] = offset                                # contains the current tubes to be connected
                        final_tube_indices[j,:] = 0

                        final_scores[j] = actioness_score[i,:,j]
                        final_temporal_rt[j] = tube_rate[i,:,j]

                    continue

                # print('p_tubes :',p_tubes[i-1].cpu().numpy())
                # print('p_tubes :',p_tubes[i].cpu().numpy())
                # overlaps_ = tube_overlaps(p_tubes[i-1,:,int(self.sample_duration*4/2):],p_tubes[i,:,:int(self.sample_duration*4/2)]).type_as(p_tubes)
                overlaps_ = tube_overlaps(p_tubes[i-1,:,-1*4:],p_tubes[i,:,:1*4]).type_as(p_tubes)

                pos = torch.zeros(rois_per_image,n_clips,2).int().cuda()-1
                pos[:,0,0] = torch.ones(rois_per_image).int().cuda() * i
                pos[:,0,1] = offset
                pos_indices = torch.zeros(rois_per_image).type_as(final_tube_indices)

                for j in range(1,self.n_classes):
                # for j in range(1,2):

                    new_tubes, new_tube_indices, \
                    new_N_up, new_N_down, new_actioness, \
                    new_temporal_scr, new_temporal_rt = self.calc(torch.Tensor([n_clips]),torch.Tensor([rois_per_image]),\
                                                                  torch.Tensor([final_tubes_pos[j].size(0)]),
                                                                  final_tubes_pos[j].contiguous(), final_tube_indices[j].contiguous(), \
                                                                  final_scores[j].contiguous(), final_temporal_scr[j].contiguous(),\
                                                                  final_temporal_rt[j].contiguous(),
                                                                  overlaps_, \
                                                                  actioness_score[i,:,j].contiguous(), \
                                                                  tube_rate[i,:,j].contiguous(), N_up[j].contiguous(), \
                                                                  N_down[j].contiguous(),  torch.Tensor([i]))


                    # print('new_tubes.shape :',new_tubes.shape)
                    # print('new_tubes_indices.shape :',new_tube_indices.shape)
                    # print('final_tubes[i].shape :',final_tubes_pos[j].shape)
                    # print('new_tubes.shape :', new_tubes.shape)
                    # print('new_tubes.dim() :',new_tubes.dim())

                    if new_tubes.dim() == 2:

                        new_tubes = new_tubes.unsqueeze(0)
                        new_tube_indices = new_tube_indices.unsqueeze(0)
                        new_N_up = new_N_up.unsqueeze(0)
                        new_N_down = new_N_down.unsqueeze(0)
                        new_actioness = new_actioness.unsqueeze(0)
                        new_temporal_scr = new_temporal_scr.unsqueeze(0)
                        new_temporal_rt = new_temporal_rt.unsqueeze(0)



                    mix_tubes = torch.cat([final_tubes_pos[j], new_tubes, pos], dim=0)
                    mix_tubes_indices = torch.cat([final_tube_indices[j], new_tube_indices, pos_indices], dim=0)
                    mix_N_up = torch.cat([N_up[j], new_N_up, torch.zeros(rois_per_image).type_as(N_up)], dim=0)
                    mix_N_down = torch.cat([N_down[j], new_N_down,torch.zeros(rois_per_image).type_as(N_down)], dim=0)
                    mix_actioness = torch.cat([final_scores[j], new_actioness, actioness_score[i,:,j]], dim=0)
                    mix_temporal_scr = torch.cat([final_temporal_scr[j], new_temporal_scr,\
                                                  torch.zeros(rois_per_image).type_as(final_temporal_scr)], dim=0)
                    mix_temporal_rt = torch.cat([final_temporal_rt[j], new_temporal_rt,tube_rate[i,:,j]], dim=0)

                    # print('mix_tubes_indices.shape :',mix_tubes_indices.shape)
                    # print('mix_N_up.shape :',mix_N_up.shape)
                    # print('mix_N_down.shape :',mix_N_down.shape)
                    # print('mix_actioness.shape :',mix_actioness.shape)
                    # print('mix_temporal_scr.shape :',mix_temporal_scr.shape)
                    # print('mix_temporal_rt.shape :',mix_temporal_rt.shape)
                    # avg_actioness = mix_actioness / (mix_tubes_indices.float()+1)
                    # print('avg_actioness :',avg_actioness)
                    # _, best_tubes_indices = torch.topk(avg_actioness, rois_per_image)
                    scrs, best_tubes_indices = torch.topk(mix_actioness, rois_per_image)
                    final_tubes_pos[j] = mix_tubes[best_tubes_indices]

                    final_tube_indices[j] = mix_tubes_indices[best_tubes_indices]
                    N_up[j] = mix_N_up[best_tubes_indices]
                    N_down[j] = mix_N_down[best_tubes_indices]
                    final_scores[j] = mix_actioness[best_tubes_indices]
                    final_temporal_scr[j] = mix_temporal_scr[best_tubes_indices]
                    final_temporal_rt[j] = mix_temporal_rt[best_tubes_indices]


                # if f_scores.dim() == 0:
                #     f_scores = f_scores.unsqueeze(0)
                #     pos = pos.unsqueeze(0)
                #     pos_indices = pos_indices.unsqueeze(0)
                #     actioness_scr = actioness_scr.unsqueeze(0)
                #     overlaps_scr = overlaps_scr.unsqueeze(0)
                # if final_scores.dim() == 0:
                #     final_scores = final_scores.unsqueeze(0)
                #     final_poss = final_poss.unsqueeze(0)

                # try:
                #     final_scores = torch.cat((final_scores, f_scores))
                # except:
                #     print('final_scores :',final_scores)
                #     print('final_scores.shape :',final_scores.shape)
                #     print('final_scores.dim() :',final_scores.dim())
                #     print('f_scores :',f_scores)
                #     print('f_scores.shape :',f_scores.shape)
                #     print('f_scores.dim() :',f_scores.dim())
                #     exit(-1)
                # try:
                #     final_poss = torch.cat((final_poss, pos))                    
                # except:
                #     print('final_poss :',final_poss)
                #     print('final_poss.shape :',final_poss.shape)
                #     print('final_poss.dim() :',final_poss.dim())
                #     print('pos :',pos)
                #     print('pos.shape :',pos.shape)
                #     print('pos.dim() :',pos.dim())
                #     exit(-1)


                # # add new tubes
                # pos= torch.cat((pos,zeros_t))
                # pos[-rois_per_image:,0,0] = ones_t * i
                # pos[-rois_per_image:,0,1] = offset

                # pos_indices   = torch.cat((pos_indices,torch.zeros((rois_per_image)).type_as(pos_indices)))
                # actioness_scr = torch.cat((actioness_scr, actioness_score[i]))
                # overlaps_scr  = torch.cat((overlaps_scr, torch.zeros((rois_per_image)).type_as(overlaps_scr)))

        final_tubes = torch.zeros(self.n_classes,rois_per_image, num_frames, 4)

        f_tubes  = []

        # print('final_tubes_pos.shape :',final_tubes_pos.shape)
        for i in range(1,self.n_classes):

            for k in range(rois_per_image):

                tub = []

                for j in range(n_clips):

                    curr_ = final_tubes_pos[i,k,j]
                    # print('curr_ :',curr_)
                    start_fr = curr_[0]* int(self.sample_duration/2)
                    end_fr = min((curr_[0]*int(self.sample_duration/2)+self.sample_duration).type_as(num_frames), num_frames).type_as(start_fr)
                    if curr_[0] == -1:
                        break
                
                    curr_frames = p_tubes[curr_[0], curr_[1]]
                    # print('curr_frames :',curr_frames)
                    tub.append((curr_[0].item(),  curr_[1].item()))
                    # print('curr_frames.view(-1,4).contiguous()[:(end_fr-start_fr).long()], :',curr_frames.view(-1,4).contiguous()[:(end_fr-start_fr).long()])
                    # print(' final_tubes[i,start_fr:end_fr].type_as(curr_frames)) :', final_tubes[i,k,start_fr:end_fr].type_as(curr_frames))
                    ## TODO change with avg
                    final_tubes[i,k,start_fr:end_fr] =  torch.max( curr_frames.view(-1,4).contiguous()[:(end_fr-start_fr).long()],
                                                                   final_tubes[i,k,start_fr:end_fr].type_as(curr_frames))
                    # print('final_tubes[i,k,start_fr:end_fr] :',final_tubes[i,k,start_fr:end_fr])

                f_tubes.append(tub)

        # final_tubes = final_tubes.view(self.n_classes * rois_per_image, num_frames, 4).contiguous()
        # ret_tubes = torch.zeros(1,conf.UPDATE_THRESH, ret_n_frames,4).type_as(final_tubes).float() -1
        # ret_tubes[0,:final_tubes.size(0),:num_frames] = final_tubes
        print('final_tubes.shape :',final_tubes.shape)
        print('final_tubes.is_cuda :',final_tubes.is_cuda)
        
        return final_tubes.unsqueeze(0), torch.Tensor([0]).cuda(), torch.Tensor([final_tubes.size(0)]).cuda()

        # ###################################################
        # #          Choose gth Tubes for RCNN\TCN          #
        # ###################################################
        # if self.training:

        #     # # get gt tubes and feats
        #     ##  calculate overlaps

        #     boxes_ = boxes.permute(1,0,2).contiguous()
        #     boxes_ = boxes_[:,:,:4].contiguous().view(num_actions,-1)

        #     if final_tubes.nelement() == 0:

        #         print('problem final_tubes ...')
        #         print('boxes :',boxes.cpu().numpy())
        #         print('boxes_ :',boxes_)
        #         print('boxes_.shape :',boxes_.shape)
        #         print('final_tubes :',final_tubes )
        #         print('self.calc.thresh:',self.calc.thresh)
        #         print('final_scores :',final_scores.shape)
        #         print('final_pos.shape :',final_poss.shape)
        #     #     exit(-1)
        #     # if boxes_.nelement() == 0:
        #     #     print('problem boxes_')
        #     #     print('boxes_ :',boxes_)
        #     #     print('boxes_.shape :',boxes_.shape)
        #     #     print('final_tubes :',final_tubes)
        #     #     exit(-1)
        #     if final_tubes.nelement() > 0:
        #         overlaps = tube_overlaps(final_tubes.view(-1,num_frames*4), boxes_.type_as(final_tubes))
        #         max_overlaps,_ = torch.max(overlaps,1)
        #         max_overlaps = max_overlaps.clamp_(min=0)

        #         ## TODO change numbers
        #         bg_tubes_indices = max_overlaps.lt(0.3).nonzero()
        #         if bg_tubes_indices.nelement() > 0:
        #             bg_tubes_indices_picked = (torch.rand(2)*bg_tubes_indices.size(0)).long()
        #             bg_tubes_list = [f_tubes[i] for i in bg_tubes_indices[bg_tubes_indices_picked]]
        #             bg_labels = torch.zeros(len(bg_tubes_list))
        #         else:
        #             bg_tubes_list = []
        #             bg_labels = torch.Tensor([])
        #     else:
        #         bg_tubes_list = []
        #         bg_labels = torch.Tensor([])

        #     gt_tubes_list = [[] for i in range(num_actions)]

        #     # print('n_clips :',n_clips)

        #     for i in range(n_clips):
        #         # print('i :',i)
        #         # print('p_tubes.shape :',p_tubes.shape)
        #         # print('f_gt_tubes.shape :',f_gt_tubes.shape)
        #         # print('p_tubes.shape :',p_tubes[i])
        #         # print('f_gt_tubes.shape :',f_gt_tubes[i])

        #         overlaps = tube_overlaps(p_tubes[i], f_gt_tubes[i].type_as(p_tubes))
        #         # print('overlaps :',overlaps)
        #         max_overlaps, argmax_overlaps = torch.max(overlaps, 0)

        #         for j in range(num_actions):
        #             if max_overlaps[j] == 1.0: 
        #                 gt_tubes_list[j].append((i,j))
        #     gt_tubes_list = [i for i in gt_tubes_list if i != []]
        #     if len(gt_tubes_list) != num_actions:
        #         print('len(gt_tubes_list :', len(gt_tubes_list))
        #         print('num_actions :',num_actions)
        #         print('boxes.cpu().numpy() :',boxes.cpu().numpy())
                
        #     # print('gt_tubes_list :',gt_tubes_list)
        #     ## concate fb, bg tubes
        #     if gt_tubes_list == [[]]:
        #         print('overlaps :',overlaps)
        #         print('max_overlaps :',max_overlaps)
        #         print('p_tubes :',p_tubes)
        #         print('f_gt_tubes :',f_gt_tubes)
        #         exit(-1)
        #     if bg_tubes_list != []:
        #         f_tubes = gt_tubes_list + bg_tubes_list
        #         target_lbl = torch.cat([labels, bg_labels],dim=0)
        #     else:
        #         f_tubes = gt_tubes_list
        #         target_lbl = labels

        # # print('num_frames :',num_frames)
        # # print('gt_tubes_list :',gt_tubes_list, ' labels :',labels)
        # # print('f_tubes :',f_tubes, ' target_lbl :',target_lbl)    
        # ##############################################

        # if len(f_tubes) == 0:
        #     print('------------------')
        #     print('    empty tube    ')
        #     print(' vid_id :', vid_id)
        #     print('self.calc.thresh :',self.calc.thresh)
        #     return torch.Tensor([]).cuda(), torch.Tensor([]).cuda(), None
        # max_seq = reduce(lambda x, y: y if len(y) > len(x) else x, f_tubes)
        # max_length = len(max_seq)

        # ## calculate input rois
        # prob_out = torch.zeros(len(f_tubes), self.n_classes).cuda()
        # final_feats = []

        # for i in range(len(f_tubes)):

        #     seq = f_tubes[i]
        #     feats = torch.Tensor(len(seq),self.p_feat_size,self.sample_duration, self.POOLING_SIZE, self.POOLING_SIZE)

        #     for j in range(len(seq)):

        #         # feats[j] = features[seq[j][0],seq[j][1]].mean(1)
        #         feats[j] = features[seq[j][0],seq[j][1]]
        #         # tmp_tube[j] = p_tubes[seq[j]][1:7]

        #     # prob_out[i] = self.act_rnn(feats.cuda())
        #     feats = torch.mean(feats, dim=-1)
        #     feats = torch.mean(feats, dim=-1)
        #     feats = torch.mean(feats, dim=0)

        #     if mode == 'extract':
        #         final_feats.append(feats)

        #     try:
        #         prob_out[i] = self.act_rnn(feats.view(-1).cuda())
        #     except Exception as e:
        #         print('feats.shape :',feats.shape)
        #         print('seq :',seq)
        #         for i in range(len(f_tubes)):
        #             print('seq[i] :',f_tubes[i])
                    
        #         print('e :',e)
        #         exit(-1)
        #     if prob_out[i,0] != prob_out[i,0]:
        #         print(' prob_out :', prob_out ,' feats :',feats.cpu().numpy(), ' numpy(), feats.shape  :,', feats.shape ,' target_lbl :',target_lbl, \
        #               ' \ntmp_tube :',tmp_tube, )
        #         exit(-1)

        # if mode == 'extract':
        #     # now we use mean so we can have a tensor containing all features
        #     final_feats = torch.stack(final_feats).cuda()
        #     final_tubes = final_tubes.cuda()
        #     target_lbl = target_lbl.cuda()
        #     max_length = torch.Tensor([max_length]).cuda()
        #     return final_feats, target_lbl, max_length
        # # ##########################################
        # # #           Time for Linear Loss         #
        # # ##########################################

        # cls_loss = torch.Tensor([0]).cuda()

        # final_tubes = final_tubes.type_as(final_poss)
        # # # classification probability
        # if self.training:
        #     cls_loss = F.cross_entropy(prob_out.cpu(), target_lbl.long()).cuda()

        # if self.training:
        #     return None, None,  cls_loss, 
        # else:
        #     prob_out = F.softmax(prob_out)

        #     # init padding tubes because of multi-GPU system
        #     if final_tubes.size(0) > conf.UPDATE_THRESH:
        #         _, indices = torch.sort(final_scores)
        #         final_tubes = final_tubes[indices[:conf.UPDATE_THRESH]].contiguous()
        #         prob_out = prob_out[indices[:conf.UPDATE_THRESH]].contiguous()

        #     ret_tubes = torch.zeros(1,conf.UPDATE_THRESH, ret_n_frames,4).type_as(final_tubes).float() -1
        #     ret_prob_out = torch.zeros(1,conf.UPDATE_THRESH,self.n_classes).type_as(final_tubes).float() - 1
        #     ret_tubes[0,:final_tubes.size(0),:num_frames] = final_tubes
        #     ret_prob_out[0,:final_tubes.size(0)] = prob_out
            # return ret_tubes, ret_prob_out, torch.Tensor([final_tubes.size(0)]).cuda()
        
            # return final_tubes, prob_out, None
        

    def deactivate_action_net_grad(self):

        for p in self.act_net.parameters() : p.requires_grad=False
        # self.act_net.eval()
        # for key, value in dict(self.named_parameters()).items():
        #     print(key, value.requires_grad)

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

            act_net = Act_Prog(self.classes,self.sample_duration)

            act_net.create_architecture()
            act_net.load_state_dict(new_state_dict)
            self.act_net = act_net

        else:

            self.act_net = Act_Prog(self.classes,self.sample_duration)
            self.act_net.create_architecture()
            
        # load lstm
        if rnn_path != None:

            # act_rnn = Act_RNN(self.p_feat_size,int(self.p_feat_size/2),self.n_classes)
            # act_rnn_data = torch.load(rnn_path)
            # act_rnn.load(act_rnn_data)


            act_rnn = nn.Sequential(
                # nn.Linear(64*self.sample_duration, 256),
                # nn.ReLU(True),
                # nn.Dropout(0.8),
                # nn.Linear(256,self.n_classes),
                nn.Linear(64*self.sample_duration, self.n_classes),
                # # nn.ReLU(True),
                # # nn.Dropout(0.8),
                # # nn.Linear(256,self.n_classes),

            )
            act_rnn_data = torch.load(rnn_path)
            print('act_rnn_data :',act_rnn_data.keys())
            act_rnn.load_state_dict(act_rnn_data)
            self.act_rnn = act_rnn

        else:

            # self.act_rnn =Act_RNN(self.p_feat_size,int(self.p_feat_size/2),self.n_classes)
            self.act_rnn = nn.Sequential(
                # nn.Linear(64*self.sample_duration, 256),
                # nn.ReLU(True),
                # nn.Dropout(0.8),
                # nn.Linear(256,self.n_classes),
                nn.Linear(64*self.sample_duration, self.n_classes),
                # # nn.ReLU(True),
                # # nn.Dropout(0.8),
                # # nn.Linear(256,self.n_classes),

            )
            for m in self.act_rnn.modules():
                if m == nn.Linear:
                    m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
