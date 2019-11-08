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
from calc_score_v2.calc import Calculator

from create_tubes_from_boxes import create_video_tube, create_tube_from_tubes, create_tube_with_frames
from connect_tubes import connect_tubes, get_gt_tubes_feats_label, get_tubes_feats_label
from resize_rpn import resize_boxes, resize_tube

from ucf_dataset import single_video

from bbox_transform import bbox_overlaps_connect
from collections import OrderedDict
from box_functions import bbox_transform, tube_transform_inv, clip_boxes, tube_overlaps
from nms_3d_whole_video.nms_gpu import nms_gpu
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

        # self.act_net = ACT_net(actions,sample_duration)

        ## general options
        self.sample_duration = sample_duration
        self.sample_size = sample_size
        self.step = sample_duration

        # self.p_feat_size = 64 # 128 # 256 # 512
        self.p_feat_size  = 256 # 512

        self.POOLING_SIZE = 7
        self.n_ret_tubes = conf.CALC_THRESH

        # For connection 
        self.max_num_tubes = conf.MAX_NUMBER_TUBES
        self.connection_thresh = conf.CONNECTION_THRESH
        self.update_thresh = conf.UPDATE_THRESH
        self.calc = Calculator(self.max_num_tubes, self.update_thresh, self.connection_thresh)
        


    def forward(self,n_devs, dataset_folder, vid_names, clips, vid_id, boxes, mode, cls2idx, num_actions, num_frames, h_, w_):
        '''
        TODO describe procedure
        '''

        clips = clips.squeeze(0)
        ret_n_frames = clips.size(0)
        clips = clips[:num_frames]
        
        if self.training:
            boxes = boxes.squeeze(0).permute(1,0,2).cpu()
            boxes = boxes[:num_frames,:num_actions].clamp_(min=0)
            self.step = self.sample_duration-1

        batch_size = 16 # 
        num_images = 1

        rois_per_image = int(conf.TRAIN.BATCH_SIZE / num_images) if self.training else 150

        data = single_video(dataset_folder,h_,w_, vid_names, vid_id, frames_dur= self.sample_duration, sample_size =self.sample_size,step=self.step,
                            classes_idx=cls2idx, n_frames=num_frames)

        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, pin_memory=False,# num_workers=num_workers, pin_memory=True,
                                                  # shuffle=False, num_workers=8)
                                                  shuffle=False)

        n_clips = data.__len__()

        features = torch.zeros(n_clips, rois_per_image, self.p_feat_size, self.sample_duration,self.POOLING_SIZE, self.POOLING_SIZE).type_as(clips)
        p_tubes = torch.zeros(n_clips, rois_per_image,  self.sample_duration*4).type_as(clips) # all the proposed tube-rois
        actioness_score = torch.zeros(n_clips, rois_per_image).type_as(clips)
        overlaps_scores = torch.zeros(n_clips, rois_per_image, rois_per_image).type_as(clips)

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

                # print('i :',i)
                if i == 0:

                    # Init tensors for connecting
                    offset = torch.arange(0,rois_per_image).int().cuda()
                    ones_t = torch.ones(rois_per_image).int().cuda()
                    zeros_t = torch.zeros(rois_per_image,n_clips,2).int().cuda()-1

                    pos = torch.zeros(rois_per_image,n_clips,2).int().cuda() -1 # initial pos
                    pos[:,0,0] = 0
                    pos[:,0,1] = offset.contiguous()                                # contains the current tubes to be connected
                    pos_indices = torch.zeros(rois_per_image).int().cuda()          # contains the pos of the last element of the previous tensor
                    actioness_scr = actioness_score[0].float().cuda()               # actioness sum of active tubes
                    overlaps_scr = torch.zeros(rois_per_image).float().cuda()       # overlaps  sum of active tubes
                    final_scores = torch.Tensor().float().cuda()                    # final scores
                    final_poss   = torch.Tensor().int().cuda()                      # final tubes

                    continue
                

                # calculate overlaps
                # overlaps_ = tube_overlaps(p_tubes[i-1,:,6*4:],p_tubes[i,:,:2*4]).type_as(p_tubes)  #
         
                overlaps_ = tube_overlaps(p_tubes[i-1,:,-1*4:],p_tubes[i,:,:1*4]).type_as(p_tubes)  #
                # print('overlaps :',overlaps_)
                # connect tubes
                # print('pos.shape :',pos.shape)
                pos, pos_indices, \
                f_scores, actioness_scr, \
                overlaps_scr = self.calc(torch.Tensor([n_clips]),torch.Tensor([rois_per_image]),torch.Tensor([pos.size(0)]),
                                         pos, pos_indices, actioness_scr, overlaps_scr,
                                         overlaps_, actioness_score[i], torch.Tensor([i]))
               
                if pos.size(0) > self.update_thresh:

                    final_scores, final_poss, pos , pos_indices, \
                    actioness_scr, overlaps_scr,  f_scores = self.calc.update_scores(final_scores,final_poss, f_scores, pos, pos_indices, actioness_scr, overlaps_scr)

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
                actioness_scr = torch.cat((actioness_scr, actioness_score[i]))
                overlaps_scr  = torch.cat((overlaps_scr, torch.zeros((rois_per_image)).type_as(overlaps_scr)))

        ## add only last layers
        indices = actioness_score[-1].ge(self.calc.thresh).nonzero().view(-1)

        if indices.nelement() > 0:
            zeros_t[:,0,0] = idx_e-1
            zeros_t[:,0,1] = offset
            final_poss = torch.cat([final_poss, zeros_t[indices]])

        if pos.size(0) > self.update_thresh:
            # print('Updating thresh...', final_scores.shape, final_poss.shape, pos.shape, f_scores.shape, pos_indices.shape)
            final_scores, final_poss, pos , pos_indices, \
                actioness_scr, overlaps_scr,  f_scores = self.calc.update_scores(final_scores,final_poss, f_scores, pos, pos_indices, actioness_scr, overlaps_scr)
            # print('Updating thresh...', final_scores.shape, final_poss.shape, pos.shape, f_scores.shape, pos_indices.shape)
            
        ######################################################################################################
        # conn_time = time.time()
        # print('conection_time :',conn_time-init_time)

        # pick best scoring tubes
        _, indices = torch.topk(final_scores,min(self.n_ret_tubes,final_scores.size(0)))
        final_combinations = final_poss[indices]
        final_comb_scores  = final_scores[indices]

        ######################################################################################################
        final_tubes = torch.zeros(self.n_ret_tubes, num_frames, 4)

        f_tubes  = []

        # ### TO Uncommenct during testing
        # for i in range(self.n_ret_tubes):

        #     non_zr = final_combinations[i,:,0].ne(-1).nonzero().view(-1)
        #     theseis = final_combinations[i,non_zr].long()
        #     p_tubes_ = p_tubes[theseis[:,0], theseis[:,1]]

        #     start_fr = theseis[0,0]* int(self.step)
        #     end_fr = torch.min((theseis[-1,0]+1)*self.step, num_frames)
        #     if self.training:
        #         print('theseis:',theseis)
        #         print('p_tubes[theseis[:,0], theseis[:,1]].contiguous().shape :',p_tubes[theseis[:,0], theseis[:,1]].contiguous().\
        #               view(non_zr.size(0)*self.sample_duration,4).shape)
        #         print('p_tubes[theseis[:,0], theseis[:,1]].contiguous().shape :',p_tubes[theseis[:,0], theseis[:,1]].contiguous().shape)
        #         print('final_tubes.shape :',final_tubes.shape)

        #     final_tubes[i, start_fr:end_fr] = p_tubes[theseis[:,0], theseis[:,1]].contiguous().\
        #                                       view(non_zr.size(0)*self.sample_duration,4)

        #     f_tubes.append(theseis.cpu().tolist())

        ### To comment after training
        for i in range(final_combinations.size(0)):

            tub = []

            for j in range(n_clips):
                
                curr_ = final_combinations[i,j]
                start_fr = curr_[0]* int(self.step)
                end_fr = min((curr_[0]*int(self.step)+self.sample_duration).type_as(num_frames), num_frames).type_as(start_fr)

                if curr_[0] == -1:
                    break
                
                curr_frames = p_tubes[curr_[0], curr_[1]]
                tub.append([curr_[0].item(),  curr_[1].item()])
                ## TODO change with avg
                final_tubes[i,start_fr:end_fr] =  torch.max( curr_frames.view(-1,4).contiguous()[:(end_fr-start_fr).long()],
                                                             final_tubes[i,start_fr:end_fr].type_as(curr_frames))
            if len(tub)>0:
                f_tubes.append(tub)

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
        if self.training:

            tubes_per_video = 36
            tubes_labels = torch.zeros(tubes_per_video).type_as(final_tubes)
            picked_tubes = torch.zeros(tubes_per_video, num_frames, 4).type_as(final_tubes)

            fg_tubes_per_video = 12 

            boxes_ = boxes.permute(1,0,2).contiguous()
            boxes_ = boxes_[:,:,:4].contiguous().view(num_actions,-1)

            overlaps = tube_overlaps(final_tubes.view(-1,num_frames*4), boxes_.type_as(final_tubes))

            max_overlaps,_ = torch.max(overlaps,1)
            max_overlaps = max_overlaps.clamp_(min=0)
            gt_max_overlaps,_ = torch.max(overlaps, 0)
            
            ## If there is no tube that contains an action
            if gt_max_overlaps.ne(1.0).nonzero().numel() != 0:

                gt_tubes_list = [[] for i in range(num_actions)]
                for i in range(n_clips):

                    overlaps = tube_overlaps(p_tubes[i], f_gt_tubes[i].type_as(p_tubes))
                    max_overlaps, argmax_overlaps = torch.max(overlaps, 0)

                    for j in range(num_actions):
                        if max_overlaps[j] > 0.9: 
                            gt_tubes_list[j].append([i,j])
                            
                f_tubes = gt_tubes_list + f_tubes
                final_tubes = torch.cat([ boxes_.view(-1,num_frames,4).type_as(final_tubes),final_tubes])

                # evaluate again overlaps
                overlaps = tube_overlaps(final_tubes.view(-1,num_frames*4), boxes_.type_as(final_tubes))
                max_overlaps,_ = torch.max(overlaps,1)
                max_overlaps = max_overlaps.clamp_(min=0)

            
            ## TODO change numbers
            fg_tubes_indices = max_overlaps.ge(0.7).nonzero().view(-1)
            fg_num_tubes = fg_tubes_indices.numel()

            bg_tubes_indices = torch.nonzero((max_overlaps >= 0.1 ) &
                                         (max_overlaps <  0.3 )).view(-1)
                                         # (max_overlaps <  0.5 )).view(-1)
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

            tubes_labels[:fg_tubes_per_this_video] = labels[0]
            picked_tubes = final_tubes[keep_inds]

            fg_tubes_list = [f_tubes[i] for i in fg_tubes_indices]
            bg_tubes_list = [f_tubes[i] for i in bg_tubes_indices]

            f_tubes = fg_tubes_list + bg_tubes_list


        ##############################################

        if len(f_tubes) == 0:
            print('------------------')
            print('    empty tube    ')
            print(' vid_id :', vid_id)
            print('self.calc.thresh :',self.calc.thresh)
            return torch.Tensor([]).cuda(), torch.Tensor([]).cuda(), None

        max_seq = reduce(lambda x, y: y if len(y) > len(x) else x, f_tubes)
        max_length = len(max_seq)

        ## calculate input rois
        prob_out = torch.zeros(len(f_tubes), self.n_classes).cuda()
        final_feats = []
        len_tubes = []

        for i in range(len(f_tubes)):

            seq = f_tubes[i]
            len_tubes.append(torch.Tensor([len(seq)]))
            feats = torch.zeros(max_length,self.p_feat_size,self.sample_duration, self.POOLING_SIZE,self.POOLING_SIZE)

            for j in range(len(seq)):
                feats[j] = features[seq[j][0],seq[j][1]]

            if mode == 'extract':
                final_feats.append(feats)

            feats = torch.mean(feats, dim=0)

        if mode == 'extract':

            # now we use mean so we can have a tensor containing all features
            final_feats = torch.stack(final_feats).cuda()
            picked_tubes = picked_tubes.cuda()
            tubes_labels = tubes_labels.cuda()
            len_tubes = torch.stack(len_tubes).cuda()
            return final_feats, picked_tubes, tubes_labels, len_tubes

        # ##########################################
        # #           Time for Linear Loss         #
        # ##########################################

        cls_loss = torch.Tensor([0]).cuda()

        final_tubes = final_tubes.type_as(final_poss)
        # # classification probability

        if self.training:
            cls_loss = F.cross_entropy(prob_out.cpu(), tubes_labels.long()).cuda()

        if self.training:
            return None, None,  cls_loss, 
        else:
            prob_out = F.softmax(prob_out)

            # init padding tubes because of multi-GPU system
            if final_tubes.size(0) > conf.UPDATE_THRESH:
                _, indices = torch.sort(final_scores)
                final_tubes = final_tubes[indices[:conf.UPDATE_THRESH]].contiguous()
                prob_out = prob_out[indices[:conf.UPDATE_THRESH]].contiguous()

            ret_tubes = torch.zeros(1,conf.UPDATE_THRESH, ret_n_frames,4).type_as(final_tubes).float() -1
            ret_prob_out = torch.zeros(1,conf.UPDATE_THRESH,self.n_classes).type_as(final_tubes).float() - 1
            ret_tubes[0,:final_tubes.size(0),:num_frames] = final_tubes
            ret_prob_out[0,:final_tubes.size(0)] = prob_out
            return ret_tubes, ret_prob_out, torch.Tensor([final_tubes.size(0)]).cuda()
        
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

            act_net = ACT_net(self.classes,self.sample_duration)

            act_net.create_architecture()
            act_net.load_state_dict(new_state_dict)
            self.act_net = act_net

        else:
            self.act_net = ACT_net(self.classes,self.sample_duration)
            self.act_net.create_architecture()
            
        # load lstm
        if rnn_path != None:

            # act_rnn = Act_RNN(self.p_feat_size,int(self.p_feat_size/2),self.n_classes)
            # act_rnn_data = torch.load(rnn_path)
            # act_rnn.load(act_rnn_data)

            
            act_rnn = nn.Sequential(
                # nn.Linear(self.p_feat_size*self.sample_duration*self.POOLING_SIZE*self.POOLING_SIZE, 256),
                # nn.ReLU(True),
                # nn.Dropout(0.8),
                # nn.Linear(256,self.n_classes),
                # # nn.Linear(64*self.sample_duration, self.n_classes),
                # # nn.ReLU(True),
                # # nn.Dropout(0.8),
                # # nn.Linear(256,self.n_classes),
                nn.Linear(self.p_feat_size*self.sample_duration*self.POOLING_SIZE*self.POOLING_SIZE, self.n_classes),

            )
            act_rnn_data = torch.load(rnn_path)
            print('act_rnn_data :',act_rnn_data.keys())
            act_rnn.load_state_dict(act_rnn_data)
            self.act_rnn = act_rnn

        else:

            # self.act_rnn =Act_RNN(self.p_feat_size,int(self.p_feat_size/2),self.n_classes)
            self.act_rnn = nn.Sequential(
                # nn.Linear(self.p_feat_size*self.sample_duration*self.POOLING_SIZE*self.POOLING_SIZE, 256),
                # nn.ReLU(True),
                # nn.Dropout(0.8),
                # nn.Linear(256,self.n_classes),

                # nn.Linear(64*self.sample_duration, self.n_classes),
                # # nn.ReLU(True),
                # # nn.Dropout(0.8),
                # # nn.Linear(256,self.n_classes),
                nn.Linear(self.p_feat_size*self.sample_duration*self.POOLING_SIZE*self.POOLING_SIZE, self.n_classes),
            )
            for m in self.act_rnn.modules():
                if m == nn.Linear:
                    m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
