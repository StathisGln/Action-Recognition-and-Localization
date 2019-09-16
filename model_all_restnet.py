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
from rest_model import RestNet
from all_scores.calc import Calculator

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
        self.POOLING_SIZE = 7
        self.pooling_time = 2
        # self.POOLING_SIZE = 4

        # For connection 
        self.max_num_tubes = conf.MAX_NUMBER_TUBES
        self.connection_thresh = conf.CONNECTION_THRESH
        self.update_thresh = conf.UPDATE_THRESH
        self.n_ret_tubes = 500 if self.training else conf.ALL_SCORES_THRESH
        self.calc = Calculator(self.max_num_tubes, self.update_thresh, self.connection_thresh, self.n_ret_tubes)
        
        # for nms
        self.post_nms_topN = 50
        
    def forward(self,n_devs, dataset_folder, vid_names, clips, vid_id, boxes, mode, cls2idx, num_actions, num_frames, h_, w_):
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

        # init_time = time.time()
        # print('init_time :',init_time)
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
            # print('pooled_feat.shape :',pooled_feat.shape)
            # print('pooled_feat.shape :',pooled_feat.device)

            # # regression
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

        # proposals_time = time.time()
        # print('proposals_time :',proposals_time-init_time)

        # calculate 500 best tubes
        pos, conn_scores = self.calc(torch.Tensor([n_clips]),torch.Tensor([rois_per_image]),
                                actioness_score, overlaps_scores)
        pos = pos.view(-1,n_clips,2)

        # print('connect :',time.time()-proposals_time)
        final_tubes = torch.zeros(pos.size(0), num_frames, 4).type_as(conn_scores)
        f_tubes  = []

        # print('final_tubes.device :',final_tubes.device)
        for i in range(pos.size(0)):

            tub = []
            for j in range(pos.size(1)):
                
                curr_ = pos[i,j]
                start_fr = curr_[0]* int(self.step)
                end_fr = min((curr_[0]*int(self.step)+self.sample_duration).type_as(num_frames), num_frames).type_as(start_fr)

                if curr_[0] == -1:
                    break
                
                curr_frames = p_tubes[curr_[0], curr_[1]]
                tub.append((curr_[0].item(),  curr_[1].item()))
                final_tubes[i,start_fr:end_fr] =  torch.max( curr_frames.view(-1,4).contiguous()[:(end_fr-start_fr).long()],
                                                             final_tubes[i,start_fr:end_fr].type_as(curr_frames))

            f_tubes.append(tub)

        # # print('p_tubes.device :',p_tubes.device)
        # for i in range(pos.size(0)):

        #     # tub = []

        #     non_zr = pos[i,:,0].ne(-1).nonzero().view(-1)
        #     thesis = pos[i,non_zr].long()

        #     p_tubes_ = p_tubes[thesis[:,0], thesis[:,1]]

        #     start_fr = thesis[0,0]* int(self.step)
        #     end_fr = torch.min((thesis[-1,0]+1)*self.step, num_frames)
        #     print('p_tubes[thesis[:,0], thesis[:,1]].contiguous().view(-1,4).shape :',\
        #           p_tubes[thesis[:,0], thesis[:,1]].contiguous().view(-1,4).shape)
        #     print('final_tubes[i, start_fr:end_fr] :',final_tubes[i, start_fr:end_fr].shape)

        #     final_tubes[i, start_fr:end_fr] = p_tubes[thesis[:,0], thesis[:,1]].contiguous().\
        #                                       view(-1,4)
        #                                       # view(non_zr.size(0)*self.sample_duration,4)

        #     f_tubes.append(thesis)


        # connect_and_translate_time = time.time()
        # print('connect_and_translate :',connect_and_translate_time-proposals_time)

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
        # f_tubes = [f_tubes[int(i)] for i in keep_idx_i]

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
        # feats = torch.zeros(29,self.p_feat_size, self.sample_duration)
        for i in range(len(f_tubes)):

            seq = f_tubes[i]
            feats = torch.Tensor(max_length,self.p_feat_size,self.sample_duration, \
                                 self.POOLING_SIZE,self.POOLING_SIZE).type_as(features)
            # feats = torch.Tensor(5,self.p_feat_size, self.POOLING_SIZE,self.POOLING_SIZE)
            
            for j in range(len(seq)):
                # feats[j] = features[seq[j][0],seq[j][1]].mean(1)
                feats[j] = features[seq[j][0],seq[j][1]]

            # final_feats.append(torch.mean(feats, dim=0))
            # final_feats.append(feats.max(0)[0])
            final_feats.append(feats)
            # feats = torch.mean(feats, dim=0)

        final_feats = torch.stack(final_feats)

        # ##########################################
        # #           Time for Linear Loss         #
        # ##########################################

        # print('final_feats.shape:',final_feats.shape)
        # print('final_feats.shape:',final_feats.device)
        # print('svmWeights.device :',self.svmWeights.device)
        
        f_features = self.temporal_pool(final_feats)
        f_features= f_features.max(3)[0]
        f_features = f_features.permute(0,2,1,3,4).contiguous()

        f_features = self.rest_net(f_features).mean(4).mean(3).squeeze()

        scores = self.cls(f_features)

        # print('rest time :',time.time() - connect_and_translate_time)
        if self.training:

            return None, None,  cls_loss, 

        else:
            # prob_out = F.softmax(prob_)
            prob_out = scores

            # init padding tubes because of multi-GPU system
            if final_tubes.size(0) > conf.ALL_SCORES_THRESH:
                _, indices = torch.sort(conn_scores)
                print('indices.shape :',indices.shape)
                final_tubes = final_tubes[indices[:conf.ALL_SCORES_THRESH]].contiguous()
                prob_out = prob_out[indices[:conf.ALL_SCORES_THRESH]].contiguous()

            ret_tubes = torch.zeros(1,conf.ALL_SCORES_THRESH, ret_n_frames,4).type_as(final_tubes).float() -1
            ret_prob_out = torch.zeros(1,conf.ALL_SCORES_THRESH,self.n_classes).type_as(final_tubes).float() - 1
            ret_tubes[0,:final_tubes.size(0),:num_frames] = final_tubes
            ret_prob_out[0,:final_tubes.size(0)] = prob_out

            return ret_tubes, ret_prob_out, torch.Tensor([final_tubes.size(0)]).cuda()
        
            # return final_tubes, prob_out, None

    def temporal_pool(self, features):

        batch_size = features.size(0)
        n_filters = features.size(2)
        x_axis = features.size(4)
        y_axis = features.size(5)
        
        indexes = torch.linspace(0, features.size(1), self.pooling_time+1).int()
        ret_feats = features.new(batch_size,self.pooling_time, n_filters,\
                                self.sample_duration,x_axis, y_axis).zero_()

        if features.size(1) < 2:

            ret_feats[0] = features
            return ret_feats
    
        for i in range(self.pooling_time):
            
            t = features[:,indexes[i]:indexes[i+1]].permute(0,2,1,3,4,5).\
                contiguous().view(batch_size*n_filters,-1,self.sample_duration,x_axis,y_axis)
            t = t.view(t.size(0),t.size(1),t.size(2),-1)
            t = F.max_pool3d(t,kernel_size=(indexes[i+1]-indexes[i],1,1)).squeeze()

            ret_feats[:,i] = t.view(batch_size,n_filters,self.sample_duration,x_axis,y_axis)

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

            model = RestNet(self.classes, self.sample_duration)
            model.create_architecture()
            model_data = torch.load(rnn_path)
            model.load_state_dict(model_data)

            self.rest_net = nn.Sequential(model.top_net)
            self.cls = model.cls
        else:
 
            model = RestNet(self.classes, self.sample_duration)
            model.create_architecture()

            self.rest_net = nn.Sequential(model.top_net)
            self.cls = model.cls


