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
from calc_score.calc import Calculator

from create_tubes_from_boxes import create_video_tube, create_tube_from_tubes, create_tube_with_frames
from connect_tubes import connect_tubes, get_gt_tubes_feats_label, get_tubes_feats_label
from resize_rpn import resize_boxes, resize_tube

from ucf_dataset import single_video

from config import cfg
from bbox_transform import bbox_overlaps_connect, bbox_transform_inv, bbox_overlaps_batch_3d, clip_boxes
from collections import OrderedDict

# import gc
# from pympler.asizeof import asizeof

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

        # self.act_net = ACT_net(actions,sample_duration)

        ## general options
        self.sample_duration = sample_duration
        self.sample_size = sample_size
        self.step = int(self.sample_duration/2)
        self.p_feat_size = 256 # 512
        
        # For connection 
        self.max_num_tubes = conf.MAX_NUMBER_TUBES
        self.connection_thresh = conf.CONNECTION_THRESH
        self.update_thresh_step = conf.UPDATE_THRESH
        self.calc = Calculator(self.max_num_tubes, self.update_thresh_step, self.connection_thresh)
        


    def forward(self,n_devs, dataset_folder, vid_names, clips, vid_id, boxes, mode, cls2idx, num_actions, num_frames, h_, w_):
        '''
        TODO describe procedure
        '''

        if self.training :
            self.calc.training = True
            boxes = boxes.squeeze(0).permute(1,0,2).cpu()
            boxes = boxes[:num_frames,:num_actions]
            
        clips = clips.squeeze(0)
        # print('----------Inside----------')
        # print('boxes.shape :',boxes.shape)

        ## define a dataloader for the whole video

        self.act_net.module.reg_layer.eval()

        clips = clips[:num_frames]

        # batch_size = 2 # 
        batch_size = 16 # 

        num_images = 1
        rois_per_image = int(cfg.TRAIN.BATCH_SIZE / num_images) *2 if self.training else 25 *2

        data = single_video(dataset_folder,h_,w_, vid_names, vid_id, frames_dur= self.sample_duration, sample_size =self.sample_size,
                            classes_idx=cls2idx, n_frames=num_frames)

        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, pin_memory=False,# num_workers=num_workers, pin_memory=True,
                                                  # shuffle=False, num_workers=8)
                                                  shuffle=False)

        n_clips = data.__len__()

        # features = torch.zeros(n_clips, rois_per_image, self.p_feat_size, self.sample_duration)
        features = torch.zeros(n_clips, rois_per_image, self.p_feat_size, self.sample_duration)

        p_tubes = torch.zeros(n_clips, rois_per_image,  6) # all the proposed rois
        sgl_rois_preds = torch.zeros(n_clips, self.sample_duration, rois_per_image, 4) 
        actioness_score = torch.zeros(n_clips, rois_per_image)
        f_tubes = []

        if self.training:
            
            f_gt_tubes = torch.zeros(n_clips,num_actions,7) # gt_tubes
            tubes_labels = torch.zeros(n_clips,rois_per_image)  # tubes rois
            loops = int(np.ceil(n_clips / batch_size))

            rpn_loss_cls_     = torch.zeros(loops) 
            rpn_loss_bbox_    = torch.zeros(loops)
            act_loss_bbox_    = torch.zeros(loops)
            rpn_loss_cls_16_  = torch.zeros(loops) 
            rpn_loss_bbox_16_ = torch.zeros(loops)
            act_loss_bbox_16_ = torch.zeros(loops)

        for step, dt in enumerate(data_loader):

            # if step == 1:
            #     break

            frame_indices, im_info, start_fr = dt
            clips_ = clips[frame_indices].cuda()
            
            if self.training:
                boxes_ = boxes[frame_indices].cuda()
                gt_tubes = create_tube_with_frames(boxes_.permute(0,2,1,3), im_info, self.sample_duration)
                gt_tubes_ = gt_tubes.type_as(clips).cuda()

                boxes_  =boxes_.permute(0,2,1,3).float()
            else:
                
                boxes_ = None
                gt_tubes_ = None
                
            im_info = im_info.cuda()
            start_fr = start_fr.cuda()

            tubes,  pooled_feat, \
            rpn_loss_cls,  rpn_loss_bbox, \
            rpn_loss_cls_16, \
            rpn_loss_bbox_16, rois_label, \
            sgl_rois_bbox_pred, sgl_rois_bbox_loss = self.act_net(clips_.permute(0,2,1,3,4),
                                                        im_info,
                                                        gt_tubes_,
                                                        boxes_,
                                                        start_fr)

            pooled_feat = pooled_feat.view(-1,rois_per_image,self.p_feat_size,self.sample_duration)

            indexes_ = (torch.arange(0, tubes.size(0))*int(self.sample_duration/2) + start_fr[0].cpu()).unsqueeze(1)
            indexes_ = indexes_.expand(tubes.size(0),tubes.size(1)).type_as(tubes)

            tubes[:,:,3] = tubes[:,:,3] + indexes_
            tubes[:,:,6] = tubes[:,:,6] + indexes_

            idx_s = step * batch_size 
            idx_e = step * batch_size + batch_size

            features[idx_s:idx_e] = pooled_feat
            p_tubes[idx_s:idx_e ] = tubes[:,:,1:7]
            sgl_rois_preds[idx_s:idx_e] = sgl_rois_bbox_pred
            actioness_score[idx_s:idx_e] = tubes[:,:,7]

            if self.training:

                f_gt_tubes[idx_s:idx_e] = gt_tubes.type_as(f_gt_tubes)
                tubes_labels[idx_s:idx_e] = rois_label.squeeze(-1).type_as(tubes_labels)

                rpn_loss_cls_[step] = rpn_loss_cls.mean().unsqueeze(0)
                rpn_loss_bbox_[step] = rpn_loss_bbox.mean().unsqueeze(0)
                rpn_loss_cls_16_[step] = rpn_loss_cls_16.mean().unsqueeze(0)
                rpn_loss_bbox_16_[step] = rpn_loss_bbox_16.mean().unsqueeze(0)

        #######################################################################
        #          Calculate overlaps and candidate tube-connections          #
        #######################################################################
        if self.training :
            # limit = int(rois_per_image/2)
            limit = 10
                    
        else:
            limit = rois_per_image

        overlaps_scores = torch.zeros(n_clips-1, int(rois_per_image/2) , int(rois_per_image/2) )
        
        # for i in range(n_clips-1):
        im_shape = torch.Tensor([self.sample_size,self.sample_size]).unsqueeze(0).expand(self.sample_duration,2)

        for i in range(1):
            # 1 from tubes to rois
            rois_curr = torch.zeros(limit, self.sample_duration, 4).type_as(p_tubes)
            rois_next = torch.zeros(limit, self.sample_duration, 4).type_as(p_tubes)

            for j in range(limit):

                r = p_tubes[i,j]
                inds = torch.arange(r[2].long(), (r[5]+1).long()) - (i * int( self.sample_duration/2))
                rois_curr[j,inds]= r[[0,1,3,4]]

                # for next_boxes
                r = p_tubes[i+1,j]
                inds = torch.arange(r[2].long(), (r[5]+1).long()) - ((i+1)* int(self.sample_duration/2))
                rois_next[j,inds]= r[[0,1,3,4]]

            # 2 transform inv
            empty_lines_rois_curr  = rois_curr.sum(2).eq(0) # 1 is where there is empty line
            empty_lines_rois_next  = rois_next.sum(2).eq(0) # 1 is where there is empty line

            rois_curr = rois_curr.permute(1,0,2)
            rois_next = rois_next.permute(1,0,2)

            empty_lines_rois_curr = empty_lines_rois_curr.permute(1,0)
            empty_lines_rois_next = empty_lines_rois_next.permute(1,0)

            ## TODO change only sgl_rois_preds
            # get  padding boxes lines
            rois_curr = bbox_transform_inv(rois_curr, sgl_rois_preds[i,:,:limit], rois_curr.size(0))
            rois_curr = clip_boxes(rois_curr, im_shape, rois_curr.size(0))

            rois_next = bbox_transform_inv(rois_curr, sgl_rois_preds[i+1,:,:limit], rois_curr.size(0))
            rois_next = clip_boxes(rois_next, im_shape, rois_next.size(0))

            # rois_curr is frames 
            for j in range(rois_curr.size(0)):

                poss_ = empty_lines_rois_curr[j].nonzero()
                if poss_.nelement() == 0:
                    continue
                poss_ = poss_.view(-1)
                rois_curr[j,poss_] = 0 #torch.zeros(4)

            for j in range(rois_next.size(0)):                
                poss_ = empty_lines_rois_next[j].nonzero()
                if poss_.nelement() == 0:
                    continue
                poss_ = poss_.view(-1)
                rois_next[j,poss_] = 0 #torch.zeros(4)

            rois_curr  = rois_curr.permute(1,0,2)
            rois_next  = rois_next.permute(1,0,2)
            ## TODO add clip

            if self.training:

                boxes_ = boxes[ i * int(self.sample_duration/2): i * int(self.sample_duration/2)+self.sample_duration].permute(1,0,2)
                boxes_n = boxes[ (i+1) * int(self.sample_duration/2): (i+1) * int(self.sample_duration/2)+self.sample_duration].permute(1,0,2)

                for t in range(f_gt_tubes.size(1)):

                    # for curr tensor
                    overlap = bbox_overlaps_batch_3d(p_tubes[i,:limit], f_gt_tubes[i,t].unsqueeze(0).unsqueeze(0))
                    overlap_ind = overlap.eq(1.).nonzero()

                    if overlap_ind.nelement() != 0:
                        overlap_ind = overlap_ind[:,1].view(-1)
                        rois_curr[overlap_ind] = boxes_[t,:,:4].float().unsqueeze(0)

                    # for next tensor
                    overlap = bbox_overlaps_batch_3d(p_tubes[(i+1),:limit], f_gt_tubes[(i+1),t].unsqueeze(0).unsqueeze(0))
                    overlap_ind = overlap.eq(1.).nonzero()
                    if overlap_ind.nelement() != 0:
                        overlap_ind = overlap_ind[:,1].view(-1)
                        rois_next[overlap_ind] = boxes_n[t,:,:4].float()

            # 4 calculate overlaps
            ret  = bbox_overlaps_connect(rois_curr, rois_next, \
                                         self.sample_duration, middle_fr = int(self.sample_duration/2))
            # print('ret.shape :',ret.shape)

        if self.training:
            actioness_ = actioness_score[:,:limit].contiguous()

        final_scores, final_tubes = self.calc(overlaps_scores.cuda(), actioness_score.cuda(),
                                             torch.Tensor([n_clips]),torch.Tensor([rois_per_image/2]))
        print('final_scores.shape :',final_scores.shape)

        if self.training:
            _indx = final_scores.ne(2).nonzero()
            if _indx.nelement() != 0:
                _indx = _indx.view(-1)
                
                final_tubes = final_tubes[_indx]
                final_scores = final_scores[_indx]
                _,_indx  = torch.sort(final_scores)
                bg_tubes_t = final_tubes[_indx[:10]].contiguous() ## TODO find best number
                # print('bg_tubes_t.shape :',bg_tubes_t.shape)
                # bg_tubes = final_tubes[_indx[:10]].cpu().tolist()       ## TODO find best number
                bg_tubes =  [[] for i in range( bg_tubes_t.size(0))]

                for i in range(bg_tubes_t.size(0)):
                    for j in range(n_clips):
                        if bg_tubes_t[i,j,0] == -1:
                            break
                        bg_tubes[i] += [(bg_tubes_t[i,j,0].tolist(), bg_tubes_t[i,j,1].tolist())]
                bg_tubes = [x for x in bg_tubes if x != []] # remove empty lists
                bg_lbl = torch.zeros((len(bg_tubes))).type_as(f_gt_tubes)

                # for i in range(bg_tubes_t.size(0)):
                #     print(bg_tubes[i])
                # print('bg_tubes_t :',bg_tubes_t)
                # exit(-1)
            else:
                print('only gt')
                bg_lbl = torch.Tensor([])
                bg_tubes =  []
        else:
            f_tubes = [[] for i in range(final_scores.size(0))]
            for i in range(final_scores.size(0)):
                for j in range(n_clips):
                    if final_tubes[i,j,0] == -1:
                        break
                    f_tubes[i] +=[(final_tubes[i,j,0].tolist(), final_tubes[i,j,1].tolist())]

        ###############################################
        #          Choose Tubes for RCNN\TCN          #
        ###############################################
 
        if self.training:

            f_rpn_loss_cls = rpn_loss_cls_.mean().cuda()
            f_rpn_loss_bbox = rpn_loss_bbox_.mean().cuda()
            f_act_loss_bbox = act_loss_bbox_.mean().cuda()

            # get gt tubes and feats
            gt_tubes_feats,gt_tubes_list = get_gt_tubes_feats_label(None, p_tubes, features, tubes_labels, f_gt_tubes)

            gt_tubes_list = [x for x in gt_tubes_list if x != []]
            gt_lbl = torch.zeros(len(gt_tubes_list)).type_as(f_gt_tubes)
        
            for i in torch.arange(len(gt_tubes_list)).long():
                gt_lbl[i] = f_gt_tubes[gt_tubes_list[i][0][0],i,6]

            ## concate fb, bg tubes
            f_tubes = gt_tubes_list +  bg_tubes
            target_lbl = torch.cat( (gt_lbl, bg_lbl))

        ##############################################

        max_seq = reduce(lambda x, y: y if len(y) > len(x) else x, f_tubes)
        max_length = len(max_seq)

        ## calculate input rois
        ## f_feats.shape : [#f_tubes, max_length, 512]
        final_video_tubes = torch.zeros(len(f_tubes),6).cuda()
        prob_out = torch.zeros(len(f_tubes), self.n_classes).cuda()

        f_features = torch.zeros(len(f_tubes), max_length, self.p_feat_size, self.sample_duration)
        final_tubes = torch.zeros(len(f_tubes), max_length, 6) - 1
        final_sgl_fr_pred = torch.zeros(len(f_tubes), max_length, self.sample_duration, 4)
        final_poss = torch.zeros(len(f_tubes), max_length, 2) - 1

        for i in range(len(f_tubes)):

            seq = f_tubes[i]
            tmp_tube = torch.Tensor(len(seq),6)
            feats = torch.Tensor(len(seq),self.p_feat_size)

            for j in range(len(seq)):
                final_poss[i,j,0] = seq[j][0]
                final_poss[i,j,1] = seq[j][1]
                f_features[i,j] = features[seq[j][0],seq[j][1]]
                final_tubes[i,j] = p_tubes[seq[j]]
                tmp_tube[j] = p_tubes[seq[j]]

                final_sgl_fr_pred[i][j] = sgl_rois_preds[seq[j][0],:, seq[j][1]]

            final_video_tubes[i] = create_tube_from_tubes(tmp_tube).type_as(clips)

        # ##########################################
        # #           Time for Linear Loss         #
        # ##########################################

        cls_loss = torch.Tensor([0]).cuda()

        # # classification probability
        if self.training:
            cls_loss = F.cross_entropy(prob_out.cpu(), target_lbl.long()).cuda()
            return final_video_tubes, f_features,  final_tubes, target_lbl, final_poss, final_sgl_fr_pred
        else:
            return final_video_tubes, f_features,  final_tubes, None, final_poss, final_sgl_fr_pred
        # return final_video_tubes, None,  prob_out, f_rpn_loss_cls, f_rpn_loss_bbox, f_act_loss_bbox, cls_loss, 


    def deactivate_action_net_grad(self):

        for p in self.act_net.parameters() : p.requires_grad=False
        # for key, value in dict(self.named_parameters()).items():
        #     print(key, value.requires_grad)

    def load_part_model(self, action_model_path=None, rnn_path=None):


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
            
        if rnn_path != None:

            act_rnn = Act_RNN(256,128,self.n_classes)
            linear = nn.Linear(self.p_feat_size, self.n_classes).cuda()

            act_rnn_data = torch.load(rnn_path)
            act_rnn.load(act_rnn_data)
            self.act_rnn = act_rnn
        else:
            self.act_rnn =Act_RNN(256,128,self.n_classes)

