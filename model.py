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

        # self.act_net = ACT_net(actions,sample_duration)

        ## general options
        self.sample_duration = sample_duration
        self.sample_size = sample_size
        self.step = int(self.sample_duration/2)
        self.p_feat_size = 64 # 128 # 256 # 512
        
        # For connection 
        self.max_num_tubes = conf.MAX_NUMBER_TUBES
        self.connection_thresh = conf.CONNECTION_THRESH
        self.update_thresh_step = conf.UPDATE_THRESH
        self.calc = Calculator(self.max_num_tubes, self.update_thresh_step, self.connection_thresh)
        


    def forward(self,n_devs, dataset_folder, vid_names, clips, vid_id, boxes, mode, cls2idx, num_actions, num_frames, h_, w_):
        '''
        TODO describe procedure
        '''

        # print('boxes.shape :',boxes.shape)

        ## define a dataloader for the whole video
        # print('----------Inside----------')
        print('num_frames :',num_frames)
        print('clips.shape :',clips.shape)

        clips = clips.squeeze(0)
        clips = clips[:num_frames]

        if self.training:
            boxes = boxes.squeeze(0).permute(1,0,2).cpu()
            boxes = boxes[:num_frames,:num_actions]

        batch_size = 2 # 
        # batch_size = 16 # 

        num_images = 1
        rois_per_image = int(cfg.TRAIN.BATCH_SIZE / num_images) if self.training else 150

        data = single_video(dataset_folder,h_,w_, vid_names, vid_id, frames_dur= self.sample_duration, sample_size =self.sample_size,
                            classes_idx=cls2idx, n_frames=num_frames)

        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, pin_memory=False,# num_workers=num_workers, pin_memory=True,
                                                  # shuffle=False, num_workers=8)
                                                  shuffle=False)

        n_clips = data.__len__()

        features = torch.zeros(n_clips, rois_per_image, self.p_feat_size, self.sample_duration)
        p_tubes = torch.zeros(n_clips, rois_per_image,  self.sample_duration*4) # all the proposed tube-rois
        actioness_score = torch.zeros(n_clips, rois_per_image)
        overlaps_scores = torch.zeros(n_clips-1, rois_per_image, rois_per_image)

        f_tubes = []

        if self.training:
            
            f_gt_tubes = torch.zeros(n_clips,num_actions,self.sample_duration*4) # gt_tubes
            tubes_labels = torch.zeros(n_clips,rois_per_image)  # tubes rois
            loops = int(np.ceil(n_clips / batch_size))
            labels = torch.zeros(num_actions)

            for i in range(num_actions):
                idx = boxes[:,i,4].nonzero().view(-1)
                labels[i] = boxes[i,idx[0],4]

        for step, dt in enumerate(data_loader):

            # if step == 1:
            #     break
            print('\tstep :',step)

            frame_indices, im_info, start_fr = dt
            # print('frame_indices :',frame_indices)
            clips_ = clips[frame_indices].cuda()

            if self.training:
                boxes_ = boxes[frame_indices].cuda()
                box_ = boxes_.permute(0,2,1,3).float().contiguous()[:,:,:,:-1]
            else:
                box_ = None
                
            # gt_tubes = create_tube_with_frames(boxes_.permute(0,2,1,3), im_info, self.sample_duration)
            # gt_tubes_ = gt_tubes.type_as(clips).cuda()

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

            pooled_feat = pooled_feat.view(-1,rois_per_image,self.p_feat_size,self.sample_duration)

            indexes_ = (torch.arange(0, tubes.size(0))*int(self.sample_duration/2) + start_fr[0].cpu()).unsqueeze(1)
            indexes_ = indexes_.expand(tubes.size(0),tubes.size(1)).type_as(tubes)

            idx_s = step * batch_size 
            idx_e = step * batch_size + batch_size

            features[idx_s:idx_e] = pooled_feat
            p_tubes[idx_s:idx_e,] = tubes[:,:,1:-1]
            actioness_score[idx_s:idx_e] = tubes[:,:,-1]

            if self.training:

                box = boxes_.permute(0,2,1,3).contiguous()[:,:,:,:-2]
                box = box.contiguous().view(box.size(0),box.size(1),-1)

                f_gt_tubes[idx_s:idx_e] = box
                tubes_labels[idx_s:idx_e] = rois_label.squeeze(-1).type_as(tubes_labels)

        ########################################################
        #          Calculate overlaps and connections          #
        ########################################################

        overlaps_scores = torch.zeros(n_clips-1, rois_per_image, rois_per_image).type_as(overlaps_scores)

        for i in range(n_clips-1):
            overlaps_scores[i] = tube_overlaps(p_tubes[i,:,int(self.sample_duration*4/2):],p_tubes[i+1,:,:int(self.sample_duration*4/2)])

        final_scores, final_poss = self.calc(overlaps_scores.cuda(), actioness_score.cuda(),
                                             torch.Tensor([n_clips]),torch.Tensor([rois_per_image]))
        ## Now connect the tubes
        final_tubes = torch.zeros(final_poss.size(0), num_frames, 4)
        f_tubes  = []
        for i in range(final_poss.size(0)):
            tub = []
            for j in range(final_poss.size(1)):

                curr_ = final_poss[i,j]
                start_fr = curr_[0]* int(self.sample_duration/2)
                end_fr = curr_[0]*int(self.sample_duration/2)+self.sample_duration

                if curr_[0] == -1:
                    break
                
                curr_frames = p_tubes[curr_[0], curr_[1]]
                tub.append((curr_[0].item(),  curr_[1].item()))
                ## TODO change with avg
                final_tubes[i,start_fr:end_fr] =  torch.max( curr_frames.view(-1,4), final_tubes[i,start_fr:end_fr])
            f_tubes.append(tub)


        ###################################################
        #          Choose gth Tubes for RCNN\TCN          #
        ###################################################
        if self.training:

            # # get gt tubes and feats
            ##  calculate overlaps
            boxes_ = boxes.permute(1,0,2).contiguous()
            boxes_ = boxes_[:,:,:4].contiguous().view(num_actions,-1)

            overlaps = tube_overlaps(final_tubes.view(-1,num_frames*4), boxes_.type_as(final_tubes))
            max_overlaps,_ = torch.max(overlaps,1)
            max_overlaps = max_overlaps.clamp_(min=0)
            ## TODO change numbers
            bg_tubes_indices = max_overlaps.lt(0.3).nonzero()
            bg_tubes_indices_picked = (torch.rand(5)*bg_tubes_indices.size(0)).long()
            bg_tubes_list = [f_tubes[i] for i in bg_tubes_indices[bg_tubes_indices_picked]]
            bg_labels = torch.zeros(len(bg_tubes_list))

            gt_tubes_list = [[] for i in range(num_actions)]

            for i in range(n_clips):

                overlaps = tube_overlaps(p_tubes[i], f_gt_tubes[i])
                max_overlaps, argmax_overlaps = torch.max(overlaps, 0)
                
                for j in range(num_actions):
                    if max_overlaps[j] == 1.0: 
                        gt_tubes_list[j].append((i,j))

            ## concate fb, bg tubes
            f_tubes = gt_tubes_list + bg_tubes_list
            target_lbl = torch.cat([labels, bg_labels],dim=0)

        ##############################################
 
        max_seq = reduce(lambda x, y: y if len(y) > len(x) else x, f_tubes)
        max_length = len(max_seq)

        ## calculate input rois
        ## f_feats.shape : [#f_tubes, max_length, 512]
        final_video_tubes = torch.zeros(len(f_tubes),6).cuda()
        prob_out = torch.zeros(len(f_tubes), self.n_classes).cuda()

        for i in range(len(f_tubes)):

            seq = f_tubes[i]
            tmp_tube = torch.Tensor(len(seq),6)
            feats = torch.Tensor(len(seq),self.p_feat_size)
            
            for j in range(len(seq)):

                feats[j] = features[seq[j][0],seq[j][1]].mean(1)
                tmp_tube[j] = p_tubes[seq[j]][1:7]

            prob_out[i] = self.act_rnn(feats.cuda())
            if prob_out[i,0] != prob_out[i,0]:
                print('tmp_tube :',tmp_tube, ' prob_out :', prob_out ,' feats :',feats.cpu().numpy(), ' numpy(), feats.shape  :,', feats.shape ,' target_lbl :',target_lbl, \
                      ' \ntmp_tube :',tmp_tube, )
                exit(-1)

            
        # ##########################################
        # #           Time for Linear Loss         #
        # ##########################################

        cls_loss = torch.Tensor([0]).cuda()

        final_tubes = final_tubes.type_as(final_poss)
        # # classification probability
        if self.training:
            cls_loss = F.cross_entropy(prob_out.cpu(), target_lbl.long()).cuda()

        if self.training:
            return final_tubes, prob_out,  cls_loss, 
        else:
            return final_tubes, prob_out, None

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

            act_rnn = Act_RNN(self.p_feat_size,int(self.p_feat_size/2),self.n_classes)

            act_rnn_data = torch.load(rnn_path)
            act_rnn.load(act_rnn_data)
            self.act_rnn = act_rnn

        else:
            self.act_rnn =Act_RNN(self.p_feat_size,int(self.p_feat_size),self.n_classes)

