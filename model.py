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

from create_tubes_from_boxes import create_video_tube, create_tube_from_tubes, create_tube_with_frames
from connect_tubes import connect_tubes, get_gt_tubes_feats_label, get_tubes_feats_label
from resize_rpn import resize_boxes, resize_tube

from ucf_dataset import single_video

from config import cfg

from collections import OrderedDict


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
        # For now a linear classifier only



    def forward(self,n_devs, dataset_folder, vid_names, vid_id, spatial_transform, temporal_transform, boxes, mode, cls2idx, num_actions, num_frames, h_, w_):
        '''
        TODO describe procedure
        '''
        boxes = boxes.squeeze(0)

        print('boxes.shape :',boxes.shape)
        # print('boxes.type() :',boxes.type())
        ## define a dataloader for the whole video
        batch_size = n_devs * 4 # 
        num_workers = n_devs * 4
        num_images = 1
        rois_per_image = int(cfg.TRAIN.BATCH_SIZE / num_images) if self.training else 10

        data = single_video(dataset_folder,h_,w_, vid_names, vid_id, frames_dur= self.sample_duration, sample_size =self.sample_size,
                            boxes=boxes.cpu().numpy(), classes_idx=cls2idx)

        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, # num_workers=num_workers, pin_memory=True,
                                                  shuffle=False, num_workers=num_workers)
        n_clips = data.__len__()
        max_sim_actions = data.__max_sim_actions__()

        features = torch.zeros(n_clips, rois_per_image, self.p_feat_size, self.sample_duration)
        p_tubes = torch.zeros(n_clips, rois_per_image,  8) # all the proposed rois

        f_tubes = []

        if self.training:
            
            f_gt_tubes = torch.zeros(n_clips,boxes.size(0),7) # gt_tubes
            f_gt_feats = torch.zeros(n_clips,boxes.size(1),7) # gt_tubes' feat
            tubes_labels = torch.zeros(n_clips,rois_per_image)  # tubes rois
            loops = int(np.ceil(n_clips / batch_size))

            rpn_loss_cls_  = torch.zeros(loops) 
            rpn_loss_bbox_ = torch.zeros(loops)
            act_loss_bbox_ = torch.zeros(loops)

        print('n_clips :',n_clips)
        for step, dt in enumerate(data_loader):

            print('step :',step)
            # if step == 1:
            #     break
            clips, frame_indices, im_info, start_fr = dt
            print('frame_indices :',frame_indices.shape)
            boxes_ = boxes[:, frame_indices]

            gt_tubes = create_tube_with_frames(boxes_, im_info, self.sample_duration).permute(1,0,2)
            # print('tubes.shape :',tubes.shape)
            # print('tubes :',tubes)
            padding_lines = gt_tubes[:,:,-1].lt(1).nonzero()
            # print(padding_lines)
            for i in padding_lines:
                gt_tubes[i[0],i[1]] = torch.zeros((7)).type_as(gt_tubes)

            clips = clips.cuda()
            gt_tubes = gt_tubes.type_as(clips).cuda()
            im_info = im_info.cuda()
            start_fr = start_fr.cuda()

            print('gt_tubes.shape :',gt_tubes.shape)
            tubes,  bbox_pred, pooled_feat, \
            rpn_loss_cls,  rpn_loss_bbox, \
            act_loss_bbox, rois_label = self.act_net(clips,
                                                     im_info,
                                                     gt_tubes,
                                                     None,
                                                     start_fr)

            pooled_feat = pooled_feat.view(-1,rois_per_image,self.p_feat_size,self.sample_duration)
            indexes_ = (torch.arange(0, tubes.size(0))*int(self.sample_duration/2) + start_fr[0].cpu()).unsqueeze(1)
            indexes_ = indexes_.expand(tubes.size(0),tubes.size(1)).type_as(tubes)

            tubes[:,:,3] = tubes[:,:,3] + indexes_
            tubes[:,:,6] = tubes[:,:,6] + indexes_

            idx_s = step * batch_size 
            idx_e = step * batch_size + batch_size

            features[idx_s:idx_e] = pooled_feat
            p_tubes[idx_s:idx_e] = tubes

            if self.training:

                f_gt_tubes[idx_s:idx_e] = gt_tubes
                tubes_labels[idx_s:idx_e] = rois_label.squeeze(-1)

                rpn_loss_cls_[step] = rpn_loss_cls.mean().unsqueeze(0)
                rpn_loss_bbox_[step] = rpn_loss_bbox.mean().unsqueeze(0)
                act_loss_bbox_[step] = act_loss_bbox.mean().unsqueeze(0)

            # print('----------Out TPN----------')
            # # print('p_tubes.type() :',p_tubes.type())
            # # print('tubes.type() :',tubes.type())
            # print('----------Connect TUBEs----------')

            f_tubes = connect_tubes(f_tubes,tubes.cpu(), p_tubes, pooled_feat, rois_label, step*batch_size)
            # print('----------End Tubes----------')

        ###############################################
        #          Choose Tubes for RCNN\TCN          #
        ###############################################

        if self.training:

            f_rpn_loss_cls = rpn_loss_cls_.mean()
            f_rpn_loss_bbox = rpn_loss_bbox_.mean()
            f_act_loss_bbox = act_loss_bbox_.mean()

            ## first get video tube
            video_tubes = create_video_tube(torch.Tensor(boxes).type_as(clips))
            video_tubes =  resize_tube(video_tubes.unsqueeze(0), h_,w_,self.sample_size)
            
            # get gt tubes and feats
            gt_tubes_feats,gt_tubes_list = get_gt_tubes_feats_label(f_tubes, p_tubes, features, tubes_labels, f_gt_tubes)
            bg_tubes = get_tubes_feats_label(f_tubes, p_tubes, features, tubes_labels, video_tubes.cpu())

            gt_tubes_list = [x for x in gt_tubes_list if x != []]
            gt_lbl = torch.zeros(len(gt_tubes_list)).type_as(f_gt_tubes)
        
            for i in torch.arange(len(gt_tubes_list)).long():
                gt_lbl[i] = f_gt_tubes[gt_tubes_list[i][0][0],i,6]
            bg_lbl = torch.zeros((len(bg_tubes))).type_as(f_gt_tubes)
            
            ## concate fb, bg tubes
            f_tubes = gt_tubes_list + bg_tubes
            target_lbl = torch.cat((gt_lbl,bg_lbl),0)

        ##############################################

        max_seq = reduce(lambda x, y: y if len(y) > len(x) else x, f_tubes)
        max_length = len(max_seq)

        ## calculate input rois
        f_feat_mean = torch.zeros(len(f_tubes),self.p_feat_size).cuda() #.to(device)

        final_video_tubes = torch.zeros(len(f_tubes),6).cuda()
        
        for i in range(len(f_tubes)):

            seq = f_tubes[i]

            tmp_tube = torch.Tensor(len(seq),6)
            feats = torch.Tensor(len(seq),self.p_feat_size)
            for j in range(len(seq)):
                # print('features[seq[j]].mean(1).shape :',features[seq[j]].mean(1).shape)
                feats[j] = features[seq[j]].mean(1)
                tmp_tube[j] = p_tubes[seq[j]][1:7]
            final_video_tubes[i] = create_tube_from_tubes(tmp_tube)
            f_feat_mean[i,:] = torch.mean(feats,0).unsqueeze(0)
        
        # ######################################
        # #           Time for Linear          #
        # ######################################

        ## TODO : to add TCN or RNN

        cls_loss = 0
        prob_out = self.linear(f_feat_mean)
        # print('prob_out.shape :',prob_out.shape)
        # # classification probability

        if self.training:
            # print('prob_out.type() :',prob_out.type())
            # print('target_lbl.long() :',target_lbl.long().type())
            cls_loss = F.cross_entropy(prob_out.cpu(), target_lbl.long())

        if self.training:
            return tubes, bbox_pred,  prob_out, f_rpn_loss_cls, f_rpn_loss_bbox, f_act_loss_bbox, cls_loss, 
        else:
            return final_video_tubes, bbox_pred, prob_out
        # return  torch.Tensor([0]).cuda(),torch.Tensor([0]).cuda(),Variable(torch.Tensor([0]).cuda()),Variable(torch.Tensor([0]).cuda()),torch.Tensor([0]).cuda(),torch.Tensor([0]).cuda(),   Variable( torch.Tensor([0]).cuda(), requires_grad=True)

    # def create_architecture(self):

    #     self.act_net.create_architecture()

    def deactivate_action_net_grad(self):

        for p in self.act_net.parameters() : p.requires_grad=False
        # for key, value in dict(self.named_parameters()).items():
        #     print(key, value.requires_grad)

    def load_part_model(self, action_model_path=None, linear_path=None):


        if action_model_path != None:
            
            act_data = torch.load('./action_net_model.pwf')

            ## to remove module
            new_state_dict = OrderedDict()
            for k, v in act_data.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v

            act_net = ACT_net(self.classes,self.sample_duration)

            act_net.create_architecture()
            act_net.load_state_dict(new_state_dict)
            self.act_net = act_net

        else:
            self.act_net = ACT_net(self.classes,self.sample_duration)
            self.act_net.create_architecture()
            
        if linear_path != None:

            linear = nn.Linear(self.p_feat_size, self.n_classes).cuda()

            linear_data = torch.load(linear_path)
            linear.load_state_dict(linear_data)
            self.linear = linear
        else:
            self.linear = nn.Linear(self.p_feat_size, self.n_classes).cuda()

