import os
import numpy as np
import glob
from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from action_net import ACT_net
from act_rnn import Act_RNN

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

    def forward(self,n_devs, dataset_folder, vid_names, clips, vid_id, boxes, mode, cls2idx, num_actions, num_frames, h_, w_):
        '''
        TODO describe procedure
        '''
        # print('boxes.shape :',boxes.shape)
        boxes = boxes.squeeze(0).permute(1,0,2).cpu()
        clips = clips.squeeze(0)

        ## define a dataloader for the whole video
        boxes = boxes[:num_frames,:num_actions]
        clips = clips[:num_frames]

        batch_size = 16 # 
        num_images = 1
        rois_per_image = int(cfg.TRAIN.BATCH_SIZE / num_images) * 2 if self.training else 10 *2

        data = single_video(dataset_folder,h_,w_, vid_names, vid_id, frames_dur= self.sample_duration, sample_size =self.sample_size,
                            classes_idx=cls2idx, n_frames=num_frames)

        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, pin_memory=False,# num_workers=num_workers, pin_memory=True,
                                                  # shuffle=False, num_workers=8)
                                                  shuffle=False)
        n_clips = data.__len__()

        features = torch.zeros(n_clips, rois_per_image, self.p_feat_size, self.sample_duration)
        p_tubes = torch.zeros(n_clips, rois_per_image,  8) # all the proposed rois

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

        for step, dt in enumerate(data_loader):

            # print('Memory :',torch.cuda.memory_allocated(device=None))
            frame_indices, im_info, start_fr = dt
            boxes_ = boxes[frame_indices].cuda()
            clips_ = clips[frame_indices].cuda()

            gt_tubes = create_tube_with_frames(boxes_.permute(0,2,1,3), im_info, self.sample_duration)
            gt_tubes_ = gt_tubes.type_as(clips).cuda()
            im_info = im_info.cuda()
            start_fr = start_fr.cuda()

            tubes,  _, pooled_feat, \
            _, _, _, _, _, _, rois_label, _, _ = self.act_net(clips_.permute(0,2,1,3,4),
                                                     im_info,
                                                     gt_tubes_,
                                                     boxes_.permute(0,2,1,3).float(),
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

            f_gt_tubes[idx_s:idx_e] = gt_tubes.type_as(f_gt_tubes)
            tubes_labels[idx_s:idx_e] = rois_label.squeeze(-1).type_as(tubes_labels)

            f_tubes = connect_tubes(f_tubes,tubes.cpu(), p_tubes, pooled_feat, rois_label, step*batch_size)

        ###############################################
        #          Choose Tubes for RCNN\TCN          #
        ###############################################

            # get gt tubes and feats

            video_tubes = create_video_tube(boxes.permute(1,0,2).unsqueeze(0).type_as(clips))

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
        final_video_tubes = torch.zeros(len(f_tubes),max_length,6).cuda()
        final_feats = torch.zeros(len(f_tubes), max_length,256).cuda()

        for i in range(len(f_tubes)):

            seq = f_tubes[i]
            tmp_tube = torch.Tensor(len(seq),6)
            feats = torch.Tensor(len(seq),self.p_feat_size)

            for j in range(len(seq)):
                feats[j] = features[seq[j]].mean(1)
                tmp_tube[j] = p_tubes[seq[j]][1:7]

            final_video_tubes[i,:tmp_tube.size(0)] = tmp_tube
            final_feats[i,:tmp_tube.size(0)] = feats

        return final_video_tubes, final_feats

    def deactivate_grad(self):

        for p in self.parameters() : p.requires_grad=False # deactivate parameters' grad
        # for key, value in dict(self.named_parameters()).items():
        #     print(key, value.requires_grad)

    def load_part_model(self, action_model_path=None, rnn_path=None):


        if action_model_path != None:
            
            act_data = torch.load('./action_net_model.pwf')

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

