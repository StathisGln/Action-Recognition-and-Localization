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
        # For now a linear classifier only



    def forward(self,n_devs, dataset_folder, vid_names, clips, vid_id, boxes, mode, cls2idx, num_actions, num_frames, h_, w_):
        '''
        TODO describe procedure
        '''
        # print('boxes.shape :',boxes.shape)
        boxes = boxes.squeeze(0).permute(1,0,2).cpu()
        clips = clips.squeeze(0)
        # print('----------Inside----------')
        # print('boxes.shape :',boxes.shape)

        # print('boxes :',boxes.cpu().numpy())
        # print('boxes.type() :',boxes.type())
        ## define a dataloader for the whole video
        boxes = boxes[:num_frames,:num_actions]
        clips = clips[:num_frames]

        # print('clips.shape :',clips.shape)
        # print('clips.shape :',clips.type())
        # print('boxes.shape :',boxes.shape)
        # print('boxes :',boxes.cpu().numpy())
        batch_size = 16 # 
        num_images = 1
        rois_per_image = int(cfg.TRAIN.BATCH_SIZE / num_images) *2 if self.training else 10 *2

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
            act_loss_bbox_16_ = torch.zeros(loops)

        for step, dt in enumerate(data_loader):

            # if step == 1:
            #     break
            # print('step :',step)
            # print('Memory :',torch.cuda.memory_allocated(device=None))
            # print('boxes :',boxes)
            frame_indices, im_info, start_fr = dt
            boxes_ = boxes[frame_indices].cuda()
            clips_ = clips[frame_indices].cuda()

            # print('clips_.shape :',clips_.shape)
            # print('boxes_.permute(0,2,1,3) :',boxes_.shape)
            # print('boxes_.permute(0,2,1,3) :',boxes_.permute(0,2,1,3).shape)

            gt_tubes = create_tube_with_frames(boxes_.permute(0,2,1,3), im_info, self.sample_duration)
            # print('boxes_.shape :',boxes_)
            # print('gt_tubes :',gt_tubes)
            # print('boxes_.shape :',boxes_.shape)
            # print('gt_tubes :',gt_tubes.shape)
            gt_tubes_ = gt_tubes.type_as(clips).cuda()
            im_info = im_info.cuda()
            start_fr = start_fr.cuda()


            # clips = Variable(clips, volatile=True)
            # gt_tubes_ = Variable(gt_tubes_, volatile=True)
            # im_info = Variable(im_info, volatile=True)
            # start_fr = Variable(start_fr, volatile=True)

            tubes,  bbox_pred, pooled_feat, \
            rpn_loss_cls,  rpn_loss_bbox, \
            act_loss_bbox, act_loss_bbox_16, \
            rpn_loss_cls_16, \
            rpn_loss_bbox_16, rois_label, \
            sgl_rois_bbox_pred, sgl_rois_bbox_loss = self.act_net(clips_.permute(0,2,1,3,4),
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
            # print('idx_s :', idx_s, ' idx_e :',idx_e )
            # print('features.shape :',features.shape)
            features[idx_s:idx_e] = pooled_feat
            p_tubes[idx_s:idx_e] = tubes

            if self.training:
                # print('gt_tubes.shape :',gt_tubes.shape)
                # print('f_gt_tubes.element_size() * f_gt_tubes.nelement() :',f_gt_tubes.element_size() * f_gt_tubes.nelement())
                # print('idx_s :',idx_s)
                # print('idx_e :',idx_e)
                # print('f_gt_tubes.shape :',f_gt_tubes.shape)
                # print('f_gt_tubes.device :',f_gt_tubes.device)

                f_gt_tubes[idx_s:idx_e] = gt_tubes.type_as(f_gt_tubes)
                tubes_labels[idx_s:idx_e] = rois_label.squeeze(-1).type_as(tubes_labels)

                rpn_loss_cls_[step] = rpn_loss_cls.mean().unsqueeze(0)
                rpn_loss_bbox_[step] = rpn_loss_bbox.mean().unsqueeze(0)
                act_loss_bbox_[step] = act_loss_bbox.mean().unsqueeze(0)
                rpn_loss_cls_16_[step] = rpn_loss_cls_16.mean().unsqueeze(0)
                rpn_loss_bbox_16_[step] = rpn_loss_bbox_16.mean().unsqueeze(0)
                act_loss_bbox_16_[step] = act_loss_bbox_16.mean().unsqueeze(0)

            # print('----------Out TPN----------')
            # # print('p_tubes.type() :',p_tubes.type())
            # # print('tubes.type() :',tubes.type())
            # print('----------Connect TUBEs----------')

            f_tubes = connect_tubes(f_tubes,tubes.cpu(), p_tubes, pooled_feat, rois_label, step*batch_size)

            # print('----------End Tubes----------')

        ###########################################p####
        #          Choose Tubes for RCNN\TCN          #
        ###############################################
        # print('f_gt_tubes :',f_gt_tubes)
        # print('f_gt_tubes.shape :',f_gt_tubes.shape)
        if self.training:

            f_rpn_loss_cls = rpn_loss_cls_.mean().cuda()
            f_rpn_loss_bbox = rpn_loss_bbox_.mean().cuda()
            f_act_loss_bbox = act_loss_bbox_.mean().cuda()

            ## first get video tube
            # print('boxes.shape :',boxes.shape )
            video_tubes = create_video_tube(boxes.permute(1,0,2).unsqueeze(0).type_as(clips))
            # print('video_tubes :',video_tubes)
            # print('video_tubes.shape :',video_tubes.shape)

            # get gt tubes and feats
            gt_tubes_feats,gt_tubes_list = get_gt_tubes_feats_label(f_tubes, p_tubes, features, tubes_labels, f_gt_tubes)
            bg_tubes = get_tubes_feats_label(f_tubes, p_tubes, features, tubes_labels, video_tubes.cpu())

            gt_tubes_list = [x for x in gt_tubes_list if x != []]
            gt_lbl = torch.zeros(len(gt_tubes_list)).type_as(f_gt_tubes)
        
            for i in torch.arange(len(gt_tubes_list)).long():
                gt_lbl[i] = f_gt_tubes[gt_tubes_list[i][0][0],i,6]
            bg_lbl = torch.zeros((len(bg_tubes))).type_as(f_gt_tubes)
            

            ## concate fb, bg tubes
            f_tubes = gt_tubes_list ## + bg_tubes
            # target_lbl = torch.cat((gt_lbl,bg_lbl),0)
            target_lbl = gt_lbl

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

            final_video_tubes[i] = create_tube_from_tubes(tmp_tube).type_as(boxes)
        
        # ##########################################
        # #           Time for Linear Loss         #
        # ##########################################

        cls_loss = torch.Tensor([0]).cuda()

        # # classification probability
        if self.training:
            cls_loss = F.cross_entropy(prob_out.cpu(), target_lbl.long()).cuda()

        if self.training:
            return final_video_tubes, bbox_pred,  prob_out, f_rpn_loss_cls, f_rpn_loss_bbox, f_act_loss_bbox, cls_loss, 
        else:
            return final_video_tubes, bbox_pred, prob_out

    def deactivate_action_net_grad(self):

        for p in self.act_net.parameters() : p.requires_grad=False
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

            act_rnn_data = torch.load(rnn_path)
            act_rnn.load(act_rnn_data)
            self.act_rnn = act_rnn
        else:
            self.act_rnn =Act_RNN(256,128,self.n_classes)

