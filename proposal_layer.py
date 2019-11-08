from __future__ import absolute_import, print_function
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
import yaml

from conf import conf
from generate_anchors import generate_anchors
# from bbox_transform import bbox_transform_inv, clip_boxes_3d, clip_boxes_batch, bbox_transform_inv_3d
# from nms_3d.py_nms import py_cpu_nms_tubes as nms_cpu
from nms_3d.nms_gpu import nms_gpu
# from nms_8fr_3d.nms_gpu import nms_gpu
# from nms_8fr_3d_cuda9.nms_gpu import nms_gpu
# from nms_4fr_3d.nms_gpu import nms_gpu

# from nms_3d.nms_wrapper import nms

# from soft_nms_3d.py_nms import py_cpu_softnms as nms_gpu

from box_functions import bbox_transform_inv,clip_boxes
import pdb

DEBUG = False

class _ProposalLayer(nn.Module):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def __init__(self, feat_stride, scales, ratios, time_dim,num_anchors):
        super(_ProposalLayer, self).__init__()

        self.sample_duration = time_dim[0]

        print('sample_duration :',self.sample_duration)


        self.time_dim = time_dim
        self.scales = scales
        self.ratios = ratios
        self._feat_stride = feat_stride
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(scales), 
                                         ratios=np.array(ratios))).float()
        self._num_anchors = num_anchors                                 
        

    def forward(self, input):

        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)


        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs

        scores = input[0][:, self._num_anchors:, :, :]
        scores_3_4 = input[1][:, self._num_anchors:, :, :]
        scores_2 = input[2][:, self._num_anchors:, :, :]
        scores_4 = input[3][:, self._num_anchors:, :, :]
        bbox_frame = input[4]
        bbox_frame_3_4 = input[5]
        bbox_frame_2 = input[6]
        bbox_frame_4 = input[7]
        im_info = input[8]
        cfg_key = input[9]

        batch_size = bbox_frame.size(0)


        pre_nms_topN  = conf[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = conf[cfg_key].RPN_POST_NMS_TOP_N

        nms_thresh    = conf[cfg_key].RPN_NMS_THRESH
        min_size      = conf[cfg_key].RPN_MIN_SIZE

        ##################
        # Create anchors #
        ##################

        # print('batch_size :', batch_size)
        feat_time = scores.size(2)
        feat_time_3_4 = scores_3_4.size(2)
        feat_time_2 = scores_2.size(2)
        feat_time_4 = scores_4.size(2)

        feat_height,  feat_width= scores.size(3), scores.size(4) # (batch_size, 512/256, 7,7, 16/8)

        shift_x = np.arange(0, feat_width) * self._feat_stride
        shift_y = np.arange(0, feat_height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                             shift_x.ravel(), shift_y.ravel(),)).transpose())
        shifts = shifts.contiguous().type_as(scores).float()

        A = self._anchors.size(0)
        K = shifts.size(0)

        anchors = self._anchors.view(1, A, 4).type_as(shifts) + shifts.view(K, 1, 4)
        anchors = anchors.view(K * A, 4)

        bboxes = [bbox_frame, bbox_frame_3_4, bbox_frame_2, bbox_frame_4]
        anchors_all = []
        bbox_frame_all = []

        # # for i in range(1,2):
        # print('self.time_dim :',self.time_dim)
        # # x_pos = torch.arange(self.time_dim[0])
        # # combs = torch.arange(self.sample_duration-self.time_dim[0]+1)
        # # print('x_pos :',x_pos)
        # # print('combs :',combs)
        # # x_pos = torch.arange(self.time_dim[1])
        # # combs = torch.arange(self.sample_duration-self.time_dim[1]+1)
        # # x_pos = x_pos.unsqueeze(1).expand(x_pos.size(0),combs.size(0))
        # # combs = combs.unsqueeze(0).expand(x_pos.size(0),combs.size(0))
        # # together = x_pos+combs
        # # print('x_pos :',x_pos)
        # # print('combs :',combs)
        # # print('together :',together.shape)
        # # x_pos = torch.arange(self.
        # print('bboxes[0].shape :',bboxes[0].shape)
        # print('bboxes[1].shape :',bboxes[1].shape)
        # print('bboxes[1][:,:,j].shape    :',bboxes[1][:,:,0].shape)
        # print('bboxes[1][:,:,j].shape    :',bboxes[1][:,:,0].permute(0,2,3,1).shape)
        # print('bboxes[1][:,:,j].shape    :',bboxes[1][:,:,0].permute(0,2,3,1).contiguous().\
        #       view(batch_size, anchors.size(0), self.time_dim[1],4).shape)
        # print('bboxes[1][:,:,j].shape    :',bboxes[1].shape)
        # print('bboxes[1][:,:,j].shape    :',bboxes[1].permute(0,2,3,1).shape)
        # # print('bboxes[1][:,:,j].shape    :',bboxes[1].permute(0,2,3,1).contiguous().\
        # #       view(batch_size, anchors.size(0), self.time_dim[1],4).shape)

        
        # i = 1
        # x_pos = torch.arange(self.time_dim[i])
        # combs = torch.arange(self.sample_duration-self.time_dim[i]+1)
        # combs = combs.unsqueeze(1).expand(combs.size(0),x_pos.size(0))
        # x_pos = x_pos.unsqueeze(0).expand(combs.size(0),x_pos.size(0))
        # together = x_pos+combs
        # final = torch.cat([combs.unsqueeze(-1),together.unsqueeze(-1)],dim=2).view(-1,2)
        # print('final :',final.shape)
        

        # exit(-1)
        # for i in range(1,2):
        # # for i in range(len(self.time_dim)):
            
        #     x_pos = torch.arange(self.time_dim[i])
        #     combs = torch.arange(self.sample_duration-self.time_dim[i]+1)
        #     combs = combs.unsqueeze(1).expand(combs.size(0),x_pos.size(0))
        #     x_pos = x_pos.unsqueeze(0).expand(combs.size(0),x_pos.size(0))
        #     together = x_pos+combs
        #     final = torch.cat([combs.unsqueeze(-1),together.unsqueeze(-1)],dim=2).view(-1,2)
        #     anchors_ = torch.zeros(self.sample_duration-self.time_dim[i]+1,\
        #                            self.sample_duration, anchors.size(0),4).type_as(anchors)
        #     anchors_[final[:,0],final[:,1]] = anchors
        # print('anchors.size(0) :',anchors.size(0))
        # print('anchors.size(0) :',anchors.shape)
        # print('anchors.device :',anchors.device)
        # # anc = torch.zeros((self.sample_duration,anchors.size(0),4))
        # print('anchors_ :',anchors_)
        # print('x_pos :',x_pos)
        # print('combs :',combs)
        # print('x_pos :',x_pos.shape)
        # print('combs :',combs.shape)
        # print('together :',together.shape)
        # print('together :',together)

        # exit(-1)
        
        for i in range(len(self.time_dim)):
            for j in range(0,self.sample_duration-self.time_dim[i]+1):
                anc = torch.zeros((self.sample_duration,anchors.size(0),4))
                bbox =  torch.zeros((batch_size, anchors.size(0),self.sample_duration,4))

                anc[ j:j+self.time_dim[i]] = anchors
                anc = anc.permute(1,0,2)

                t = bboxes[i][:,:,j].permute(0,2,3,1).contiguous().view(batch_size, anchors.size(0), self.time_dim[i],4)
                bbox[:,:,j:j+self.time_dim[i],:] = t
                anchors_all.append(anc)
                bbox_frame_all.append(bbox)

        anchors_all = torch.stack(anchors_all,0).type_as(scores)
        bbox_frame_all = torch.stack(bbox_frame_all,1).type_as(scores)

        anchors_all = anchors_all.view(1, -1, self.sample_duration * 4)
        anchors_all = anchors_all.expand(batch_size, anchors_all.size(1), self.sample_duration * 4)
        bbox_frame_all = bbox_frame_all.view(batch_size, -1, self.sample_duration * 4)

        # print('bbox_frame_all.shape :',bbox_frame_all.shape)
        # print('bbox_frame[0,3000] :',bbox_frame_all[0,17500:18000].cpu().numpy())
        # print('anchors_all.shape :',anchors_all.shape)
        # print('anchors_all[0,2000:3000,:12] :',anchors_all[0,2900:3000,:12].cpu().numpy())
        # print('bbox_frame_all[0,2000:3000,:12] :',bbox_frame_all[0,2900:3000,:12].cpu().numpy())

        # # Same story for the scores:

        scores = scores.permute(0, 2, 3, 4, 1).contiguous()
        scores = scores.view(batch_size, -1)

        scores_3_4 = scores_3_4.permute(0, 2, 3, 4, 1).contiguous()
        scores_3_4 = scores_3_4.view(batch_size, -1)

        scores_2 = scores_2.permute(0, 2, 3, 4, 1).contiguous()
        scores_2 = scores_2.view(batch_size, -1)

        scores_4 = scores_4.permute(0, 2, 3, 4, 1).contiguous()
        scores_4 = scores_4.view(batch_size, -1)
        scores_all = torch.cat([scores, scores_3_4, scores_2, scores_4],1)
        # print('anchors_all[0,:150,:4] :',anchors_all[0,:150,:4].cpu().numpy())
        # print('bbox_frame_all[0,:150,:4] :',bbox_frame_all[0,:150,:4].cpu().numpy())
        # Convert anchors into proposals via bbox transformations
        # print('anchors_all.shape :',anchors_all.shape)
        # print('bbox_frame_all.shape :',bbox_frame_all.shape)
        proposals = bbox_transform_inv(anchors_all.contiguous().view(-1, self.sample_duration*4), \
                                       bbox_frame_all.contiguous().view(-1, self.sample_duration*4), \
                                       (1.0, 1.0, 1.0, 1.0)) # proposals have 441 * time_dim shape

        # 2. clip predicted boxes to image
        ## if any dimension exceeds the dims of the original image, clamp_ them
        proposals = proposals.view(batch_size,-1,self.sample_duration*4)

        proposals = clip_boxes(proposals, im_info, batch_size)

        scores_keep = scores_all
        proposals_keep = proposals

        _, order = torch.sort(scores_all, 1, True)


        output = scores.new(batch_size, post_nms_topN, self.sample_duration*4+2).zero_()
        # print('output.shape :',output.shape)
        for i in range(batch_size):
            if cfg_key == 'TEST':
                # 3. remove predicted boxes with either height or width < threshold
                # (NOTE: convert min_size to input image scale stored in im_info[2])
                proposals_single = proposals_keep[i]
                scores_single = scores_keep[i]
                # print('scores_single.shape :',scores_single.shape)
                # print('proposals_single.shape :',proposals_single.shape)
                # print('order[i].shape :',order[i].shape)
                # exit(-1)
                order_single = order[i]

                if pre_nms_topN > 0 and pre_nms_topN < scores_keep.numel():
                    order_single = order_single[:pre_nms_topN]

                proposals_single = proposals_single[order_single, :]
                scores_single = scores_single[order_single].view(-1,1)

                keep_idx_i = nms_gpu(torch.cat((proposals_single, scores_single), 1),nms_thresh).type_as(scores_single)
                # keep_idx_i =nms_gpu(proposals_single.view(-1, self.sample_duration, 4).contiguous(), \
                #                     scores_single.squeeze(),nms_thresh, method=1).type_as(scores_single)

                keep_idx_i = keep_idx_i.long().view(-1)
                # print('keep_idx_i :',keep_idx_i.cpu().numpy(),keep_idx_i.nelement())
                # exit(-1)

                keep_idx_i = keep_idx_i[:post_nms_topN]

                proposals_single = proposals_single[keep_idx_i, :]
                # print('proposal_single :',proposals_single[:,:4].cpu().numpy())
                # exit(-1)
                scores_single = scores_single[keep_idx_i, :]

                # adding score at the end.
                num_proposal = proposals_single.size(0)
                output[i,:,0] = i
                output[i,:num_proposal,1:-1] = proposals_single
                output[i,:num_proposal,-1] = scores_single.squeeze()

                
            else:

                proposals_single = proposals_keep[i]
                scores_single = scores_keep[i]
                order_single = order[i]
                
                proposals_single = proposals_single[order_single, :]
                scores_single = scores_single[order_single].view(-1,1)

                proposals_single = proposals_single[:post_nms_topN, :]
                scores_single = scores_single[:post_nms_topN]

                # adding score at the end.
                num_proposal = proposals_single.size(0)
                output[i,:num_proposal,0] = i
                output[i,:num_proposal,1:-1] = proposals_single
                output[i,:num_proposal,-1] = scores_single.squeeze()

        return output

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _filter_boxes(self, boxes, min_size):
        """Remove all boxes with any side smaller than min_size."""
        ws = boxes[:, :, 2] - boxes[:, :, 0] + 1
        hs = boxes[:, :, 3] - boxes[:, :, 1] + 1
        keep = ((ws >= min_size.view(-1,1).expand_as(ws)) & (hs >= min_size.view(-1,1).expand_as(hs)))
        return keep

if __name__ == '__main__':

    np.set_printoptions(threshold=np.inf)
    anchor_scales = [1, 2, 4, 8, 16 ]
    anchor_ratios = [0.5, 1, 2]
    feat_stride = [16, ]
    sample_duration = 16
    anchor_duration = [sample_duration,int(sample_duration*3/4), int(sample_duration/2)] #,int(sample_duration/4)] # add 
            
    # define proposal layer
    RPN_proposal = _ProposalLayer(feat_stride, anchor_scales, anchor_ratios, anchor_duration,  len(anchor_scales) * len(anchor_ratios) )

    batch_size = 2
    
    rpn_cls_score = torch.rand([batch_size, 30, 1, 14, 14])
    rpn_cls_score_3_4 = torch.rand([batch_size, 30, 5, 14, 14])
    rpn_cls_score_2 = torch.rand([batch_size, 30, 9, 14, 14])

    # rpn_bbox_pred = torch.arange(14*14*15).view(1,15,1,1,14,14).expand(1,15,64,1,14,14).contiguous()
    # rpn_bbox_pred = rpn_bbox_pred.view(1,-1,1,14,14)
    rpn_bbox_pred = torch.arange(batch_size*14*14).view(batch_size,1,1,14,14).expand(batch_size,15*64,1,14,14).contiguous()
    rpn_bbox_pred_3_4 = torch.arange(batch_size*14*14*5).view(batch_size,1,5,14,14).expand(batch_size,15*48,5,14,14).contiguous()
    # rpn_bbox_pred = torch.arange(960).view(1,960,1).expand(1,960,14*14).contiguous().view(1,960,1,14,14)
    # rpn_bbox_pred = rpn_bbox_pred.view(1,-1,1,14,14)

    print('rpn_bbox_pred[0,0] :',rpn_bbox_pred[0,0])
    # print('rpn_bbox_pred[0,0] :',rpn_bbox_pred[0,1])
    print('rpn_bbox_pred[0,0] :',rpn_bbox_pred[0,1])
    print('rpn_bbox_pred[0,0] :',rpn_bbox_pred.shape)
    # rpn_bbox_pred = torch.rand([batch_size, 960, 1, 14, 14])
    # rpn_bbox_pred_3_4 = torch.rand([batch_size, 720, 5, 14, 14])
    rpn_bbox_pred_2 = torch.rand([batch_size, 480, 9, 14, 14])

    im_info = torch.Tensor([[112,112,16]]*batch_size)
    cfg_key = 'TRAIN'
    ret = RPN_proposal((rpn_cls_score.data, rpn_cls_score_3_4.data, rpn_cls_score_2.data,
                        rpn_bbox_pred.data, rpn_bbox_pred_3_4.data, rpn_bbox_pred_2.data, 
                        im_info, cfg_key))
