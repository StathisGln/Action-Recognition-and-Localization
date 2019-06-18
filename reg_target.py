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
import numpy.random as npr
from conf import conf
from bbox_transform import tube_overlaps, bbox_transform_rois
import pdb

class _Regression_TargetLayer(nn.Module):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def __init__(self,  sample_duration):
        super(_Regression_TargetLayer, self).__init__()
        self.sample_duration = sample_duration

        self.BBOX_NORMALIZE_MEANS = torch.FloatTensor(conf.TRAIN.BBOX_NORMALIZE_MEANS).repeat(sample_duration)
        self.BBOX_NORMALIZE_STDS = torch.FloatTensor(conf.TRAIN.BBOX_NORMALIZE_STDS).repeat(sample_duration)
        self.BBOX_INSIDE_WEIGHTS = torch.FloatTensor(conf.TRAIN.BBOX_INSIDE_WEIGHTS).repeat(sample_duration)


    def forward(self, all_tubes, gt_boxes, gt_tubes_all):

        """
        all_tubes : [b, tubes, 7]
        gt_boxes  : [b,n_actions, n_frames, 5]
        """

        self.BBOX_NORMALIZE_MEANS = self.BBOX_NORMALIZE_MEANS.type_as(gt_boxes)
        self.BBOX_NORMALIZE_STDS  = self.BBOX_NORMALIZE_STDS.type_as(gt_boxes)
        self.BBOX_INSIDE_WEIGHTS  = self.BBOX_INSIDE_WEIGHTS.type_as(gt_boxes)

        batch_size = gt_boxes.size(0)
        n_actions = gt_boxes.size(1)
        rois_per_image = all_tubes.size(1)

        gt_tubes = gt_boxes[:,:,:,:4].contiguous().view(batch_size, n_actions, self.sample_duration*4)
        labels = torch.zeros(batch_size,n_actions).type_as(all_tubes)

        for i in range(batch_size):
            for j in range(n_actions):
                lbl =  gt_boxes[i,j,:,4].nonzero().view(-1)
                if lbl.nelement() != 0:
                    labels[i,j] = gt_boxes[i,j,lbl[0],4]



        gt_tubes = torch.cat([gt_tubes, labels.unsqueeze(2)],dim=2)

        # modify all_tubes to 64tubes
        all_rois = torch.zeros(batch_size, rois_per_image, self.sample_duration, 4).type_as(gt_tubes)
        for i in range(batch_size):
            for j in range(rois_per_image):

                start_fr = all_tubes[i,j,3].round().int()
                end_fr =  all_tubes[i,j,6].round().int()
                all_rois[i,j,start_fr:end_fr+1] = all_tubes[i,j,[1,2,4,5]]

        offset = torch.arange(0,batch_size).unsqueeze(1).unsqueeze(1).expand(batch_size,rois_per_image,1).type_as(all_rois)

        all_rois = all_rois.view(batch_size, rois_per_image, self.sample_duration*4).contiguous()
        all_rois = torch.cat([offset, all_rois],dim=2)


        # Include ground-truth boxes in the set of candidate rois
        gt_boxes_append = gt_tubes.new(gt_tubes.size()).zero_()
        gt_boxes_append[:,:,1:] = gt_tubes[:,:,:-1] # in pos 0 is the score

        gt_tubes_append = gt_tubes_all.new(gt_tubes_all.size()).zero_()
        gt_tubes_append[:,:,1:] = gt_tubes_all[:,:,:-1] # in pos 0 is the score

        all_rois = torch.cat([all_rois, gt_boxes_append], 1)
        all_tubes = torch.cat([all_tubes, gt_tubes_append], 1)

        num_images = 1
        rois_per_image = int(conf.TRAIN.BATCH_SIZE / num_images)
        fg_rois_per_image = int(np.round(conf.TRAIN.FG_FRACTION * rois_per_image))
        fg_rois_per_image = 1 if fg_rois_per_image == 0 else fg_rois_per_image

        labels, rois, tubes_batch, bbox_targets, bbox_inside_weights, = self._sample_rois_pytorch(
            all_rois, all_tubes, gt_tubes, fg_rois_per_image,
            rois_per_image)

        bbox_outside_weights = (bbox_inside_weights > 0).float()
        return rois, tubes_batch, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _get_bbox_regression_labels_pytorch(self, bbox_target_data, labels_batch):
        """Bounding-box regression targets (bbox_target_data) are stored in a
        compact form b x N x (class, tx, ty, tz, tw, th, tt)

        This function expands those targets into the 4-of-4*K representation used
        by the network (i.e. only one class has non-zero targets).

        Returns:
            bbox_target (ndarray): b x N x 4K blob of regression targets
            bbox_inside_weights (ndarray): b x N x 4K blob of loss weights
        """
        batch_size = labels_batch.size(0)
        rois_per_image = labels_batch.size(1)
        clss = labels_batch
        bbox_targets = bbox_target_data.new(batch_size, rois_per_image, 4*self.sample_duration).zero_()
        bbox_inside_weights = bbox_target_data.new(bbox_targets.size()).zero_()
        for b in range(batch_size):

            # assert clss[b].sum() > 0
            if clss[b].sum() == 0:
                continue
            inds = torch.nonzero(clss[b] > 0).view(-1)
            for i in range(inds.numel()):
                ind = inds[i]
                bbox_targets[b, ind, :] = bbox_target_data[b, ind, :]
                bbox_inside_weights[b, ind, :] = self.BBOX_INSIDE_WEIGHTS

        return bbox_targets, bbox_inside_weights


    def _compute_targets_pytorch(self, ex_rois, gt_rois):
        """Compute bounding-box regression targets for an image."""

        assert ex_rois.size(1) == gt_rois.size(1)
        assert ex_rois.size(2) == 4*self.sample_duration
        assert gt_rois.size(2) == 4*self.sample_duration

        # print('ex_rois :',ex_rois.cpu().numpy())
        # print('gt_rois :',gt_rois.cpu().numpy())
        batch_size = ex_rois.size(0)
        rois_per_image = ex_rois.size(1)
        # print('ex_rois.shape :',ex_rois.shape)
        # print('gt_rois.shape :',gt_rois.shape)
        targets = bbox_transform_rois(ex_rois.contiguous().view(-1,self.sample_duration*4).contiguous(),\
                                      gt_rois.contiguous().view(-1,self.sample_duration*4).contiguous())
        targets = targets.view(batch_size, rois_per_image, self.sample_duration*4).contiguous()

        if conf.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
            targets = ((targets - self.BBOX_NORMALIZE_MEANS.expand_as(targets))
                        / self.BBOX_NORMALIZE_STDS.expand_as(targets))
        return targets


    def _sample_rois_pytorch(self, all_rois, all_tubes, gt_boxes, fg_rois_per_image, rois_per_image):
        """Generate a random sample of RoIs comprising foreground and background
        examples.
        """
        # overlaps: (rois x gt_boxes)

        batch_size = all_rois.size(0)
        overlaps = []
        for i in range(batch_size):
            overlaps.append(tube_overlaps(all_rois[i,:,1:].contiguous().view(-1,self.sample_duration*4),\
                                          gt_boxes[i,:,:-1].contiguous().view(-1,self.sample_duration*4)))
        overlaps = torch.stack(overlaps)
        overlaps = overlaps.view(batch_size,all_rois.size(1),gt_boxes.size(1))

        max_overlaps, gt_assignment = torch.max(overlaps, 2)

        num_proposal = overlaps.size(1)

        num_boxes_per_img = overlaps.size(2)
        
        offset = torch.arange(0, batch_size)*gt_boxes.size(1)
        offset = offset.view(-1, 1).type_as(gt_assignment) + gt_assignment

        labels = gt_boxes[:,:,-1].contiguous().view(-1)[(offset.view(-1),)].view(batch_size, -1)

        n_last_dim = all_rois.size(2)

        labels_batch = labels.new(batch_size, rois_per_image).zero_()
        rois_batch  = all_rois.new(batch_size, rois_per_image, n_last_dim).zero_()
        tubes_batch = all_tubes.new(batch_size, rois_per_image, 7).zero_()
        gt_rois_batch = torch.zeros(batch_size, rois_per_image, n_last_dim).to(rois_batch.device)

        # Guard against the case when an image has fewer than max_fg_rois_per_image
        # foreground RoIs
        for i in range(batch_size):

            gt_boxes_single = gt_boxes[i]
            gt_indexes = gt_boxes_single[...,-1].gt(0).nonzero().view(-1)

            gt_boxes_single = gt_boxes_single[gt_indexes]

            if gt_boxes_single.byte().any() == 0:
                continue

            max_overlaps_single =max_overlaps[i]

            fg_inds = torch.nonzero(max_overlaps_single >= conf.TRAIN.FG_THRESH).view(-1)
            fg_num_rois = fg_inds.numel()

            # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
            bg_inds = torch.nonzero((max_overlaps_single < conf.TRAIN.BG_THRESH_HI) &
                                    (max_overlaps_single >= conf.TRAIN.BG_THRESH_LO)).view(-1)
            bg_num_rois = bg_inds.numel()

            if fg_num_rois > 0 and bg_num_rois > 0:
                # sampling fg
                fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)
                rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(gt_boxes).long()
                fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]
                # sampling bg
                bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image

                rand_num = np.floor(np.random.rand(bg_rois_per_this_image) * bg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                bg_inds = bg_inds[rand_num]

            elif fg_num_rois > 0 and bg_num_rois == 0:

                # sampling fg
                rand_num = np.floor(np.random.rand(rois_per_image) * fg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                fg_inds = fg_inds[rand_num]
                fg_rois_per_this_image = rois_per_image
                bg_rois_per_this_image = 0

            elif bg_num_rois > 0 and fg_num_rois == 0:

                # sampling bg
                rand_num = np.floor(np.random.rand(rois_per_image) * bg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()

                bg_inds = bg_inds[rand_num]
                bg_rois_per_this_image = rois_per_image
                fg_rois_per_this_image = 0
            else:
                print('gt_boxes :',gt_boxes)
                print('i :',i)
                print('gt_boxes_single :',gt_boxes_single)
                print('max_overlaps_single :',max_overlaps_single.cpu().tolist())
                print('num_rois_pre :',num_rois_pre)
                print('all_rois :',all_rois.cpu().tolist())
                raise ValueError("bg_num_rois = 0 and fg_num_rois = 0, this should not happen!")

            # The indices that we're selecting (both fg and bg)
            keep_inds = torch.cat([fg_inds, bg_inds], 0)
            # Select sampled values from various arrays:
            labels_batch[i].copy_(labels[i][keep_inds])

            # Clamp labels for the background RoIs to 0
            if fg_rois_per_this_image < rois_per_image:
                labels_batch[i][fg_rois_per_this_image:] = 0
            rois_batch[i] = all_rois[i,keep_inds]
            rois_batch[i,:,0] = i
            tubes_batch[i] = all_tubes[i,keep_inds]
            tubes_batch[i,:,0] = i

            gt_rois_batch[i] = gt_boxes[i][gt_assignment[i][keep_inds]]

        bbox_target_data = self._compute_targets_pytorch(
                rois_batch[:,:,1:], gt_rois_batch[:,:,:-1])
        
        bbox_targets, bbox_inside_weights = \
                self._get_bbox_regression_labels_pytorch(bbox_target_data, labels_batch)

        return labels_batch, rois_batch, tubes_batch, bbox_targets, bbox_inside_weights

if __name__ == '__main__':

    t = torch.arange(0,280).view(4,5,2,7).unsqueeze(-1).expand(4,5,2,7,7).float()
    tubes_ = torch.Tensor([[[ 0.,  4.,  3.,  6.,  8.,  3., 10.],
                            [ 0.,  3.,  6.,  5.,  6.,  5.,  9.],
                            [ 0.,  3.,  8., 10.,  4.,  9., 15.],
                            [ 0., 10.,  7.,  5.,  13.,  9.,  8.]]]).expand(4,4,7)
    gt_rois = torch.Tensor([[[[ 10, 11, 22, 23, 10], [ 0,  0,  0,  0, -1]],
                            [[ 22, 25, 32, 55, 11], [32, 12, 78, 32, 10]],
                             [[  0,  0,  0,  0, -1], [53, 42, 98, 60, 10]]]]).expand(4,3,2,5)

    # print('gt_rois :',gt_rois)
    # print('t :',t.shape)
    # print('tubes_.shape :',tubes_.shape)
    # print('gt_rois :',gt_rois.shape)
    reg_target = _Regression_TargetLayer(2)
    rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = reg_target(tubes_, gt_rois, torch.Tensor(3))
    print('rois.shape :',rois.shape)
    print('labels.shape :',labels.shape)
    print('bbox_inside_weights.shape :',bbox_inside_weights.shape)
    print('bbox_outside_weights.shape :',bbox_outside_weights.shape)

