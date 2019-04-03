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

from bbox_transform import bbox_overlaps_batch, bbox_transform_batch
import pdb

class _Regression_TargetLayer(nn.Module):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def __init__(self,  sample_duration):
        super(_Regression_TargetLayer, self).__init__()
        self.sample_duration = sample_duration
        self.BBOX_NORMALIZE_MEANS = torch.FloatTensor(conf.TRAIN.BBOX_NORMALIZE_MEANS)
        self.BBOX_NORMALIZE_STDS = torch.FloatTensor(conf.TRAIN.BBOX_NORMALIZE_STDS)
        self.BBOX_INSIDE_WEIGHTS = torch.FloatTensor(conf.TRAIN.BBOX_INSIDE_WEIGHTS)

    def forward(self, all_tubes, gt_boxes):

        """
        all_tubes : [b, tubes, 7]
        gt_boxes  : [b,n_actions, n_frames, 5]
        """

        self.BBOX_NORMALIZE_MEANS = self.BBOX_NORMALIZE_MEANS.type_as(gt_boxes)
        self.BBOX_NORMALIZE_STDS  = self.BBOX_NORMALIZE_STDS.type_as(gt_boxes)
        self.BBOX_INSIDE_WEIGHTS  = self.BBOX_INSIDE_WEIGHTS.type_as(gt_boxes)

        batch_size  = gt_boxes.size(0)
        num_actions = gt_boxes.size(1)

        gt_boxes_append = gt_boxes.new(gt_boxes.size()).zero_()
        gt_boxes_append[:,:,:,1:] = gt_boxes[:,:,:,:4]

        all_rois = all_tubes.unsqueeze(-2).expand(all_tubes.size(0),all_tubes.size(1),self.sample_duration,7).contiguous()
        all_rois = all_rois[:,:,:,[0,1,2,4,5]]

        all_rois = torch.cat([all_rois, gt_boxes_append], 1)

        num_images = 1
        rois_per_image = int(conf.TRAIN.BATCH_SIZE / num_images)
        fg_rois_per_image = int(np.round(conf.
                                         TRAIN.FG_FRACTION * rois_per_image))
        fg_rois_per_image = 1 if fg_rois_per_image == 0 else fg_rois_per_image

        labels, rois, bbox_targets, bbox_inside_weights = self._sample_rois_pytorch(
            all_rois, gt_boxes, fg_rois_per_image,
            rois_per_image)

        bbox_outside_weights = (bbox_inside_weights > 0).float()
        return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

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
        sample_dur = bbox_target_data.size(2)
        clss = labels_batch
        bbox_targets = bbox_target_data.new(batch_size, rois_per_image, sample_dur,  4).zero_()
        bbox_inside_weights = bbox_target_data.new(bbox_targets.size()).zero_()
        print('bbox_target_data.shape :',bbox_target_data.shape)
        for b in range(batch_size):
            # assert clss[b].sum() > 0
            if clss[b].sum() == 0:
                continue
            inds = torch.nonzero(clss[b] > 0).view(-1)
            for i in range(inds.numel()):
                ind = inds[i]
                print('ind :',ind)
                print('exwww bbox_target_data.shape :',bbox_target_data[b,ind,:,:].cpu().numpy())
                bbox_targets[b, ind] = bbox_target_data[b, ind]
                bbox_inside_weights[b, ind, :] = self.BBOX_INSIDE_WEIGHTS

        print('self.BBOX_INSIDE_WEIGHTS:',self.BBOX_INSIDE_WEIGHTS)
        print('labels_batch :',labels_batch)
        print('bbox_targets :',bbox_targets)
        print('bbox_inside_weights :',bbox_inside_weights)
        print('bbox_inside_weights :',bbox_inside_weights.shape)

        return bbox_targets, bbox_inside_weights


    def _compute_targets_pytorch(self, ex_rois, gt_rois):
        """Compute bounding-box regression targets for an image."""

        assert ex_rois.size(1) == gt_rois.size(1)
        assert ex_rois.size(-1) == 4
        assert gt_rois.size(-1) == 4

        batch_size = ex_rois.size(0)
        rois_per_image = ex_rois.size(1)
        # print('batch_size :',batch_size)
        # print('rois_per_image :',rois_per_image)
        # print('ex_rois.shape :',ex_rois.shape)
        # print('gt_rois.shape :',gt_rois.shape)
        targets = bbox_transform_batch(ex_rois.view(batch_size,-1,4), gt_rois.view(batch_size,-1,4))

        if conf.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
            targets = ((targets - self.BBOX_NORMALIZE_MEANS.expand_as(targets))
                        / self.BBOX_NORMALIZE_STDS.expand_as(targets))
        targets = targets.view(ex_rois.shape)
        print('targets :', targets[0])
        return targets



    def _sample_rois_pytorch(self, all_rois, gt_boxes, fg_rois_per_image, rois_per_image):
        """Generate a random sample of RoIs comprising foreground and background
        examples.
        """
        # overlaps: (rois x gt_boxes)
        batch_size  = gt_boxes.size(0)
        num_actions = gt_boxes.size(1)

        # all_rois = all_rois.permute(0,2,1,3).contiguous()
        # gt_boxes = gt_boxes.permute(0,2,1,3).contiguous()
        # print('gt_boxes.view(gt_boxes.size(0),-1,5):',gt_boxes.view(gt_boxes.size(0),-1,5).cpu().numpy())
        # print('gt_boxes.view(gt_boxes.size(0),-1,5):',gt_boxes.view(gt_boxes.size(0),-1,5).)
        overlaps = bbox_overlaps_batch(all_rois.view(batch_size,-1,5), gt_boxes.view(batch_size,-1,5))
        # print('overlaps :',overlaps[0].cpu().numpy())
        print('overlaps.shape :',overlaps.shape)
        # exit(-1)
        overlaps = overlaps.view(batch_size,all_rois.size(1),all_rois.size(2),gt_boxes.size(1),gt_boxes.size(2))
        
        overlaps_mean = torch.zeros(batch_size,all_rois.size(1), gt_boxes.size(1))
        for j in range(batch_size):
            for i in range(gt_boxes.size(1)):

                non_empty_frames = gt_boxes[j,i].nonzero()
                if non_empty_frames.nelement() == 0:
                    continue
                non_empty_frames = torch.unique(non_empty_frames[:,0],False)
                tmp_ = torch.zeros(non_empty_frames.size(0),overlaps.size(1))

                for z in range(non_empty_frames.size(0)):
                    pos = non_empty_frames[z].long()
                    tmp_[z] = overlaps[j,:,pos,i,pos]
                tmp_ = tmp_.permute(1,0)
                overlaps_mean[j,:,i]= torch.mean(tmp_,1)

        max_overlaps, gt_assignment = torch.max(overlaps_mean, 2)

        batch_size = overlaps.size(0)
        num_proposal = overlaps.size(1)

        num_boxes_per_img = overlaps.size(2)
        
        offset = torch.arange(0, batch_size)*gt_boxes.size(1)
        offset = offset.view(-1, 1).type_as(gt_assignment) + gt_assignment

        labels = gt_boxes[:,:,:,-1].contiguous().view(-1)[(offset.view(-1),)].view(batch_size, -1)
        print('gt_boxes.shape :',gt_boxes[:,:,:,-1].shape)
        print('gt_boxes[:,:,:,-1] :',gt_boxes[:,:,:,-1].cpu().numpy())
        print('gt_boxes[:,;,:,-1].contiguous().view(-1)[(offset.view(-1),)] :', \
              gt_boxes[:,;,:,-1].contiguous().view(-1)[(offset.view(-1),)])
        print('offset :',offset)
        print('offset.shape :',offset.shape)
        print('labels :',labels.shape)
        print('labels :',labels.cpu().numpy())
        exit(-1)
        sample_dur = all_rois.size(2)
        n_last_dim = all_rois.size(3)

        labels_batch = labels.new(batch_size, rois_per_image).zero_()
        rois_batch  = all_rois.new(batch_size, rois_per_image, sample_dur, n_last_dim).zero_()
        gt_rois_batch = torch.zeros(batch_size, rois_per_image, sample_dur,  n_last_dim).to(rois_batch.device)

        # Guard against the case when an image has fewer than max_fg_rois_per_image
        # foreground RoIs

        for i in range(batch_size):

            gt_boxes_single = gt_boxes[i]
            gt_indexes = gt_boxes_single[...,-1].gt(0).nonzero()

            if gt_indexes.nelement() == 0:
                continue

            gt_indexes = torch.unique(gt_indexes[:,0], False)
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
                print('all_rois :',all_rois.cpu().tolist())
                raise ValueError("bg_num_rois = 0 and fg_num_rois = 0, this should not happen!")

            # The indices that we're selecting (both fg and bg)
            keep_inds = torch.cat([fg_inds, bg_inds], 0)
            # Select sampled values from various arrays:
            labels_batch[i].copy_(labels[i][keep_inds])

            # Clamp labels for the background RoIs to 0
            if fg_rois_per_this_image < rois_per_image:
                labels_batch[i][fg_rois_per_this_image:] = 0
            print('rois_batch.shape:',rois_batch.shape)
            print('keep_inds :',keep_inds)
            print('all_rois.shape :',all_rois.shape)
            print('all_rois[i,keep_inds] :',all_rois[i,keep_inds].cpu().numpy())

            rois_batch[i] = all_rois[i,keep_inds]
            rois_batch[i,:,:,0] = i
            gt_rois_batch[i] = gt_boxes[i][gt_assignment[i][keep_inds]]

        bbox_target_data = self._compute_targets_pytorch(
                rois_batch[:,:,:,1:5], gt_rois_batch[:,:,:,:4])
        print('bbox_target_data.shape before :',bbox_target_data.shape )
        bbox_targets, bbox_inside_weights = \
                self._get_bbox_regression_labels_pytorch(bbox_target_data, labels_batch)
        print('bbox_targets.shape:',bbox_targets.shape)
        print('rois_batch.shape :',rois_batch.shape)
        print('bbox_inside_weights:',bbox_inside_weights.shape)
        return labels_batch, rois_batch, bbox_targets, bbox_inside_weights

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


