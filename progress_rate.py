from __future__ import absolute_import

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Progress_Rate(nn.Module):

    def __init__(self, din, n_classes, sample_duration):

        super(Progress_Rate, self).__init__()

        self.din = din
        self.sample_duration = sample_duration
        self.n_classes = n_classes
        
        self.POOLING_SIZE = 7
        self.ACTIONESS_THRESH = 0.5
        # self.prg_rt = nn.Linear(self.din*self.sample_duration*self.POOLING_SIZE*self.POOLING_SIZE, self.n_classes)
        # self.prg    = nn.Linear(self.din*self.sample_duration*self.POOLING_SIZE*self.POOLING_SIZE, self.n_classes)

        self.prg_rt = nn.Linear(self.din*self.POOLING_SIZE*self.POOLING_SIZE, self.n_classes)
        self.prg    = nn.Linear(self.din*self.POOLING_SIZE*self.POOLING_SIZE, self.n_classes)

        # self.prg_rt = nn.Linear(self.din*self.sample_duration, self.n_classes)
        # self.prg    = nn.Linear(self.din*self.sample_duration, self.n_classes)
        # self.cls    = nn.Linear(self.din*self.sample_duration, self.n_classes)

        self.avg_pool =nn.MaxPool3d((self.sample_duration,1,1), stride=1)
        self.max_num_tubes = 2

    def forward(self, feats, rate, progress, labels, scores):

        if self.training:

            batch_size = rate.size(0)
            rois_per_image = rate.size(1)

            rate_ = torch.zeros(batch_size, self.max_num_tubes, self.n_classes).type_as(feats)
            progress_ = torch.zeros(batch_size, self.max_num_tubes, self.n_classes).type_as(feats)
            feats_ = torch.zeros(batch_size, self.max_num_tubes, self.din,self.sample_duration,self.POOLING_SIZE,self.POOLING_SIZE).type_as(feats)
            labels_ = torch.zeros(batch_size,self.max_num_tubes).type_as(feats)

            print('labels.shape :',labels.shape)
            labels = labels.view(batch_size,-1)
            feats = feats.view(batch_size, rois_per_image, self.din,self.sample_duration,self.POOLING_SIZE,self.POOLING_SIZE)

            for i in range(batch_size):


                fg_inds = labels[i].ne(0).nonzero().view(-1)
                fg_num_rois = fg_inds.numel()

                bg_inds = labels[i].eq(0).nonzero().view(-1)
                bg_num_rois = bg_inds.numel()

                both_ind = (labels[i].eq(0) & scores[i].ge(0.5)).nonzero().view(-1)

                if both_ind.nelement() != 0:
                    for j in both_ind:
                        progress[i,j,0] = 1

                if fg_num_rois > 0 and bg_num_rois > 0:

                    fg_rois_per_this_image = min( self.max_num_tubes, fg_num_rois)
                    rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).long()
                    fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

                    bg_rois_per_this_image = self.max_num_tubes - fg_rois_per_this_image

                    rand_num = np.floor(np.random.rand(bg_rois_per_this_image) * bg_num_rois)
                    rand_num = torch.from_numpy(rand_num).long()
                    bg_inds = bg_inds[rand_num]

                elif fg_num_rois > 0 and bg_num_rois == 0:

                    # sampling fg
                    rand_num = np.floor(np.random.rand(self.max_num_tubes) * fg_num_rois)
                    rand_num = torch.from_numpy(rand_num).long()
                    fg_inds = fg_inds[rand_num]
                    fg_rois_per_this_image = self.max_num_tubes
                    bg_rois_per_this_image = 0

                elif bg_num_rois > 0 and fg_num_rois == 0:
                    
                    # sampling bg
                    rand_num = np.floor(np.random.rand(self.max_num_tubes) * bg_num_rois)
                    rand_num = torch.from_numpy(rand_num).long()
                    
                    bg_inds = bg_inds[rand_num]
                    bg_rois_per_this_image = self.max_num_tubes
                    fg_rois_per_this_image = 0

                # The indices that we're selecting (both fg and bg)
                keep_inds = torch.cat([fg_inds, bg_inds], 0)
                rate_[i] = rate[i,keep_inds]
                progress_[i] = progress[i,keep_inds]
                feats_[i] = feats[i,keep_inds]
                labels_[i] = labels[i,keep_inds]

            feats = feats_


        # predicted_cls  = self.cls(feats.view(-1,self.din*self.sample_duration*self.POOLING_SIZE*self.POOLING_SIZE))

        feats = self.avg_pool(feats.view(-1, self.din,self.sample_duration,self.POOLING_SIZE,self.POOLING_SIZE))
        feats = feats.view(-1,self.din*self.POOLING_SIZE*self.POOLING_SIZE)

        predicted_rate = self.prg_rt(feats)
        predicted_progress = self.prg(feats)

        self.prg_rate_loss = 0
        self.prg_loss = 0

        if self.training:

            rate_ = rate_.view(-1, self.n_classes)
            progress_ = progress_.view(-1, self.n_classes)
            labels_ = labels_.view(-1)

            self.prg_rate_loss = F.mse_loss(predicted_rate, rate_)
            self.prg_loss = F.mse_loss(predicted_progress, progress_)

            return rate, self.prg_rate_loss, progress, self.prg_loss, 
        

        return predicted_rate, self.prg_rate_loss, predicted_progress, self.prg_loss, 
