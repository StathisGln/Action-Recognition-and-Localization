import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from act_rnn import Act_RNN
# from rest_model import RestNet
from rest_model_ucf import RestNet


class _RNN_wrapper(nn.Module):
    """ region proposal network """
    def __init__(self,  n_inputs, n_neurons, n_outputs, p_feats, sample_duration):
        super(_RNN_wrapper, self).__init__()
        self.sample_duration = sample_duration
        self.n_classes = n_outputs
        self.POOLING_SIZE = 7
        self.p_feats = p_feats

        # self.act_rnn = RestNet(self.n_classes, self.sample_duration, pooling_time=20)
        self.act_rnn = RestNet(self.n_classes, self.sample_duration, pooling_time=2)        
        self.n_classes = n_outputs
        self.act_rnn.create_architecture()

    def forward(self, features, n_tubes, target_lbl, len_tubes):

        batch_size = features.size(0)
        max_tubes = features.size(1)
        features_ = features[:,:n_tubes[0]]
        loss = 0

        target_lbl = target_lbl[:,:n_tubes[0]].contiguous()

        prob_out = torch.zeros(batch_size, n_tubes[0], self.n_classes)
        # for b in range(batch_size):
            # print('features_[b].shape :',features_[b].shape)
            # print('features_[b].mean(1).shape :',features_[b].mean(1).shape)
            # print('prob_out.shape :',prob_out.shape)
            # print('prob_out.shape :',prob_out[b].shape)
            # print('len_tubes :',len_tubes)
            # print('len_tubes :',len_tubes[b,0,0])
            # print('features_[b,len_tubes[b,0,0]].shape :',features_[b,:,:len_tubes[b,0,0]].shape)
            # print('features_[b,len_tubes[b,0,0]].shape :',features_[b,:,:len_tubes[b,0,0]].mean(1).shape)
            # print('features_[b,len_tubes[b,0,0]].shape :',features_[b,:,:len_tubes[b,0,0]].mean(1).view(features_[b].size(0),-1).shape)
            # print('features_[b,len_tubes[b,0,0]].shape :',features_[b].size(0))
            # prob_out[b] = self.act_rnn(features_[b,:,:len_tubes[b,0,0]].mean(1).view(features_[b].size(0),-1))
            
            # prob_out[b] = self.act_rnn(features_[b,:,:len_tubes[b,0,0]].max(1)[0].view(features_[b].size(0),-1))
            # print('ret.shape :',ret.shape)

        # # print('n_tubes.shape :',n_tubes)
        # print('prob_out.shape :',prob_out.shape)
        # print('target_lbl :',target_lbl)
        # print('target_lbl :',target_lbl.shape)
        # print('len_tubes :',len_tubes)
        # print('len_tubes :',len_tubes.shape)
        # if self.training:
        #     cls_loss = F.cross_entropy(prob_out.view(-1,self.n_classes), target_lbl.view(-1).long().cpu(), ignore_index=-1).cuda()

        ################
        # Only restNet #
        ################

        _, cls_loss = self.act_rnn(features, len_tubes, target_lbl,n_tubes)
        return cls_loss
