import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from act_rnn import Act_RNN

class _RNN_wrapper(nn.Module):
    """ region proposal network """
    def __init__(self,  n_inputs, n_neurons, n_outputs, p_feats, sample_duration):
        super(_RNN_wrapper, self).__init__()
        self.sample_duration = sample_duration
        self.n_classes = n_outputs
        self.POOLING_SIZE = 7
        self.p_feats = p_feats

        self.act_rnn = nn.Sequential(
                nn.Linear(n_inputs, self.n_classes),
                # nn.ReLU(True),
                # nn.Dropout(0.8),
                # nn.Linear(256,self.n_classes),
                # nn.Linear(n_inputs, n_outputs),
                # nn.ReLU(True),
                # nn.Dropout(0.8),
                # nn.Linear(256,self.n_classes),

            )

        self.n_classes = n_outputs

    def forward(self, features, n_tubes, target_lbl, len_tubes):

        batch_size = features.size(0)
        max_tubes = features.size(1)
        features_ = features[:,:n_tubes[0]]
        loss = 0

        target_lbl = target_lbl[:,:n_tubes[0]].contiguous()
        # print('features_.shape :',features_.shape)
        prob_out = torch.zeros(batch_size, n_tubes[0], self.n_classes)
        for b in range(batch_size):
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
            prob_out[b] = self.act_rnn(features_[b,:,:len_tubes[b,0,0]].mean(1).view(features_[b].size(0),-1))
            # prob_out[b] = self.act_rnn(features_[b,:,:len_tubes[b,0,0]].max(1)[0].view(features_[b].size(0),-1))
            # print('ret.shape :',ret.shape)

        # # print('n_tubes.shape :',n_tubes)
        # print('prob_out.shape :',prob_out.shape)
        # print('target_lbl :',target_lbl)
        # print('target_lbl :',target_lbl.shape)
        # print('len_tubes :',len_tubes)
        # print('len_tubes :',len_tubes.shape)
        if self.training:
            cls_loss = F.cross_entropy(prob_out.view(-1,self.n_classes), target_lbl.view(-1).long().cpu(), ignore_index=-1).cuda()

        return cls_loss
