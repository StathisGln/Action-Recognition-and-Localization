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
        self.n_classes = 22
        self.POOLING_SIZE = 7
        self.p_feats = p_feats
        self.act_rnn =  Act_RNN(n_inputs, n_neurons, n_outputs)

        # self.act_rnn = nn.Sequential(
        #         nn.Linear(self.p_feats*self.sample_duration*self.POOLING_SIZE*self.POOLING_SIZE, self.n_classes),
        #         # nn.ReLU(True),
        #         # nn.Dropout(0.8),
        #         # nn.Linear(256,self.n_classes),
        #         # nn.Linear(n_inputs, n_outputs),
        #         # nn.ReLU(True),
        #         # nn.Dropout(0.8),
        #         # nn.Linear(256,self.n_classes),

        #     )

        self.n_classes = n_outputs

    def forward(self, features, n_tubes, target_lbl, len_tubes):

        batch_size = features.size(0)
        max_tubes = features.size(1)
        prob_out = torch.zeros(batch_size, max_tubes, self.n_classes)

        # print('batch_size :',batch_size)
        # print('features.shape :',features.shape)

        # print('n_tubes.shape :',n_tubes)
        # print('prob_out.shape :',prob_out.shape)
        # print('target_lbl :',target_lbl)
        # print('target_lbl :',target_lbl.shape)
        # print('len_tubes :',len_tubes)
        # print('len_tubes :',len_tubes.shape)
        for b in range(batch_size):
            for i in range(n_tubes[b]):
                for j in range(len_tubes[b,i,0]):

                    feat = features[b,i,:len_tubes[b,i,0]]
                    prob_out[b,i]  = self.act_rnn(feat.view(feat.size(0),-1))

        if self.training:
            cls_loss = F.cross_entropy(prob_out.view(-1,self.n_classes), target_lbl.view(-1).long().cpu(), ignore_index=-1).cuda()

        return cls_loss
