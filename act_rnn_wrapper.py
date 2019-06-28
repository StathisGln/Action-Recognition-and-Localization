import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from act_rnn import Act_RNN

class _RNN_wrapper(nn.Module):
    """ region proposal network """
    def __init__(self,  n_inputs, n_neurons, n_outputs):
        super(_RNN_wrapper, self).__init__()

        # self.act_rnn =  Act_RNN(n_inputs, n_neurons, n_outputs)
        self.act_rnn = nn.Sequential(
                # nn.Linear(64*self.sample_duration, 256),
                # nn.ReLU(True),
                # nn.Dropout(0.8),
                # nn.Linear(256,self.n_classes),
                nn.Linear(n_inputs, n_outputs),
                # nn.ReLU(True),
                # nn.Dropout(0.8),
                # nn.Linear(256,self.n_classes),

            )

        self.n_classes = n_outputs

    def forward(self, features, n_tubes, target_lbl):

        batch_size = features.size(0)
        max_tubes = features.size(1)
        prob_out = torch.zeros(batch_size, max_tubes, self.n_classes)
        # print('batch_size :',batch_size)
        # print('features.shape :',features.shape)
        # print('n_tubes.shape :',n_tubes)
        # print('prob_out.shape :',prob_out.shape)
        # print('target_lbl :',target_lbl)
        # print('target_lbl :',target_lbl.shape)
        for b in range(batch_size):
            feats = features[b,:n_tubes]
            # print('feats.shape :',feats.shape)
            # print('feats.shape :',feats.view(n_tubes,-1).shape)
            # for i in range(n_tubes):
            #     if target_lbl[b,i] == -1:
            #         break
            #     # if len_tubes[b,i] == 0:
            #     #     continue
            #     # feat = features[b,i,:len_tubes[b,i]].mean(-1)
            #     prob_out[b,i]  = self.act_rnn(feat)
            prob_out[b, :n_tubes] = self.act_rnn(feats.view(n_tubes, -1))


        # print('prob_out.shape :',prob_out.shape)
        # print('prob_out.view(-1,self.n_classes).shape :',prob_out.view(-1,self.n_classes).shape)
        # print('target_lbl.view(-1).shape :',target_lbl.view(-1).shape)
        if self.training:
            cls_loss = F.cross_entropy(prob_out.view(-1,self.n_classes), target_lbl.view(-1).long().cpu(), ignore_index=-1).cuda()

        return cls_loss
