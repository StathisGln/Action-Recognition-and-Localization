import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from act_rnn import Act_RNN

class _RNN_wrapper(nn.Module):
    """ region proposal network """
    def __init__(self,  n_inputs, n_neurons, n_outputs):
        super(_RNN_wrapper, self).__init__()

        self.act_rnn =  Act_RNN(n_inputs, n_neurons, n_outputs)
        self.n_classes = n_outputs
    def forward(self, features, len_tubes, target_lbl):

        batch_size = features.size(0)
        n_tubes = features.size(1)
        prob_out = torch.zeros(batch_size, n_tubes, self.n_classes)

        for b in range(batch_size):
            for i in range(n_tubes):
                if len_tubes[b,i] == 0:
                    continue
                feat = features[b,i,:len_tubes[b,i]].mean(-1)
                prob_out[b,i]  = self.act_rnn(feat)

        if self.training:
            cls_loss = F.cross_entropy(prob_out.view(-1,self.n_classes), target_lbl.view(-1).long().cpu(), ignore_index=-1).cuda()

        return cls_loss
