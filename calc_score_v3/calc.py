import torch
from torch.autograd import Function
# from _ext import calc as c
from ._ext import calc as c

import sys
import numpy


class Calculator(Function):

    def __init__(self, k, update_thresh, thresh, final_scores_update, final_scores_keep):

        self.k = k   # max number of tubes
        self.update_thresh = update_thresh
        self.thresh = thresh
        self.final_scores_update = final_scores_update
        self.final_scores_keep = final_scores_keep

    def forward(self, N, K, array_size, pos, pos_indices, \
                actioness, overlaps_scr, overlaps, scores, indx, thresh):

        N = N.int().item()
        K = K.int().item()

        array_size = array_size.int().item()
        indx = indx.int().item()

        zeros_t = torch.zeros(K,N,2).int().cuda()-1

        next_pos_max_size = array_size * K + K
        next_pos = pos.new(next_pos_max_size, N, 2).zero_().view(-1) -1
        overthr_pos_indices = pos_indices.new(next_pos_max_size).zero_() -1
        next_actioness = actioness.new(next_pos_max_size).zero_()
        next_overlaps_scr = overlaps_scr.new(next_pos_max_size).zero_()
        f_scores = actioness.new(next_pos_max_size).zero_()

        c.calc_test_cuda(K, N, thresh, array_size, pos.view(-1), pos_indices, actioness,
                         overlaps_scr, scores, overlaps.view(-1), 0,
                         overthr_pos_indices, next_actioness,
                         next_overlaps_scr, f_scores)
        
        next_pos = pos.unsqueeze(1).expand(pos.size(0),K,pos.size(1),2).\
                   contiguous().view(-1,pos.size(1),2)
        update_indices = overthr_pos_indices.gt(0).nonzero().view(-1)

        next_pos_indices = pos_indices.unsqueeze(1).expand(pos_indices.size(0),K).\
                           contiguous().view(-1) + 1

        next_pos_indices = next_pos_indices[update_indices]
        next_pos[update_indices,next_pos_indices.long(),0] = indx
        next_pos[update_indices,next_pos_indices.long(),1] = update_indices.int() % K

        next_pos = next_pos[update_indices]

        f_scores = f_scores[update_indices]
        next_actioness = next_actioness[update_indices]
        next_overlaps_scr = next_overlaps_scr[update_indices]

        return next_pos, next_pos_indices, f_scores, next_actioness, next_overlaps_scr, thresh

    def update_scores(self, final_scores, final_poss, f_scores, pos, pos_indices, actioness, \
                      overlaps_scr, thresh):

        # print('Updating thresh', thresh, final_scores.device, final_scores.shape)  

        # ## first update next loop
        _, indices = torch.topk(f_scores, self.k)

        # if thresh == f_scores[indices[-1]].item():

        #     thresh = self.thresh + 0.001
        # else:
        #     thresh = f_scores[indices[-1]].item()

        pos = pos[indices].contiguous()
        pos_indices = pos_indices[indices].contiguous()
        actioness = actioness[indices].contiguous()
        overlaps_scr = overlaps_scr[indices].contiguous()
        f_scores = f_scores[indices].contiguous()

        ## now time for final scores to update if we save too many tubes
        n_final_tubes = final_scores.size(0)

        if n_final_tubes > self.final_scores_update:

            _,indices = torch.topk(final_scores,min(final_scores.size(0),self.final_scores_keep))

            final_scores = final_scores[indices].contiguous()
            final_poss = final_poss[indices].contiguous()

        return final_scores, final_poss, pos, pos_indices, \
            actioness, overlaps_scr,  f_scores, thresh

    def update_final_scores(self, final_scores, final_poss):

        ## now time for final scores to update if we save too many tubes
        n_final_tubes = final_scores.size(0)

        if n_final_tubes > self.final_scores_update:

            _,indices = torch.topk(final_scores,min(final_scores.size(0),self.final_scores_keep))

            final_scores = final_scores[indices].contiguous()
            final_poss = final_poss[indices].contiguous()

        return final_scores, final_poss
        
if __name__ == '__main__':

    # with open('../overlaps.pwf', 'rb') as fp:
    overlaps = torch.load('../overlaps.pwf')
    scores = torch.load('../scores.pwf')
    print('overlaps.shape :',overlaps.shape)
    print('scores.shape :',scores.shape)
    # scores = torch.rand(5,20).cuda()
    # overlaps = torch.rand(4,20,20).cuda()
    N = torch.Tensor([5])
    K = torch.Tensor([20])
    calc = Calculator(100, 1000, 1.5)
    f = calc(overlaps, scores, N,K)
    
