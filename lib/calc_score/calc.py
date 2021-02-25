import torch
from torch.autograd import Function
# from _ext import calc as c
from ._ext import calc as c

import sys
import numpy

class Calculator(Function):

    def __init__(self, k, update_thresh, thresh, ):
        self.k = k
        self.update_thresh = update_thresh
        self.thresh = thresh

    def forward(self, N, K, array_size, pos, pos_indices, \
                actioness, overlaps_scr, overlaps, scores, indx):
        # print('self.thresh :',self.thresh)
        N = N.int().item()
        K = K.int().item()
        array_size = array_size.int().item()
        indx = indx.int().item()

        zeros_t = torch.zeros(K,N,2).int().cuda()-1
        # for next
        next_pos_max_size = array_size * K + K
        next_pos = pos.new(next_pos_max_size, N, 2).zero_().view(-1) -1
        next_pos_indices = pos_indices.new(next_pos_max_size).zero_() -1
        next_actioness = actioness.new(next_pos_max_size).zero_()
        next_overlaps_scr = overlaps_scr.new(next_pos_max_size).zero_()
        f_scores = actioness.new(next_pos_max_size).zero_()
        # print('overlaps.shape :',overlaps.type())
        # print('pos.type() :',pos.type())
        # print('pos_indices.type() :',pos_indices.type())
        # print('actioness.type() :',actioness.type())
        # print('overlaps_scr.type() :',overlaps_scr.type())
        # print('scores.type() :',scores.type())
        # print('next_pos.type() :',next_pos.type())
        # print('next_pos_indices.type() :',next_pos_indices.type())
        # print('next_actioness.type() :',next_actioness.type())
        # print('next_overlaps_scr.type() :',next_overlaps_scr.type())
        # print('f_scores.type() :',f_scores.type())
        # print('overlaps.shape :',overlaps.shape)
        # print('pos.type() :',pos.shape)
        # print('pos_indices.type() :',pos_indices.shape)
        # print('actioness.type() :',actioness.shape)
        # print('overlaps_scr.type() :',overlaps_scr.shape)
        # print('scores.type() :',scores.shape)

        c.calc_test_cuda(K, N, self.thresh, array_size, pos.view(-1), pos_indices, actioness,
                         overlaps_scr, scores, overlaps.view(-1), 0,
                         next_pos, next_pos_indices, next_actioness,
                         next_overlaps_scr, f_scores)

        # print('next_overlaps_scr :',next_overlaps_scr)
        next_pos = next_pos.view(next_pos_max_size, N, 2)
        for i in range(next_pos_indices.size(0)):
            if next_pos_indices[i] > -1:
                z = i // K
                for j in range(next_pos_indices[i]):
                    next_pos[i,j,0] = pos[z,j,0]
                    next_pos[i,j,1] = pos[z,j,1]
                next_pos[i,next_pos_indices[i],0] = indx
                next_pos[i,next_pos_indices[i],1] = i % K

        over_thresh_idx = next_pos_indices.gt(-1).nonzero().squeeze()

        # if over_thresh_idx.nelement() == 0:
        #     print('Empty this loop..., self thresh: ', self.thresh, ' pos.shape :',pos.shape)

        next_pos = next_pos[over_thresh_idx]
        next_pos_indices =  next_pos_indices[over_thresh_idx]

        f_scores = f_scores[over_thresh_idx]
        next_actioness = next_actioness[over_thresh_idx]
        next_overlaps_scr = next_overlaps_scr[over_thresh_idx]

        return next_pos, next_pos_indices, f_scores, next_actioness, next_overlaps_scr

    def update_scores(self, final_scores, final_poss, f_scores, pos, pos_indices, actioness, \
                      overlaps_scr):

        # print('Updating thresh')
        ## first update next loop

        _, indices = torch.sort(f_scores,descending=True)

        if self.thresh == f_scores[indices[self.k]].item():

            print('f_scores[:self.k] :',f_scores[:self.k].cpu().numpy())
            self.thresh = self.thresh + 0.001
        else:
            self.thresh = f_scores[indices[self.k]].item()

        indices = indices[:self.k]
        self.thresh = f_scores[self.k].item()

        pos = pos[indices].contiguous()
        pos_indices = pos_indices[indices].contiguous()
        actioness = actioness[indices].contiguous()
        overlaps_scr = overlaps_scr[indices].contiguous()
        f_scores = f_scores[indices].contiguous()

        ## now time for final scores
        indices = final_scores.ge(self.thresh).nonzero()

        indices = indices.squeeze()
        # print('final_scores.shape :',final_scores.shape)
        # print('final_poss.shape :',final_poss.shape)
        final_scores = final_scores[indices].contiguous()
        final_poss = final_poss[indices].contiguous()

        return final_scores, final_poss, pos, pos_indices, \
            actioness, overlaps_scr,  f_scores
        
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
    
