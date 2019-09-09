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
        self.n_ret_tubes = 500

    def forward(self, N, K, actioness_scr, overlaps_scr): 

        N = N.int().item()    # N : n_clips
        K = K.int().item()    # K : rois_per_image
        array_size = K ** N   # array_size = rois_per_image ^ n_clips

        tube_scores = actioness_scr.new(array_size).zero_()
        overlaps_scr = overlaps_scr.view(-1).contiguous()
        actioness_scr = actioness_scr.view(-1).contiguous()
        
        c.calc_test_cuda(K, N, array_size, \
                         actioness_scr, overlaps_scr, tube_scores)

        top_tubes_score,top_tubes_idx = torch.topk(tube_scores, self.n_ret_tubes)
        ret_pos = actioness_scr.new(500,N,2).int()

        for i in range(self.n_ret_tubes):

            tmp_K = 1

            for j in range(N-1,-1,-1):
                ret_pos[i,j,0]=j
                ret_pos[i,j,1]=(top_tubes_idx[i]//tmp_K)%K
                tmp_K *= K

        return ret_pos,top_tubes_score
        
# if __name__ == '__main__':
    
