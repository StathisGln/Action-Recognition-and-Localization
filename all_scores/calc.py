import torch
from torch.autograd import Function
# from _ext import calc as c
from ._ext import calc as c

import sys
import numpy
import time
class Calculator(Function):

    def __init__(self, k, update_thresh, thresh, ):
        self.k = k
        self.update_thresh = update_thresh
        self.thresh = thresh

        # self.n_ret_tubes = 5
        self.n_ret_tubes = 500

    def forward(self, N, K, actioness_scr, overlaps_scr): 

        N = N.int().item()    # N : n_clips
        K = K.int().item()    # K : rois_per_image
        array_size = K ** N   # array_size = rois_per_image ^ n_clips

        tube_scores = actioness_scr.new(array_size).zero_()
        overlaps_scr = overlaps_scr.view(-1).contiguous()
        actioness_scr = actioness_scr.view(-1).contiguous()

        cuda_start = time.time()
        c.calc_test_cuda(K, N, array_size, \
                         actioness_scr, overlaps_scr, tube_scores)
        cuda_end = time.time()

        
        if array_size < self.n_ret_tubes:
            top_tubes_score,top_tubes_idx = torch.topk(tube_scores, array_size)
            first_dim = array_size
        else:
            top_tubes_score,top_tubes_idx = torch.topk(tube_scores, self.n_ret_tubes)
            first_dim = self.n_ret_tubes

        sort_time = time.time()

        ret_pos = actioness_scr.new(self.n_ret_tubes,N,2).int().zero_()
        ret_scores = actioness_scr.new(self.n_ret_tubes).zero_()
        r_pos =  actioness_scr.new(first_dim,N,2).int().zero_()
        
        rrr = torch.Tensor([ K ** i for i in range(N)]).unsqueeze(0).\
              expand(first_dim, N).type_as(ret_pos)

        tmp_idx = top_tubes_idx.view(-1,1).contiguous().\
                  expand(first_dim,N).type_as(rrr)

        ## Mporei kai na thelei anapoda to rrr
        ## twra einai [1,16,16**2,...]
 
        r_pos[:,:,0] = torch.arange(0,N)
        r_pos[:,:,1] = (tmp_idx // kkk) % K
        ret_pos[:first_dim] =  r_pos

        trans_time = time.time()

        return ret_pos,ret_scores
        
# if __name__ == '__main__':
    
