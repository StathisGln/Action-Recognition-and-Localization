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

    def forward(self, N, K, pos, actioness_scr, overlaps_scr): 

        N = N.int().item()    # N : n_clips
        K = K.int().item()    # K : rois_per_image

        pos = pos.view(-1, N*2).contiguous()
        array_size = pos.size(0)

        tube_scores = actioness_scr.new(pos.size(0)).zero_()
        overlaps_scr = overlaps_scr.view(-1).contiguous()
        actioness_scr = actioness_scr.view(-1).contiguous()

        c.calc_test_cuda(K, N, array_size, pos.view(-1),
                         actioness_scr, overlaps_scr, tube_scores)

        top_tubes_score,top_tubes_idx = torch.topk(tube_scores, 500)
        ret_pos = pos[top_tubes_idx]

        return ret_pos,top_tubes_score
        
# if __name__ == '__main__':
    
