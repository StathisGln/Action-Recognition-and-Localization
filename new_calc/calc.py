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
                f_actioness_scr, f_temporal_scr, f_temporal_rt,
                overlaps, \
                actioness, tube_rate, N_up, N_down, indx):

        N = N.int().item()  # number of clips
        K = K.int().item()  # rois_per_image
        array_size = array_size.int().item() # number of 
        indx = indx.int().item() # which index we are

        # for next
        next_pos_max_size = array_size * K + K

        next_pos = pos.new(next_pos_max_size, N, 2).zero_().view(-1) -1
        next_pos_indices = pos_indices.new(next_pos_max_size).zero_() -1
        next_actioness = f_actioness_scr.new(next_pos_max_size).zero_()
        next_temporal_scr = f_temporal_scr.new(next_pos_max_size).zero_()
        next_temporal_rt = f_temporal_rt.new(next_pos_max_size).zero_()
        next_N_up = N_up.new(next_pos_max_size).zero_()
        next_N_down = N_down.new(next_pos_max_size).zero_()

        # print('actioness.type() :',actioness.type())
        # print('f_actioness_scores.type() :',f_actioness_scr.type())
        # print('temporal_scr.type() :',tube_rate.type())
        # print('f_temporal_scr.type() :',f_temporal_scr.type())
        # print('next_pos.shap :',next_pos.type())
        # print('next_pos_indices.type() :',next_pos_indices.type())
        # print('next_actioness.type() :',next_actioness.type())
        # print('next_overlaps_scr.type() :',next_temporal_scr.type())
        # print('actioness.shape :',actioness.shape)
        # print('overlaps.shape :',overlaps.shape)
        # print('overlaps.shape :',overlaps.view(-1).shape)
        # print('f_actioness_scores.shape :',f_actioness_scr.shape)
        # print('temporal_scr.shape :',tube_rate.shape)
        # print('f_temporal_scr.shape :',f_temporal_scr.shape)
        # print('next_pos.shap :',next_pos.shape)
        # print('next_pos_indices.shape :',next_pos_indices.shape)
        # print('next_actioness.shape :',next_actioness.shape)
        # print('next_overlaps_scr.shape :',next_temporal_scr.shape)
        # print('next_N_up.shape :',next_N_up.shape)
        # print('tube_rate.shape :',tube_rate.shape)
        # print('tube_rate :',tube_rate)
        c.calc_test_cuda(K, N, self.thresh, array_size, pos.view(-1), pos_indices, N_up, N_down, 
                         f_actioness_scr, f_temporal_scr, f_temporal_rt, overlaps.view(-1),
                         actioness, tube_rate, indx, 
                         next_pos, next_pos_indices, next_actioness,
                         next_temporal_scr, next_temporal_rt, next_N_up, next_N_down)

        next_pos = next_pos.view(next_pos_max_size, N, 2)

        for i in range(next_pos_indices.size(0)):

            if next_pos_indices[i] > -1:
                z = i // K
                for j in range(next_pos_indices[i]):
                    next_pos[i,j,0] = pos[z,j,0]
                    next_pos[i,j,1] = pos[z,j,1]
                next_pos[i,next_pos_indices[i],0] = indx
                next_pos[i,next_pos_indices[i],1] = i % K

        new_tubes_indices = next_pos_indices.gt(-1).nonzero().squeeze()

        # for i in range(next_pos_indices.size(0)):
        #     if next_pos_indices[i] > -1:
        #         print('score :',next_actioness[i].item())
        #         for j in range(next_pos_indices[i]+1):
        #             print('next_pos[i,j,0] :',next_pos[i,j,0].item(), ' ', next_pos[i,j,1].item())
        #         print()

        # print('new_tubes_indices :',new_tubes_indices)
        # if new_tubes_indices.nelement() == 0 :
        #     print('pos :',pos)
        #     print('overlaps :',overlaps)
        #     print('pos_indices :',pos_indices)
        #     print('N_up :',N_up)
        #     print('N_down :',N_down)
        #     print('f_actioness_scr :',f_actioness_scr)
        #     print('f_temporal_scr :',f_temporal_scr)
        #     print('f_temporal_rt :',f_temporal_rt)
        #     print('actioness :',actioness)
        #     print('tube_rate :',tube_rate)
        #     exit(-1)
        next_pos = next_pos[new_tubes_indices]
        next_pos_indices =  next_pos_indices[new_tubes_indices]
        next_N_up = next_N_up[new_tubes_indices]
        next_N_down = next_N_down[new_tubes_indices]

        next_actioness = next_actioness[new_tubes_indices]
        next_temporal_scr = next_temporal_scr[new_tubes_indices]
        next_temporal_rt = next_temporal_rt[new_tubes_indices]

        return next_pos, next_pos_indices, next_N_up, next_N_down, next_actioness,\
            next_temporal_scr, next_temporal_rt
    

    def update_scores(self, final_scores, final_poss, f_scores, pos, pos_indices, actioness, \
                      overlaps_scr, prg_rt_scr):

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
        prg_rt_scr = prg_rt_scr[indices].contiguous()

        ## now time for final scores
        indices = final_scores.ge(self.thresh).nonzero()

        indices = indices.squeeze()
        # print('final_scores.shape :',final_scores.shape)
        # print('final_poss.shape :',final_poss.shape)
        final_scores = final_scores[indices].contiguous()
        final_poss = final_poss[indices].contiguous()

        return final_scores, final_poss, pos, pos_indices, \
            actioness, overlaps_scr, prg_rt_scr, f_scores
        
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
    
