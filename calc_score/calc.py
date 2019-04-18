import torch
from torch.autograd import Function
from ._ext import calc as c
import sys
import numpy

class Calculator(Function):

    def __init__(self, k, update_thresh, thresh, ):
        self.k = k
        self.update_thresh = update_thresh
        self._thresh = thresh
        self.training = False
        
    def forward(self, overlaps, scores, N, K):

        self.thresh = self._thresh

        N = N.int().item()
        K = K.int().item()
        pos = torch.zeros(K,N,2).int().cuda() -1 # initial pos
        array_size = K

        # for later
        offset = torch.arange(0,K).int().cuda()
        ones_t = torch.ones(K).int().cuda()
        zeros_t = torch.zeros(K,N,2).int().cuda()-1

        pos[:,0,0] = 0
        pos[:,0,1] = offset.contiguous()

        pos_indices = torch.zeros(K).int().cuda()
        actioness = scores[0].float().cuda()
        overlaps_scr = torch.zeros(K).float().cuda()
        
        final_scores = torch.Tensor().float().cuda()
        final_poss   = torch.Tensor().int().cuda()

        for indx in range(1,N):

            # print('indx  :',indx )
            # first find number of combinations
            next_pos_max_size = array_size * K + K
            next_pos = pos.new(next_pos_max_size, N, 2).zero_().view(-1) -1
            next_pos_indices = pos_indices.new(next_pos_max_size).zero_() -1
            next_actioness = actioness.new(next_pos_max_size).zero_()
            next_overlaps_scr = overlaps_scr.new(next_pos_max_size).zero_()
            f_scores = actioness.new(next_pos_max_size).zero_()

            c.calc_test_cuda(K, N, self.thresh, array_size, pos.view(-1), pos_indices, actioness,
                             overlaps_scr, scores, overlaps.view(-1), indx,
                             next_pos, next_pos_indices, next_actioness,
                             next_overlaps_scr, f_scores)

            next_pos = next_pos.view(next_pos_max_size, N, 2)

            for i in range(next_pos_indices.size(0)):
                if next_pos_indices[i] > -1:
                    z = i // K
                    for j in range(next_pos_indices[i]):
                        next_pos[i,j,0] = pos[z,j,0]
                        next_pos[i,j,1] = pos[z,j,1]
                    next_pos[i,next_pos_indices[i],0] = indx
                    next_pos[i,next_pos_indices[i],1] = i % K
            
            # # if training mode, then remove gt tubes in order to find only background tubes
            # if self.training:
            #     over_thresh_idx = f_scores.ne(2).nonzero().squeeze()

            #     next_pos = next_pos_indices[over_thresh_idx]
            #     next_pos_indices = next_pos_indices[over_thresh_idx]
            #     next_actioness = next_actioness[over_thresh_idx]
            #     next_overlaps_scr = next_overlaps_scr[over_thresh_idx]

            #     f_poss = next_pos
            #     f_scores = f_scores[over_thresh_idx]
            #     if f_scores.nelement() != 0:
            #         if f_scores.nelement() == 1:
            #             f_scores = f_scores.unsqueeze(0)
            #             f_poss = f_poss.unsqueeze(0)

            #     final_scores = torch.cat((final_scores, f_scores), dim=0)
            #     final_poss = torch.cat((final_poss, f_poss), dim=0)

            #     pos = torch.tensor(f_poss).contiguous()
            #     pos_indices = torch.tensor(next_pos_indices[over_thresh_idx]).contiguous()

            #     actioness = torch.tensor(next_actioness[over_thresh_idx]).float().contiguous()
            #     overlaps_scr = torch.tensor(next_overlaps_scr[over_thresh_idx]).contiguous()
            #     array_size = pos.size(0)

            #     if pos_indices.dim() == 0:
            #         pos_indices = pos_indices.unsqueeze(0)
            #         actioness = actioness.unsqueeze(0)
            #         overlaps_scr = overlaps_scr.unsqueeze(0)

            # regular proc
            over_thresh_idx = next_pos_indices.gt(-1).nonzero().squeeze()
            f_poss = next_pos[over_thresh_idx]
            f_scores = f_scores[over_thresh_idx]
            if f_scores.nelement() != 0:
                if f_scores.nelement() == 1:

                    f_scores = f_scores.unsqueeze(0)
                    f_poss = f_poss.unsqueeze(0)

                final_scores = torch.cat((final_scores, f_scores), dim=0)
                final_poss = torch.cat((final_poss, f_poss), dim=0)

            pos = torch.tensor(f_poss).contiguous()
            pos_indices = torch.tensor(next_pos_indices[over_thresh_idx]).contiguous()

            actioness = torch.tensor(next_actioness[over_thresh_idx]).float().contiguous()
            overlaps_scr = torch.tensor(next_overlaps_scr[over_thresh_idx]).contiguous()
            array_size = pos.size(0)

            if pos_indices.dim() == 0:
                pos_indices = pos_indices.unsqueeze(0)
                actioness = actioness.unsqueeze(0)
                overlaps_scr = overlaps_scr.unsqueeze(0)
            
            while pos.size(0) > self.update_thresh:
                final_scores, final_poss, pos, pos_indices, \
                    actioness, overlaps_scr, array_size, f_scores= self.update_scores(final_scores, final_poss, f_scores, pos, \
                                                                                      pos_indices, actioness, overlaps_scr)
                if self.training and self.thresh >= 2: # means no background
                    print('epaeeeeeee')
                    return torch.Tensor(), torch.Tensor()


            ## add the new tubes scores

            pos= torch.cat((pos,zeros_t),dim=0)
            
            pos[-K:,0,0] = ones_t * indx
            pos[-K:,0,1] = offset

            pos_indices = torch.cat((pos_indices,torch.zeros((K)).type_as(pos_indices)),dim=0)
            actioness = torch.cat((actioness, scores[indx]),dim=0)

            overlaps_scr =  torch.cat((overlaps_scr, torch.zeros((K)).type_as(overlaps_scr)),dim=0)
            array_size = array_size + K

        return final_scores, final_poss

    def update_scores(self, final_scores, final_poss, f_scores, pos, pos_indices, actioness, \
                      overlaps_scr):

        print('Updating thresh ', self.thresh)
        # self.thresh = self.thresh + 
        _, indices = torch.sort(f_scores,descending=True)
        # print('f_scores :',f_scores.cpu().numpy())
        # if we have the same thresh
        if self.thresh == f_scores[indices[self.k]].item():

            q = self.k +1
            while f_scores[indices[q]] == self.thresh and q + 1 < f_scores.size(0):
                print('q :', q,  ' thresh :',self.thresh, ' f_scores[indices[q]] :', f_scores[indices[q]])
                q += 1
            if q + 1 > f_scores.size(0):
                # extreme case, random pick
                indices = torch.rand(self.k) * f_scores.size(0)
                self.thresh = f_scores[indices[q]]
        else:
            self.thresh = f_scores[indices[self.k]].item()
            indices = indices[:self.k].long()

        indices = indices.squeeze()
        
        pos = pos[indices].contiguous()
        f_scores = f_scores[indices].contiguous()
        pos_indices = pos_indices[indices].contiguous()
        actioness = actioness[indices].contiguous()
        overlaps_scr = overlaps_scr[indices].contiguous()
        array_size = pos.size(0)
        
        ## now time for final scores
        indices = final_scores.ge(self.thresh).nonzero()

        indices = indices.squeeze()
        final_scores = final_scores[indices].contiguous()
        final_poss = final_poss[indices].contiguous()
        return final_scores, final_poss, pos, pos_indices, \
            actioness, overlaps_scr, array_size, f_scores
        
if __name__ == '__main__':

    scores = torch.rand(5,20).cuda()
    overlaps = torch.rand(4,20,20).cuda()
    N = torch.Tensor([5])
    K = torch.Tensor([20])
    calc = Calculator(100, 1000, 1.5)
    f = calc(overlaps, scores, N,K)
    
