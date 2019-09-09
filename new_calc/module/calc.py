import torch
from torch.autograd import Function
from .._ext import calc as c

class Calculator(Function):

    def __init__(self, k, update_thresh, thresh, N, K):
        self.k = k
        self.update_thresh = update_thresh
        self.thresh = thresh
        self.N = N
        self.K = K
    def forward(self, overlaps, scores):

        pos = torch.zeros(K,N,2).int() -1 # initial pos
        array_size = K
        for i in range(array_size):
            pos[i][0][0]=0
            pos[i][0][1]=i

        pos_indices = torch.zeros(K).int()
        actioness = torch.rand(K).float()
        overlaps_scr = torch.rand(K).float()

        final_scores = torch.Tensor().float()
        final_poss   = torch.Tensor().int()
        for i in range(1,N):
            # first find number of combinations
            next_pos_max_size = array_size * K + K

            next_pos = pos.new(next_pos_max_size, N, 2).zero_().view(-1) -1
            next_pos_indices = pos_indices.new(next_pos_max_size).zero_() -1
            next_actioness = actioness.new(next_pos_max_size).zero_()
            next_overlaps_scr = overlaps_scr.new(next_pos_max_size).zero_()
            f_scores = actioness.new(next_pos_max_size).zero_()

            print(' i :',i)
            c.calc_test(K, N, self.thresh, array_size, pos.view(-1), pos_indices, actioness,
                        overlaps_scr, scores, overlaps.view(-1), i,
                        next_pos, next_pos_indices, next_actioness,
                        next_overlaps_scr, f_scores)

            next_pos = next_pos.view(next_pos_max_size, N, 2)
            over_thresh_idx = next_pos_indices.gt(-1).nonzero().squeeze()

            f_poss = next_pos[over_thresh_idx]
            f_scores = f_scores[over_thresh_idx]
            
            final_scores = torch.cat((final_scores, f_scores), dim=0)
            final_poss = torch.cat((final_poss, f_poss), dim=0)

            # pos = f_poss.new(f_poss.shape).contiguous()
            pos = torch.tensor(f_poss)
            pos_indices = torch.tensor(next_pos_indices[over_thresh_idx]).contiguous()
            # for i in range
            actioness = torch.tensor(f_scores).contiguous()
            overlaps_scr = torch.tensor(next_overlaps_scr[over_thresh_idx]).contiguous()
            array_size = pos.size(0)
            # print('type(array_size) :',type(array_size))
            # print('pos.shape :',pos.shape, ' actioness.shape :',actioness.shape, ' overlaps_scr.shape :',overlaps_scr.shape, ' array_size :', array_size)
            print('f_scoresn.shape :',f_scores.shape, ' final_scores.shape :',final_scores.shape)
            while final_scores.size(0) > self.update_thresh:
                final_scores, final_poss = self.update_scores(final_scores, final_poss)
                print('After final_scores.shape :',final_scores.shape, ' final_poss.shape :',final_poss.shape, 'thresh :',self.thresh)
        print('final_scores.shape :',final_scores.shape)

        # for i in range(final_scores.size(0)):
        #     print('score : {}', final_scores[i].item(), end=' ')
        #     for j in range(N):
        #         if final_poss[i,j,0] == -1:
        #             break
        #         print(' ', final_poss[i,j,0].item(), ' ', final_poss[i,j,1].item(), ' |',end='')
        #     print()
        return final_scores

    def update_scores(self, final_scores, final_poss):

        self.thresh = self.thresh + 0.1
        print('final_scores.shape :',final_scores.shape)
        
        indices = final_scores.gt(self.thresh).nonzero()
        print('final_scores.shape :',final_scores.shape)
        print('indices.shape :',indices.shape)
        indices = indices.squeeze()
        final_scores = final_scores[indices].contiguous()
        final_poss = final_poss[indices].contiguous()
        return final_scores, final_poss
        
if __name__ == '__main__':

    scores = torch.rand(5,20)
    overlaps = torch.rand(4,20,20)
    N = 5
    K = 20
    calc = Calculator(100, 1000, 1.5, N, K)
    f = calc(overlaps, scores)
    
