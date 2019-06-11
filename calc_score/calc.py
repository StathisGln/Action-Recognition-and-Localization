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

    def forward(self, overlaps, scores, N, K):

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
            # print(indx)

            # first find number of combinations
            next_pos_max_size = array_size * K + K
            next_pos = pos.new(next_pos_max_size, N, 2).zero_().view(-1) -1
            next_pos_indices = pos_indices.new(next_pos_max_size).zero_() -1
            next_actioness = actioness.new(next_pos_max_size).zero_()
            next_overlaps_scr = overlaps_scr.new(next_pos_max_size).zero_()
            f_scores = actioness.new(next_pos_max_size).zero_()
            # print('----',indx,'----')
            # print(array_size)
            # print('pos.shpae :',pos.shape, ' pos.type()',pos.type() )
            # print('pos_indices.shape :',pos_indices.shape, ' pos_indices.type() :',pos_indices.type())
            # print('actioness.shape :',actioness.shape,' actioness.type() :',actioness.type())
            # print('overlaps_scr.shape :',overlaps_scr.shape, ' overlaps_scr.type() :',overlaps_scr.type())
            # print('scores.shape :',scores.shape, ' scores.type():',scores.type())
            # print('overlaps.shpae :',overlaps.shape, ' overlaps.type() :',overlaps.type())
            # print('next_pos.shape :',next_pos.shape, ' next_pos.type() :',next_pos.type())
            # print('next_pos_indices :',next_pos_indices.shape, ' next_pos_indices.type() :',next_pos_indices.type())
            # print('next_overlaps_scr.shape :',next_overlaps_scr.shape, ' next_overlaps_scr.type() :',next_overlaps_scr.type())
            # print('f_scores.shape :',f_scores.shape, ' f_scores.type() :',f_scores.type())

            # c.calc_test(K, N, self.thresh, array_size, pos.view(-1), pos_indices, actioness,
            #             overlaps_scr, scores, overlaps.view(-1), i,
            #             next_pos, next_pos_indices, next_actioness,
            #             next_overlaps_scr, f_scores)
            c.calc_test_cuda(K, N, self.thresh, array_size, pos.view(-1), pos_indices, actioness,
                             overlaps_scr, scores, overlaps.view(-1), indx,
                             next_pos, next_pos_indices, next_actioness,
                             next_overlaps_scr, f_scores)
            # print('vgike...')
            next_pos = next_pos.view(next_pos_max_size, N, 2)

            # print('overlaps[indx] :',overlaps[indx-1].cpu().numpy())
            # if indx== 3:
            #     for i in range(final_poss.size(0)):
            #         print('i :',i,' score ',final_scores[i].item(),' $$ ', end='')
            #         for j in range(N):
            #             if final_poss[i,j,0] == -1:
            #                 break
            #             print(' ',final_poss[i,j,0].item(),' ',final_poss[i,j,1].item(),'({0:.2f}) | '.format(scores[final_poss[i,j,0],final_poss[i,j,1]]),end='')
            #         print('')
            #     exit(-1)

            for i in range(next_pos_indices.size(0)):
                if next_pos_indices[i] > -1:
                    z = i // K
                    for j in range(next_pos_indices[i]):
                        next_pos[i,j,0] = pos[z,j,0]
                        next_pos[i,j,1] = pos[z,j,1]
                    next_pos[i,next_pos_indices[i],0] = indx
                    next_pos[i,next_pos_indices[i],1] = i % K

            over_thresh_idx = next_pos_indices.gt(-1).nonzero().squeeze()

            f_poss = next_pos[over_thresh_idx]
            f_scores = f_scores[over_thresh_idx]
            if f_scores.nelement() > 0 and final_scores.nelement() > 0:
                try:
                    if f_scores.dim() == 0:
                        f_scores = f_scores.unsqueeze(0)
                        f_poss = f_poss.unsqueeze(0)

                    if final_scores.dim() == 0:
                        final_scores = final_scores.unsqueeze(0)
                        final_poss = final_poss.unsqueeze(0)

                    final_scores = torch.cat((final_scores, f_scores), dim=0)
                except:
                    print('final_scores :',final_scores.cpu().numpy())
                    print('f_scores :',f_scores.cpu().numpy())
                    print('final_scores.dim() :',final_scores.dim())
                    print('f_scores.dim() :',f_scores.dim)
                    print('self.thresh :',self.thresh)
                    print('f_scores.shape :',f_scores.shape)
                    print('final_scores.shape :',final_scores.shape)

                    raise 
                final_poss = torch.cat((final_poss, f_poss), dim=0)
            elif final_scores.nelement() == 0  :
                # print('final_scores.shape :',final_scores.shape)
                # print('f_scores.shape :',f_scores.shape)
                # exit(-1)
                final_scores = f_scores
                final_poss = f_poss
            else:
                print('f_scores.shape :',f_scores)
                print('final_scores.shape :',final_scores.shape)
                print('self.thresh :',self.thresh)
                # exit(-1)
            pos = torch.tensor(f_poss).contiguous()
            pos_indices = torch.tensor(next_pos_indices[over_thresh_idx]).contiguous()

            actioness = torch.tensor(next_actioness[over_thresh_idx]).float().contiguous()
            overlaps_scr = torch.tensor(next_overlaps_scr[over_thresh_idx]).contiguous()
            array_size = pos.size(0)

            while pos.size(0) > self.update_thresh:
                final_scores, final_poss, pos, pos_indices, \
                    actioness, overlaps_scr, array_size, f_scores= self.update_scores(final_scores, final_poss, f_scores, pos, \
                                                                                      pos_indices, actioness, overlaps_scr)
            
            # print('New loopa...\n')
            # for i in range(pos_indices.size(0)):
            #     print('i :',i,' score ',f_scores[i].item(),' next_pos_indices[i] :',pos_indices[i].item(), end='$')
            #     for j in range(pos_indices[i]+1):
            #         print(' ',pos[i,j,0].item(),' ',pos[i,j,1].item(),'({0:.2f}) | '.format(scores[pos[i,j,0],pos[i,j,1]]),end='')
            #     print('')

            ## add the new tubes scores
            try:
                pos= torch.cat((pos,zeros_t),dim=0)
            except:
                print('----------ERROR----------')
                print('pos :',pos)
                print('pos :',pos.dim())
                pos = pos.unsqueeze(0)
                pos= torch.cat((pos,zeros_t),dim=0)
                
                
            pos[-K:,0,0] = ones_t * indx
            pos[-K:,0,1] = offset
            
            # print('pos_indices.shape :',pos_indices.shape)
            # print('pos_indices.shape :',pos_indices.dim())
            # print('indx :',indx)
            # print('scores[indx] :',scores[indx])
            # print('actioness:',actioness)
            # print('K :',K)

            if pos_indices.dim()==0:
                pos_indices = pos_indices.unsqueeze(0)
                actioness = actioness.unsqueeze(0)
                overlaps_scr = overlaps_scr.unsqueeze(0)

            pos_indices = torch.cat((pos_indices,torch.zeros((K)).type_as(pos_indices)),dim=0)
            actioness = torch.cat((actioness, scores[indx]),dim=0)

            overlaps_scr =  torch.cat((overlaps_scr, torch.zeros((K)).type_as(overlaps_scr)),dim=0)
            array_size = array_size + K

            # for i in range(next_pos_indices.size(0)):
            #     print('i :',i,' next_pos_indices[i] :',next_pos_indices[i].item(), end='')
            #     for j in range(next_pos_indices[i]+1):
            #         print(' ',next_pos[i,j,0].item(),' ',next_pos[i,j,1].item(),' | ',end='')
            #     print('')
        # print('New loopa...\n')
        # lens = torch.zeros(final_scores.size(0))
        # for i in range(final_scores.size(0)):
        #     for j in range(N):
        #         if final_poss[i,j,0] == -1:
        #             lens[i]=j
        #             break
        
        # _, indices = torch.sort(lens)
        # for i in indices:
        # for i in range(final_scores.size(0)):
        #     # print('i :',i.item(),' score ',final_scores[i].item(),' $$ ', end='')
        #     print('i :',i,' score ',final_scores[i].item(),' $$ ', end='')
        #     for j in range(N):
        #         if final_poss[i,j,0] == -1:
        #             break
        #         print(' ',final_poss[i,j,0].item(),' ',final_poss[i,j,1].item(),'({0:.2f}) | '.format(scores[final_poss[i,j,0],final_poss[i,j,1]]),end='')
        #     print('')

        # exit(-1)


        return final_scores, final_poss

    def update_scores(self, final_scores, final_poss, f_scores, pos, pos_indices, actioness, \
                      overlaps_scr):

        # print('Updating thresh')
        # self.thresh = self.thresh + 
        ## first update next loop
        _, indices = torch.sort(f_scores,descending=True)
        # print('f_scores :',f_scores.cpu().numpy())
        if self.thresh == f_scores[indices[self.k]].item():
            print('f_scores[:self.k] :',f_scores[:self.k].cpu().numpy())
            self.thresh = self.thresh + 0.001
        else:
            self.thresh = f_scores[indices[self.k]].item()

        indices = indices[:self.k].long()
        # print('self.thresh :',self.thresh)
        # print('indices :',indices.shape)
        print('indices :',indices[:self.k])
        # exit(-1)
        indices = f_scores.ge(self.thresh).nonzero()
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
    
