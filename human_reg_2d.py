from __future__ import absolute_import, print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from config import cfg
from proposal_layer import _ProposalLayer
from anchor_target_layer_mine import _AnchorTargetLayer
# from reg_target import _Regression_TargetLayer
from roi_align.modules.roi_align  import RoIAlignAvg, RoIAlign
from proposal_target_layer_cascade_original import _ProposalTargetLayer as _Regression_TargetLayer
from net_utils import _smooth_l1_loss, from_tubes_to_rois


import numpy as np

class _Regression_Layer(nn.Module):
    """
    This class gets as input, tubes and feats and finds the regression
    """
    def __init__(self, din, sample_duration):

        super(_Regression_Layer, self).__init__()
        self.din = din
        self.sample_duration = sample_duration
        self.pooling_size = 7
        self.spatial_scale = 1.0/16

        # self.Conv = nn.Conv3d(self.din, din, 1, stride=1, padding=0, bias=True)
        self.Conv_list = nn.ModuleList([])
        self.head_to_tail_list = nn.ModuleList([])
        self.bbox_pred_list = nn.ModuleList([])
        for i in range(sample_duration):
            self.Conv_list.append(nn.Conv2d(self.din, din, 1, stride=1, padding=0, bias=True).cuda())
        
            self.head_to_tail_list.append(
                nn.Sequential(
                nn.Linear(din * 7 *  7, 2048),
                nn.ReLU(True),
                nn.Dropout(0.8),
                nn.Linear(2048,512),
                nn.ReLU(True)
                ).cuda())
            
            self.bbox_pred_list.append( nn.Linear(512,4).cuda())


        self.roi_align = RoIAlign(self.pooling_size, self.pooling_size, self.spatial_scale)
        self.avg_pool = nn.AvgPool3d((int(sample_duration),1,1), stride=1)
        self.reg_target = _Regression_TargetLayer()

            
    def forward(self, base_feat, rois, gt_rois):
        
        # base_feat.shape : [num_tubes, num_channels, sample_duration, width, height] : [32,512,16,4,4]

        # for i in range(self.sample_duration):
        #     self.Conv_list[i] = self.Conv_list[i].cuda()
        #     self.head_to_tail_list[i] = self.head_to_tail_list[i].cuda()
        #     self.bbox_pred_list[i] = self.bbox_pred_list[i].cuda()

        batch_size = rois.size(0)
        rois_per_image = rois.size(1)

        base_feat = F.normalize(base_feat, p=2, dim=1)

        offset = torch.arange(0,self.sample_duration).type_as(rois).unsqueeze(0).expand(batch_size,self.sample_duration)
        offset_batch = torch.arange(0,batch_size).type_as(rois) * self.sample_duration
        offset_batch = offset_batch.view(-1,1).expand(batch_size,self.sample_duration)

        offset = offset + offset_batch
        offset = offset.view(batch_size,1,self.sample_duration,1).expand(batch_size, rois_per_image, self.sample_duration,1)

        rois = rois.view(batch_size,rois_per_image,self.sample_duration, 4)
        rois = torch.cat((offset,rois),dim=3)
        
        rois = rois.permute(0,2,1,3).contiguous()

        base_feat = base_feat.permute(0,2,1,3,4).contiguous().view(-1,base_feat.size(1),base_feat.size(3),base_feat.size(4))

        base_feat = self.roi_align(base_feat, rois.view(-1,5))
        base_feat = base_feat.view(batch_size, self.sample_duration, rois_per_image,base_feat.size(1),base_feat.size(2),base_feat.size(3)).\
                    contiguous()

        base_feat = base_feat.permute(0,2,3,1,4,5).contiguous().view(batch_size*rois_per_image, base_feat.size(3),\
                                                                      self.sample_duration, base_feat.size(4),base_feat.size(5))
        
        bbox_pred = torch.zeros(base_feat.size(0),self.sample_duration,4).type_as(base_feat)
        for i in range(self.sample_duration):
            feat = base_feat[:,:,i]
            feat = self.Conv_list[i](feat)
            feat = self.head_to_tail_list[i](feat.view(feat.size(0),-1))
            bbox_pred[:,i] = self.bbox_pred_list[i](feat)
            # feat = self.Conv_list[i].to(feat.device)(feat)
            # feat = self.head_to_tail_list[i].to(feat.device)(feat.view(feat.size(0),-1))
            # bbox_pred[:,i] = self.bbox_pred_list[i].to(feat.device)(feat)

        bbox_pred = bbox_pred.view(bbox_pred.size(0),self.sample_duration*4).contiguous()

        # conv1_feats = self.Conv(base_feat)
        # conv1_feats = self.avg_pool(conv1_feats)
        # conv1_feats = self.head_to_tail_(conv1_feats.view(conv1_feats.size(0),-1))
        # bbox_pred = self.bbox_pred(conv1_feats) # regression layer

        return bbox_pred, base_feat

    
if __name__ == '__main__':

    sample_dur = 2


    t = torch.arange(0,280).view(4,5,2,7).unsqueeze(-1).expand(4,5,2,7,7).float()
    # print('t.shape :',t.shape)
    # rois_label = torch.Tensor([[ 2.,  7., 20., 15.]])
    tubes_ = torch.Tensor([[[ 0.,  4.,  3.,  6.,  8.,  3., 10.],
                            [ 0.,  3.,  6.,  5.,  6.,  5.,  9.],
                            [ 0.,  3.,  8., 10.,  4.,  9., 15.],
                            [ 0., 10.,  7.,  5.,  13.,  9.,  8.]]]).expand(4,4,7)

    print('tubes.shape :',tubes_.shape)

    gt_rois = torch.Tensor([[[[ 10, 11, 22, 23, 10], [ 0,  0,  0,  0, -1]],
                        [[ 22, 25, 32, 55, 11], [32, 12, 78, 32, 10]],
                         [[  0,  0,  0,  0, -1], [53, 42, 98, 60, 10]]]]).expand(4,3,2,5)

    reg = _Regression_Layer(5,2)
    # reg.eval()
    reg(t,tubes_, gt_rois)

    # # print(feats.shape)
    # # print('first feat, first frame   :',t[0,0,0])
    # # print('first feat, second frames :',t[0,0,1])
    # # print('second feat, first frame  :',t[0,1,0])
    # # print('after')
    # # print('first frame, first feat :',feats[0,0])
    # # print('first frame, sencd feat :',feats[0,1])
    # # print('sencd frame, first feat :',feats[1,0])
