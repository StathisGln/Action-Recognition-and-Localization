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

        self.Conv = nn.Conv3d(self.din, din, 1, stride=1, padding=0, bias=True)
        self.head_to_tail_ = nn.Sequential(
            # nn.Linear(din*7*7 , 2048),
            nn.Linear(din*self.sample_duration*7*7 , 2048),

            nn.ReLU(True),
            nn.Dropout(0.8),
            nn.Linear(2048,512),
            nn.ReLU(True)
            )
            
        self.avg_pool = nn.AvgPool3d((sample_duration, 1, 1), stride=1)
        self.bbox_pred = nn.Linear(512,self.sample_duration*4)

        self.roi_align = RoIAlign(self.pooling_size, self.pooling_size, )
        # self.roi_align = RoIAlign(self.pooling_size, self.pooling_size, self.spatial_scale)

        self.max_pool = nn.MaxPool3d((1,7,7), stride=1)
        self.reg_target = _Regression_TargetLayer()

        self.init_head_to_tail_weights()

    def init_head_to_tail_weights(self, stddev=0, mean=0.01,):
        for m in self.head_to_tail_.modules():
            if m == nn.Linear:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
        
    def forward(self, feat_maps, rois, gt_rois, im_info):
        
        # base_feat.shape : [num_tubes, num_channels, sample_duration, width, height] : [32,512,16,4,4]

        h = rois.data[:, 4::4] - rois.data[:, 2::4] + 1
        w = rois.data[:, 3::4] - rois.data[:, 1::4] + 1
        h = torch.mean(h,dim=1)
        w = torch.mean(w,dim=1)

        roi_level = torch.log(torch.sqrt(h * w) / 112.0)
        roi_level = torch.round(roi_level + 4)
        roi_level[roi_level < 2] = 2
        roi_level[roi_level > 5] = 5
        
        roi_align_feats = []
        box_to_levels = []

        offset = torch.arange(0,self.sample_duration).view(1,self.sample_duration).contiguous().type_as(rois)

        for i, l in enumerate(range(2, 6)):

            feats = F.normalize(feat_maps[i], p=2, dim=1)

            # if there is no rois in this area
            if (roi_level == l).sum() == 0:
                continue
            idx_l = (roi_level == l).nonzero().squeeze()
            n_rois = idx_l.nelement()

            rois_ = rois[idx_l,1:].contiguous().view(n_rois, self.sample_duration, 4)
            rois_batch = rois[idx_l,0].contiguous().view(n_rois,1).expand(n_rois,self.sample_duration)
            rois_batch = rois_batch * self.sample_duration + offset.expand(n_rois,self.sample_duration)
            rois_batch = rois_batch.view(n_rois, self.sample_duration,1).contiguous()
            rois_ = torch.cat([rois_batch, rois_], 2)

            rois_ = rois_.permute(1,0,2).contiguous().view(-1,5)
            if idx_l.nelement() == 1:
                idx_l = idx_l.unsqueeze(0)
            box_to_levels.append(idx_l)
            scale = feats[i].size(3) / im_info[0][0]

            feats = feats.permute(0,2,1,3,4).contiguous().view(-1,feats.size(1),feats.size(3),feats.size(4)).contiguous()
            feat = self.roi_align(feats, rois_, scale)

            feat = feat.view( self.sample_duration, n_rois ,feat.size(1),feat.size(2),feat.size(3)).\
                    contiguous()
            feat = feat.permute(1,2,0,3,4).contiguous().view(n_rois, feat.size(2),\
                                                                      self.sample_duration, feat.size(3),feat.size(4))
                    
            roi_align_feats.append(feat)


        roi_align_feat = torch.cat(roi_align_feats, 0)
        try:
            box_to_level = torch.cat(box_to_levels, 0)
        except Exception as e:
            print('roi_level :',roi_level)
            print('box_to_levels :',box_to_levels)
            print(e)
            exit(-1)
        idx_sorted, order = torch.sort(box_to_level)

        roi_align_feat = roi_align_feat[order]

        conv1_feats = self.Conv(roi_align_feat)
        # conv1_feats = self.avg_pool(conv1_feats)
        conv1_feats = self.head_to_tail_(conv1_feats.view(conv1_feats.size(0),-1))
        bbox_pred = self.bbox_pred(conv1_feats) # regression layer
        base_feat = self.max_pool(roi_align_feat)

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
