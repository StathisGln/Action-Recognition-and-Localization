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

        self.Conv = nn.Conv2d(self.din, din, 1, stride=1, padding=0, bias=True)
        self.head_to_tail_ = nn.Sequential(
            nn.Linear(64 *7*  7, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024,4096),
            nn.ReLU(True)
            )
            
        # self.avg_pool = nn.AvgPool2d((7, 7), stride=1)
        self.bbox_pred = nn.Linear(4096,4)

        self.roi_align = RoIAlign(self.pooling_size, self.pooling_size, self.spatial_scale)

        self.reg_target = _Regression_TargetLayer()

    def forward(self, base_feat, tubes, gt_rois):

        batch_size = tubes.size(0)
        base_feat = F.normalize(base_feat, p=2, dim=1)
        offset = torch.arange(0,self.sample_duration)
        rois = from_tubes_to_rois(tubes, self.sample_duration)

        ## modify tubes and rois_label
        if self.training:

            # print('tubes :',tubes.cpu().numpy())
            # print('tubes.shape :',tubes.shape)
            # print('gt_rois :',gt_rois.cpu().numpy())
            # print('gt_rois.shaep :',gt_rois.shape)

            rois, labels, \
            bbox_targets, bbox_inside_ws,\
            bbox_outside_ws = self.reg_target(rois, gt_rois.permute(0,2,1,3))

            labels = Variable(labels.view(-1).long())
            bbox_targets = Variable(bbox_targets.view(-1, bbox_targets.size(2)))
            bbox_inside_ws = Variable(bbox_inside_ws.view(-1, bbox_inside_ws.size(2)))
            bbox_outside_ws = Variable(bbox_outside_ws.view(-1, bbox_outside_ws.size(2)))

        ## modify feat
        base_feat = base_feat.permute(0,2,1,3,4).contiguous().view(-1,base_feat.size(1),base_feat.size(3),base_feat.size(4))
        base_feat = self.roi_align(base_feat, rois.view(-1,5))

        conv1_feats = self.Conv(base_feat)
        print('conv1_feats.shape :',conv1_feats.shape)
        conv1_feats = self.head_to_tail_(conv1_feats.view(conv1_feats.size(0),-1))
        print('conv1_feats.shape :',conv1_feats.shape)
        exit(-1)
        bbox_pred = self.bbox_pred(conv1_feats) # regression layer

        if self.training:
            # bounding box regression L1 loss
            # print('bbox_inside_ws :',bbox_inside_ws)
            # print('bbox_targets :',bbox_targets.cpu().numpy())
            rois_loss_bbox = _smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_ws, bbox_outside_ws)
            return bbox_pred, rois_loss_bbox

        return bbox_pred, None

    
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
