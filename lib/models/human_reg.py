from __future__ import absolute_import, print_function
import torch
import torch.nn as nn

from .resnet_3D import resnet34

from lib.roi_packages.roi_align.modules.roi_align import RoIAlign
from .proposal_target_layer_cascade_original import _ProposalTargetLayer as _Regression_TargetLayer


class _Regression_Layer(nn.Module):
    """
    This class gets as input, tubes and feats and finds the regression
    """
    def __init__(self, din, sample_duration):


        super(_Regression_Layer, self).__init__()
        print('sample_duration :',sample_duration)
        self.din = din
        self.sample_duration = sample_duration
        self.pooling_size = 7
        self.spatial_scale = 1.0/16

        self.head_to_tail_ = nn.Sequential(

            nn.Linear(din * 2 * sample_duration * 4 * 4, 2048),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(2048,512),
            nn.ReLU(True)
            )
            

        self.avg_pool = nn.AvgPool3d((sample_duration, 1, 1), stride=1)
        self.bbox_pred = nn.Linear(512,self.sample_duration*4)

        # self.batch_norm = nn.BatchNorm3d(self.din,7,7)
        self.batch_norm = nn.BatchNorm3d(self.din)
        self.roi_align = RoIAlign(self.pooling_size, self.pooling_size, self.spatial_scale)
        self.max_pool = nn.MaxPool3d((1,7,7), stride=1)

        self.reg_target = _Regression_TargetLayer()

        self.init_head_to_tail_weights()

    def init_head_to_tail_weights(self, stddev=0, mean=0.01,):
        for m in self.head_to_tail_.modules():
            if m == nn.Linear:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
        
    def forward(self, base_feat, rois, gt_rois):
        
        # base_feat.shape : [num_tubes, num_channels, sample_duration, width, height] : [32,512,16,4,4]

        batch_size = rois.size(0)
        rois_per_image = rois.size(1)

        # base_feat = F.normalize(base_feat, p=2, dim=1)
        base_feat = self.batch_norm(base_feat)

        offset = torch.arange(0,self.sample_duration).type_as(rois).unsqueeze(0).expand(batch_size,self.sample_duration)
        offset_batch = torch.arange(0,batch_size).type_as(rois) * self.sample_duration
        offset_batch = offset_batch.view(-1,1).expand(batch_size,self.sample_duration)

        offset = offset + offset_batch
        offset = offset.view(batch_size,1,self.sample_duration,1).expand(batch_size, rois_per_image, self.sample_duration,1)

        rois = rois.view(batch_size,rois_per_image,self.sample_duration, 4)
        rois = torch.cat((offset,rois),dim=3)
        
        rois = rois.permute(0,2,1,3).contiguous()

        base_feat = base_feat.permute(0,2,1,3,4).contiguous().view(-1,base_feat.size(1),base_feat.size(3),base_feat.size(4)).contiguous()
        base_feat = self.roi_align(base_feat, rois.view(-1,5))
        base_feat = base_feat.view(batch_size, self.sample_duration, rois_per_image,base_feat.size(1),base_feat.size(2),base_feat.size(3)).\
                    contiguous()
        base_feat = base_feat.permute(0,2,3,1,4,5).contiguous().view(batch_size*rois_per_image, base_feat.size(3),\
                                                                      self.sample_duration, base_feat.size(4),base_feat.size(5))
        
        conv1_feats = self.Conv(base_feat)
        # conv1_feats = self.avg_pool(conv1_feats)
        conv1_feats = self.head_to_tail_(conv1_feats.view(conv1_feats.size(0),-1))
        bbox_pred = self.bbox_pred(conv1_feats) # regression layer
        # base_feat = self.max_pool(base_feat)
        return bbox_pred, base_feat

    def _init_modules(self, model_path='../../../resnet-34-kinetics.pth'):

        resnet_shortcut = 'A'
        # resnet_shortcut = 'B'
        
        sample_size = 112

        model = resnet34(num_classes=400, shortcut_type=resnet_shortcut,
                         sample_size=sample_size, sample_duration=self.sample_duration,
                         last_fc=False)
        # model = resnet101(num_classes=400, shortcut_type=resnet_shortcut,
        #                  sample_size=sample_size, sample_duration=self.sample_duration,
        #                  )

        model = nn.DataParallel(model, device_ids=None)
        self.model_path = model_path
        # self.model_path = '../resnext-101-kinetics.pth'
        print("Loading pretrained weights from %s" %(self.model_path))
        model_data = torch.load(self.model_path)

        model.load_state_dict(model_data['state_dict'])

        # Build resnet.
        self.Conv = nn.Sequential(model.module.layer4)

        def set_bn_fix(m):
          classname = m.__class__.__name__
          if classname.find('BatchNorm') != -1:
            for p in m.parameters(): p.requires_grad=False
    
        self.Conv.apply(set_bn_fix)




    
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
