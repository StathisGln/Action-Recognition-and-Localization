from __future__ import absolute_import, print_function

import os
import numpy as np
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F

from resnet_3D import resnet34_rest

class RestNet(nn.Module):
    """ action tube proposal """
    def __init__(self, actions,sample_duration, pooling_time):
        super(RestNet, self).__init__()

        self.classes = actions
        self.n_classes = len(actions)
        self.sample_duration = sample_duration

        self.pooling_time = pooling_time
        self.time_dim =sample_duration
        # self.cls = nn.Linear(512, self.n_classes)
        self.cls = nn.Linear(512*int(pooling_time/2), self.n_classes)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

    def temporal_pool(self, features):

        n_filters = features.size(1)
        x_axis = features.size(3)
        y_axis = features.size(4)
        
        indexes = torch.linspace(0, features.size(0), self.pooling_time+1).int()
        ret_feats = torch.zeros(self.pooling_time, n_filters,\
                                self.sample_duration,x_axis, y_axis)

        if features.size(0) < self.pooling_time:

            ret_feats[:features.size(0)] = features
            return ret_feats
    
        for i in range(self.pooling_time):
            
            t = features[indexes[i]:indexes[i+1]].permute(1,0,2,3,4)
            t = t.view(t.size(0),t.size(1),t.size(2),-1)
            t = F.max_pool3d(t,kernel_size=(indexes[i+1]-indexes[i],1,1)).squeeze()

            ret_feats[i] = t.view(t.size(0),t.size(1), features.size(3),features.size(4))

        return ret_feats

    def forward(self, features, combinations,  target, mode):

        self.cls_loss = 0
        
        batch_size = combinations.size(0)
        rois_per_image = combinations.size(1)
        n_filters = features.size(3)
        x_axis = features.size(5)
        y_axis = features.size(6)

        f_features = features.new(batch_size, rois_per_image, self.pooling_time, n_filters,\
                                 x_axis, y_axis).zero_().type_as(features)

        len_tubes = features.new(batch_size, rois_per_image)

        for b in range(batch_size):
            for i in range(24):
            # for i in range(rois_per_image):


                t = combinations[b,i].ne(-1).all(dim=1).nonzero().view(-1)
                indices = combinations[b,i,t].long()

                if t.numel() == 0:
                    break

                len_tubes[b,i]  = t.numel()

                f_features[b,i]  = self.temporal_pool(features[b,indices[:,0],indices[:,1]]).max(2)[0]

        f_features = f_features.permute(0,1,3,2,4,5)
        
        feats = self.top_net(f_features.contiguous().view(batch_size*rois_per_image,\
                                             n_filters, self.pooling_time, x_axis,y_axis))


        if mode == 'extract':

            feats = feats.view(batch_size,rois_per_image,512,4,4)

            return (feats,target, len_tubes), 0

        feats = feats.mean(4).mean(3)

        cls_scr = self.cls(feats.view(feats.size(0),-1))

        if self.training:

            target = target.view(-1).contiguous()

            self.cls_loss =  F.cross_entropy(cls_scr, target.long())


        cls_scr = F.softmax(cls_scr, dim=1)

        return  cls_scr, self.cls_loss

        

    
    def _init_weights(self):

        def normal_init(m, mean, stddev, ptruncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()


        truncated = False

        normal_init(self.cls, 0, 0.01, truncated)


    def _init_modules(self):

        resnet_shortcut = 'A'
        # resnet_shortcut = 'B'
        
        sample_size = 112

        model = resnet34_rest(num_classes=400, shortcut_type=resnet_shortcut,
                         sample_size=sample_size, sample_duration=self.sample_duration,
                         last_fc=False)


        self.model_path = '../resnet-34-kinetics.pth'
        # self.model_path = '../resnext-101-kinetics.pth'
        print("Loading pretrained weights from %s" %(self.model_path))

        model_data = torch.load(self.model_path)

        # new_state_dict = OrderedDict()
        # for k, v in model_data['state_dict'].items():
        #   name = k[7:] # remove `module.`
        #   new_state_dict[name] = v

        self.top_net = nn.Sequential(model.layer4)

        def set_bn_fix(m):
          classname = m.__class__.__name__
          if classname.find('BatchNorm') != -1:
            for p in m.parameters(): p.requires_grad=False

        self.top_net.apply(set_bn_fix)

if __name__ == '__main__':

    actions = ['__background__', 'Basketball','BasketballDunk','Biking','CliffDiving','CricketBowling',
               'Diving','Fencing','FloorGymnastics','GolfSwing','HorseRiding','IceDancing',
               'LongJump','PoleVault','RopeClimbing','SalsaSpin','SkateBoarding','Skiing',
               'Skijet','SoccerJuggling','Surfing','TennisSwing','TrampolineJumping',
               'VolleyballSpiking','WalkingWithDog']

    comb =  torch.Tensor([[[[ 0.,  5.], [ 1.,  1.],  [ 2.,  0.],  [ 3.,  0.], [ 4.,  0.],  [ 5.,  0.],  [ 6.,  0.], [ 7.,  0.],
                           [ 8.,  0.],  [ 9.,  0.],  [10.,  0.],  [11.,  0.],  [12.,  0.]],
                          [[ 0.,  0.],  [ 1.,  0.], [ 2.,  0.], [ 3.,  0.], [ 4.,  9.], [ 5.,  0.], [ 6.,  0.], [ 7.,  0.], [ 8.,  0.],
                            [ 9.,  0.], [10.,  0.], [11.,  0.], [12.,  0.]],
                          [[ 0.,  0.], [ 1.,  0.], [ 2.,  0.], [ 3.,  0.], [ 4.,  0.], [ 5.,  0.], [ 6.,  0.], [ 7., 12.], [ 8.,  0.],
                            [ 9.,  0.], [10.,  0.], [11.,  0.], [-1., -1.]]]])
    target=torch.Tensor([[12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12.,  0.,  0.,
                          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                         [ 4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,
                           4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,
                           4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.]])
    features = torch.rand([2, 13, 16, 256, 8, 7, 7])

    rest = RestNet(actions,8,8)
    rest.create_architecture()
    ret = rest(features, comb, target)
