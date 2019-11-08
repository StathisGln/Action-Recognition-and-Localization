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
    def __init__(self, actions,sample_duration):
        super(RestNet, self).__init__()

        self.classes = actions
        self.n_classes = len(actions)
        self.sample_duration = sample_duration

        self.pooling_time = 2
        self.time_dim =sample_duration
        self.cls = nn.Linear(512, self.n_classes)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

    def temporal_pool(self, features):

        batch_size = features.size(0)
        n_filters = features.size(2)
        x_axis = features.size(4)

        
        indexes = torch.linspace(0, features.size(1), self.pooling_time+1).int()
        ret_feats = torch.zeros(batch_size,self.pooling_time, n_filters,\
                                self.sample_duration,x_axis, x_axis)

        if features.size(1) < 2:

            ret_feats[0] = features
            return ret_feats
    
        for i in range(self.pooling_time):
            
            t = features[:,indexes[i]:indexes[i+1]].permute(0,2,1,3,4,5).\
                contiguous().view(batch_size*n_filters,-1,self.sample_duration,x_axis,y_axis)
            t = t.view(t.size(0),t.size(1),t.size(2),-1)
            t = F.max_pool3d(t,kernel_size=(indexes[i+1]-indexes[i],1,1)).squeeze()

            ret_feats[:,i] = t.view(batch_size,n_filters,self.sample_duration,x_axis,y_axis)

        return ret_feats

    def forward(self, features, n_tubes, len_tubes, target):

        print('features.shape :',features.shape)
        self.cls_loss = 0
        batch_size = features.size(0)
        rois_per_image = features.size(1)
        n_filters = features.size(3)
        x_axis = features.size(5)

        f_features = features.new(batch_size, rois_per_image, self.pooling_time, n_filters,\
                                 x_axis, x_axis).zero_()

        
        for i in range(batch_size):

            f_features[i] = self.temporal_pool(features[i,:n_tubes[i],:len_tubes[i,0]]).max(3)[0]

        f_features = f_features.permute(0,1,3,2,4,5).contiguous()

        feats = self.top_net(f_features.view(batch_size*rois_per_image,\
                                             n_filters, self.pooling_time, x_axis,y_axis)).\
                                             mean(4).mean(3).squeeze()
        cls_scr = self.cls(feats)

        if self.training:

            target = target.view(-1).contiguous()
            # non_zero_tubes = target.nonzero().view(-1)

            # cls_scr = cls_scr[non_zero_tubes]
            # target = target[non_zero_tubes]

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
