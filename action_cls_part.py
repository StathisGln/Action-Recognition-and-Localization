import os
from collections import OrderedDict 

import torch
import torch.nn as nn
import torch.nn.functional as F

from resnet_3D import resnet34_orig, resnet34

class ACT_cls(nn.Module):

    def __init__(self, actions, sample_duration):

        super(ACT_cls, self).__init__()

        self.classes = actions
        self.n_classes = len(actions)
        self.sample_duration = sample_duration
        self.din = 256

        self.cls = nn.Linear(512,self.n_classes).cuda()


    def forward(self, im_data, im_info, gt_tubes, gt_rois,  target):

        batch_size = im_data.size(0)

        cls_feats = self.base_net_1(im_data)
        cls_feats = self.base_net_2(cls_feats)
        cls_feats = self.top_net(cls_feats).mean(4).mean(3).squeeze()
        cls_scr = self.cls(cls_feats)

        if self.training:

            cls_loss =  F.cross_entropy(cls_scr, target)

            return  cls_scr, cls_loss

        cls_scr = F.softmax(cls_scr, dim=1)

        return  cls_scr, None


    def create_architecture(self, act_net_path=None):

        self._init_modules()

    def _init_modules(self):

        resnet_shortcut = 'A'
        sample_size = 112

        model = resnet34(num_classes=400, shortcut_type=resnet_shortcut,
                              sample_size=sample_size, sample_duration=self.sample_duration,
                              last_fc=False)


        self.model_path = '../resnet-34-kinetics.pth'

        print("Loading pretrained weights from %s" %(self.model_path))
        model_data = torch.load(self.model_path)

        new_state_dict = OrderedDict()
        for k, v in model_data['state_dict'].items():
          name = k[7:] # remove `module.`
          new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

        self.base_net_1 = nn.Sequential(model.conv1, model.bn1, model.relu,
                                        model.maxpool,model.layer1)

        model = resnet34_orig(num_classes=400, shortcut_type=resnet_shortcut,
                              sample_size=sample_size, sample_duration=self.sample_duration,
                              last_fc=False)

        model.load_state_dict(new_state_dict)

        self.base_net_2 = nn.Sequential(model.layer2,model.layer3)
        self.top_net = nn.Sequential(model.layer4)
        
        for p in self.base_net_1.parameters(): p.requires_grad=False
        for p in self.base_net_2.parameters(): p.requires_grad=False

        def set_bn_fix(m):
          classname = m.__class__.__name__
          if classname.find('BatchNorm') != -1:
            for p in m.parameters(): p.requires_grad=False

        self.base_net_1.apply(set_bn_fix)
        self.base_net_2.apply(set_bn_fix)
        self.top_net.apply(set_bn_fix)
