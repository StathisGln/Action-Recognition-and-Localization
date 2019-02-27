import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from resnet_3D import resnet34
from roi_align_3d.modules.roi_align  import RoIAlignAvg
from tcn import TCN
from act_rnn import Act_RNN


class tcn_net(nn.Module):
    def __init__(self, classes, sample_duration, sample_size, input_channels, channel_sizes, kernel_size, dropout):
        super(tcn_net, self).__init__()

        self.classes = classes
        self.n_classes = len(classes)

        self.sample_duration = sample_duration
        self.sample_size = sample_size

        # self.tcn_net =  TCN(input_channels, self.n_classes, channel_sizes, kernel_size = kernel_size, dropout=dropout)
        self.rnn_neurons = 128
        # self.act_rnn = Act_RNN(1,512,self.rnn_neurons,self.n_classes)
        self.act_rnn = Act_RNN(1,256,self.rnn_neurons,self.n_classes)
        # self.tcn_avgpool = nn.AvgPool3d((16, 4, 4), stride=1)
        self.tcn_avgpool = nn.AvgPool3d((16, 7, 7), stride=1)
        self.roi_align = RoIAlignAvg(7, 7, 16, 1.0/16.0, 1.0)


    def forward(self, clips, target, gt_tubes, n_frames, max_dim=1):
        """Inputs have to have dimension (N, C_in, L_in)"""

        ## init act_rnn hidden state_
        batch_size = clips.size(0)
        
        if n_frames < 17:
            indexes = [0]
        else:
            indexes = range(0, (n_frames.data - self.sample_duration  ), int(self.sample_duration/2))

        # features = torch.zeros(1,len(indexes),512).type_as(clips)
        features = torch.zeros(1,len(indexes),256).type_as(clips)

        rois = torch.zeros(max_dim, 7).type_as(clips)

        # for every sequence extract features
        for i in indexes:

            lim = min(i+self.sample_duration, (n_frames.item()))
            vid_indices = torch.arange(i,lim).long()
            rois[:,1:] = gt_tubes[:,int(i*2/self.sample_duration),:6]
            rois[:,3] = rois[:,3] - i
            rois[:,6] = rois[:,6] - i

            vid_seg = clips[:,:,vid_indices]

            outputs = self.base_model(vid_seg)
            # print(' outputs.shape: ',outputs.shape)
            pooled_feat = self.roi_align(outputs,rois)

            # fc7 = self.top_part(pooled_feat)
            # fc7 = self.tcn_avgpool(fc7)
            fc7 = self.tcn_avgpool(pooled_feat)
            
            # print('fc7.shape :',fc7.shape)
            fc7 = fc7.view(-1)

            features[0,int(i*2/self.sample_duration)] = fc7
        print('features :',features)
        self.act_rnn.hidden = torch.zeros(1,batch_size,self.rnn_neurons).cuda()

        output = self.act_rnn(features)
        print(' output :',output)
        output = F.log_softmax(output, 1)
        tcn_loss = F.nll_loss(output, target.long())

        if self.training:
            return output, tcn_loss

        return output, None
        
    def create_architecture(self):

        self._init_modules()
        
    def _init_modules(self):

        last_fc = False
        n_classes = 400
        resnet_shortcut = 'A'

        model = resnet34(num_classes=n_classes, shortcut_type=resnet_shortcut,
                               sample_size=self.sample_size, sample_duration=self.sample_duration,
                               last_fc=last_fc)

        model = nn.DataParallel(model)
        self.model_path = '/gpu-data2/sgal/resnet-34-kinetics.pth'
        print("Loading pretrained weights from %s" %(self.model_path))
        model_data = torch.load(self.model_path)
        model.load_state_dict(model_data['state_dict'])

        self.base_model = nn.Sequential(model.module.conv1, model.module.bn1, model.module.relu,
          model.module.maxpool,model.module.layer1,model.module.layer2, model.module.layer3)
        self.top_part = nn.Sequential(model.module.layer4)

        # Fix blocks
        for p in self.base_model[0].parameters(): p.requires_grad=False
        for p in self.base_model[1].parameters(): p.requires_grad=False

        fixed_blocks = 3
        if fixed_blocks >= 3:
          for p in self.base_model[6].parameters(): p.requires_grad=False
        if fixed_blocks >= 2:
          for p in self.base_model[5].parameters(): p.requires_grad=False
        if fixed_blocks >= 1:
          for p in self.base_model[4].parameters(): p.requires_grad=False

        def set_bn_fix(m):
          classname = m.__class__.__name__
          if classname.find('BatchNorm') != -1:
            for p in m.parameters(): p.requires_grad=False

        self.base_model.apply(set_bn_fix)
        self.top_part.apply(set_bn_fix)

