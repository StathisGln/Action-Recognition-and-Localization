import torch
import torch.nn as nn
import torch.nn.functional as F

from resnet_3D import resnet34
from lib.roi_packages.roi_align_3d import RoIAlignAvg


class tcn_net(nn.Module):
    def __init__(self, classes, sample_duration, sample_size):
        super(tcn_net, self).__init__()

        self.classes = classes
        self.n_classes = len(classes)

        self.sample_duration = sample_duration
        self.sample_size = sample_size

        # self.tcn_net =  TCN(input_channels, self.n_classes, channel_sizes, kernel_size = kernel_size, dropout=dropout)
        self.rnn_neurons = 128
        self.tcn_avgpool = nn.AvgPool3d((16, 7, 7), stride=1)
        self.roi_align = RoIAlignAvg(7, 7, 16, 1.0/16.0, 1.0)
        # self.linear = nn.Linear(512,self.n_classes)
        self.prob = nn.Linear(256,self.n_classes)
    def forward(self, clips, target, gt_tubes, n_frames, max_dim=1):
        """Inputs have to have dimension (N, C_in, L_in)"""

        ## init act_rnn hidden state_
        # print('n_frames :',n_frames)
        # print('gt_tubes.shape :',gt_tubes.shape)
        if n_frames < 17:
            n_clips = 1
            indexes = torch.Tensor([0]).type_as(clips)
        else:
            n_clips = len(range(0, n_frames.item()-self.sample_duration, int(self.sample_duration/2)))
            indexes = torch.arange(0, n_frames.item()-self.sample_duration, int(self.sample_duration/2)).type_as(clips)
        # print('indexes :',indexes)
        # features = torch.zeros(1,len(indexes),512).type_as(clips)

        features = torch.zeros(1,n_clips,256).type_as(clips)
        rois = torch.zeros(n_clips,max_dim, 7).type_as(clips)
        off_set = torch.zeros(gt_tubes.shape).type_as(gt_tubes)
        off_set[:,:n_clips,2]=indexes
        off_set[:,:n_clips,5]=indexes
        gt_tubes_r = gt_tubes - off_set
        rois[:,:,0] = torch.arange(n_clips).unsqueeze(1).expand(n_clips,max_dim)
        rois = rois.view(-1,7)
        rois[:,1:] = gt_tubes_r[:,:n_clips,:6]
        clips_ = clips[:,:n_clips] # keep only clips containing info
        feats = self.base_model(clips_.squeeze(0))
        pooled_feat = self.roi_align(feats, rois)

        # fc7 = self.top_part(pooled_feat)
        # fc7 = self.tcn_avgpool(fc7)

        fc7 = self.tcn_avgpool(pooled_feat)
        fc7 = fc7.view(n_clips,-1)#.permute(1,0)
        features_mean = torch.mean(fc7,0)
        
        output = self.prob(features_mean)
        # output = F.softmax(output, 0)
        # print(' output :',output)
        if self.training :
            tcn_loss = F.cross_entropy(output.unsqueeze(0), target.long())

        if self.training:
            return output.unsqueeze(0), tcn_loss

        output = F.softmax(output, 0)
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
        # self.top_part = nn.Sequential(model.module.layer4)

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
        # self.top_part.apply(set_bn_fix)

