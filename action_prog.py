import os
from collections import OrderedDict 

import torch
import torch.nn as nn
import torch.nn.functional as F

from action_net import ACT_net
from progress_rate import Progress_Rate

class Act_Prog(nn.Module):

    def __init__(self, actions, sample_duration):

        super(Act_Prog, self).__init__()

        self.classes = actions
        self.n_classes = len(actions)
        self.sample_duration = sample_duration

        self.act_net = ACT_net(actions, sample_duration)
        self.prg_rt = Progress_Rate(64, self.n_classes, self.sample_duration)

    def forward(self, im_data, im_info, gt_tubes, gt_rois,  start_fr, rate):

        batch_size = im_data.size(0)

        # with torch.no_grad():

        rois,  feats, rpn_loss_cls, rpn_loss_bbox, rois_rate, \
        rois_progress, rois_label, \
        sgl_rois_bbox_pred, sgl_rois_bbox_loss = self.act_net(im_data, im_info, gt_tubes,\
                                                                  gt_rois,  start_fr, rate)
        
        r_rate, rate_loss, r_prog, prog_loss, = self.prg_rt(feats, rois_rate, rois_progress, rois_label)

        feats = feats.view(batch_size,-1, feats.size(1), feats.size(2), feats.size(3), feats.size(4))

        if self.training:

            rois_label =rois_label.view(batch_size,-1).long()

            return  rois, feats, rpn_loss_cls,  rpn_loss_bbox, r_rate, rate_loss,  \
                r_prog, prog_loss,  rois_label, \
                sgl_rois_bbox_pred, sgl_rois_bbox_loss


        return  rois, feats, None, None, r_rate, None,  \
            r_prog, None, None, \
            sgl_rois_bbox_pred, None


    def create_architecture(self, act_net_path=None):

        self.act_net.create_architecture()

        if act_net_path != None:

            # self.act_net = nn.DataParallel(self.act_net, device_ids=None)
            model_data = torch.load(act_net_path)

            new_state_dict = OrderedDict()
            for k, v in model_data.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v

            self.act_net.load_state_dict(new_state_dict)


