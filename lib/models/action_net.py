from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.autograd import Variable

from .resnet_3D import resnet34
from .region_net import _RPN
from .human_reg import _Regression_Layer
# from .human_reg_3d import _Regression_Layer
# from .human_reg_2d import _Regression_Layer



from .proposal_target_layer_cascade import _ProposalTargetLayer
# from roi_align_3d.modules.roi_align  import RoIAlignAvg, RoIAlign
from lib.utils.net_utils import _smooth_l1_loss


class ACT_net(nn.Module):
    """ action tube proposal """
    def __init__(self, actions,sample_duration, device='cuda'):
        super(ACT_net, self).__init__()

        print(f'Using device : {device}')
        self.classes = actions
        self.n_classes = len(actions)
        self.sample_duration = sample_duration

        # loss
        self.act_loss_cls = 0
        self.act_loss_bbox = 0

        # # cfg.POOLING_SIZE
        # self.pooling_size = 7
        # # self.spatial_scale = 1.0/16
        # self.spatial_scale = 1.0/4
    
        # define rpn
        self.act_rpn = _RPN(256, sample_duration).to(device)

        self.act_proposal_target = _ProposalTargetLayer(sample_duration).to(device) ## background/ foreground

        self.time_dim =sample_duration
        self.temp_scale = 1.0
        # self.reg_layer = _Regression_Layer(64, self.sample_duration).cuda()
        # self.reg_layer = _Regression_Layer(128, self.sample_duration).cuda()
        # self.reg_layer = _Regression_Layer(256, self.sample_duration).cuda()
        self.reg_layer = _Regression_Layer(256, self.sample_duration)
        # print(f'self.reg_layer.device {self.reg_layer.type()}')
        self.batch_norm = nn.BatchNorm3d(256)

    def create_architecture(self, model_path='../../../resnet-34-kinetics.pth'):
        self.model_path = model_path
        self._init_modules()
        self._init_weights()
        self.reg_layer._init_modules(model_path)

    def forward(self, im_data, im_info, gt_tubes, gt_rois,  start_fr):

        batch_size = im_data.size(0)
        im_info = im_info.data

        # feed image data to base model to obtain base feature map
        base_feat_1 = self.act_base_1(im_data)
        base_feat_2 = self.act_base_2(base_feat_1)
        base_feat_3 = self.act_base_3(base_feat_2)
        base_feat_3 = self.batch_norm(base_feat_3)

        rois, _, rpn_loss_cls, rpn_loss_bbox, \
            _, _ = self.act_rpn(base_feat_3, im_info, gt_tubes, gt_rois)

        if self.training:

            roi_data = self.act_proposal_target(rois, gt_rois)

            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data
            rois_label =rois_label.view(-1).long()
            rois_target = rois_target.view(-1, rois_target.size(2))
            rois_inside_ws = rois_inside_ws.view(-1, rois_inside_ws.size(2))
            rois_outside_ws = rois_outside_ws.view(-1, rois_outside_ws.size(2))

        else:

            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0
            
        # sgl_rois_bbox_pred, feats = self.reg_layer(base_feat_1,rois[:,:,1:-1], gt_rois)
        sgl_rois_bbox_pred, feats = self.reg_layer(base_feat_3,rois[:,:,1:-1], gt_rois)

        if not self.training:
            sgl_rois_bbox_pred = Variable(sgl_rois_bbox_pred, requires_grad=False)

        if self.training:

            sgl_rois_bbox_loss = _smooth_l1_loss(sgl_rois_bbox_pred, rois_target, rois_inside_ws, rois_outside_ws,
                                                 )
            sgl_rois_bbox_pred = sgl_rois_bbox_pred.view(batch_size,-1, self.sample_duration*4)

            feats = feats.view(batch_size,-1, feats.size(1), feats.size(2), feats.size(3), feats.size(4))
            rois_label =rois_label.view(batch_size,-1).long()

            # return rois,  None, rpn_loss_cls, rpn_loss_bbox, None,None, \
            #     None, None, None
            
            return rois,  feats, rpn_loss_cls, rpn_loss_bbox, None,None, \
                rois_label, sgl_rois_bbox_pred, sgl_rois_bbox_loss

        sgl_rois_bbox_pred = sgl_rois_bbox_pred.view(batch_size,-1, self.sample_duration*4)
        feats = feats.view(batch_size,-1, feats.size(1), feats.size(2), feats.size(3), feats.size(4))

        # # TODO remove after adding reg_layer
        # if self.training:
        #     return rois,  None, rpn_loss_cls, rpn_loss_bbox, None,None, \
        #         None, None, None

        # # TODO remove after adding reg_layer
        # return rois, None, None, None, None, None, \
        #     None, None, None,

        return rois, feats, None, None, None, None, \
            None, sgl_rois_bbox_pred, None,


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
        normal_init(self.act_rpn.RPN_Conv, 0, 0.01, truncated)
        normal_init(self.act_rpn.RPN_cls_score, 0, 0.01, truncated)
        normal_init(self.act_rpn.RPN_cls_score_3_4, 0, 0.01, truncated)
        normal_init(self.act_rpn.RPN_cls_score_2, 0, 0.01, truncated)
        normal_init(self.act_rpn.RPN_cls_score_4, 0, 0.01, truncated)
        normal_init(self.act_rpn.RPN_bbox_pred, 0, 0.01, truncated)
        normal_init(self.act_rpn.RPN_bbox_pred_3_4, 0, 0.01, truncated)
        normal_init(self.act_rpn.RPN_bbox_pred_2, 0, 0.01, truncated)
        normal_init(self.act_rpn.RPN_bbox_pred_4, 0, 0.01, truncated)

    def _init_modules(self, ):

        resnet_shortcut = 'A'
        # resnet_shortcut = 'B'
        
        sample_size = 112

        model = resnet34(num_classes=400, shortcut_type=resnet_shortcut,
                         sample_size=sample_size, sample_duration=self.sample_duration,
                         last_fc=False, )
        # model = resnet101(num_classes=400, shortcut_type=resnet_shortcut,
        #                  sample_size=sample_size, sample_duration=self.sample_duration,
        #                  )

        model = nn.DataParallel(model, device_ids=None)


        print("Loading pretrained weights from %s" %(self.model_path))
        model_data = torch.load(self.model_path)

        model.load_state_dict(model_data['state_dict'])

        # Build resnet.
        self.act_base_1 = nn.Sequential(model.module.conv1, model.module.bn1, model.module.relu,
                                        model.module.maxpool,model.module.layer1)
        self.act_base_2 = nn.Sequential(model.module.layer2)
        self.act_base_3 = nn.Sequential(model.module.layer3)

        # Fix blocks
        for p in self.act_base_1[0].parameters(): p.requires_grad=False
        for p in self.act_base_1[1].parameters(): p.requires_grad=False

        fixed_blocks = 3
        if fixed_blocks >= 3:
          for p in self.act_base_3[0].parameters(): p.requires_grad=False
          # for p in self.act_base_2[0].parameters(): p.requires_grad=False
        if fixed_blocks >= 2:
          for p in self.act_base_2[0].parameters(): p.requires_grad=False
        if fixed_blocks >= 1:
          for p in self.act_base_1[4].parameters(): p.requires_grad=False

        def set_bn_fix(m):
          classname = m.__class__.__name__
          if classname.find('BatchNorm') != -1:
            for p in m.parameters(): p.requires_grad=False
    
        self.act_base_1.apply(set_bn_fix)
        self.act_base_2.apply(set_bn_fix)
        self.act_base_3.apply(set_bn_fix)

