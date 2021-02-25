import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lib.models.resnet_3D import resnet34
from lib.dataloaders.jhmdb_dataset import Video
from lib.utils.spatial_transforms import (
    Compose, Normalize, Scale, ToTensor)
from lib.utils.temporal_transforms import LoopPadding
from lib.models.region_net import _RPN

np.random.seed(42)

def validation(epoch, device, model, dataset_folder, sample_duration, spatial_transform, temporal_transform, boxes_file, splt_txt_path, cls2idx, batch_size, n_threads):

    iou_thresh = 0.5 # Intersection Over Union thresh
    # iou_thresh = 0.1 # Intersection Over Union thresh
    data = Video_UCF(dataset_folder, frames_dur=sample_duration, spatial_transform=spatial_transform,
                 temporal_transform=temporal_transform, json_file = boxes_file,
                 split_txt_path=splt_txt_path, mode='test', classes_idx=cls2idx)
    data_loader = torch.utils.data.DataLoader(data, batch_size=16,
                                              shuffle=True, num_workers=0, pin_memory=True)
    # data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size*4,
    #                                           shuffle=True, num_workers=0, pin_memory=True)

    model.eval()

    sgl_true_pos = 0
    sgl_false_neg = 0

    ## 2 rois : 1450
    tubes_sum = 0
    for step, data  in enumerate(data_loader):

        # if step == 10:
        #     break
        print('step :',step)

        clips, h, w, gt_tubes_r, gt_rois, n_actions, n_frames, im_info = data
        clips_   = clips.to(device)
        gt_tubes_r_ = gt_tubes_r.to(device)
        gt_rois_ = gt_rois.to(device)
        n_actions_ = n_actions.to(device)
        im_info_ = im_info.to(device)
        start_fr = torch.zeros(clips_.size(0)).to(device)
        # for i in range(2):
        #     print('gt_rois :',gt_rois[i,:n_actions[i]])
        tubes, _, _, _, _, _, _, \
        sgl_rois_bbox_pred, _  = model(clips,
                                       im_info,
                                       None, None,
                                       None)
        tubes_ = tubes.contiguous()
        n_tubes = len(tubes)

        tubes = tubes.view(-1, sample_duration*4+2)

        tubes[:,1:-1] = tube_transform_inv(tubes[:,1:-1],\
                                           sgl_rois_bbox_pred.view(-1,sample_duration*4),(1.0,1.0,1.0,1.0))
        tubes = tubes.view(n_tubes,-1, sample_duration*4+2)
        tubes[:,:,1:-1] = clip_boxes(tubes[:,:,1:-1], im_info, tubes.size(0))

        
        # print('tubes[0]:',tubes.shape)
        # exit(-1)
        # print('tubes.cpu().numpy() :',tubes.cpu().numpy())
        # exit(-1)
        # print('gt_rois_[:,0] :',gt_rois_[:,0])

        for i in range(tubes.size(0)): # how many frames we have
            
            tubes_t = tubes[i,:,1:-1].contiguous()
            gt_rois_t = gt_rois_[i,:,:,:4].contiguous().view(-1,sample_duration*4)

            rois_overlaps = tube_overlaps(tubes_t,gt_rois_t)
            # rois_overlaps = Tube_Overlaps()(tubes_t,gt_rois_t)

            gt_max_overlaps_sgl, max_indices = torch.max(rois_overlaps, 0)

            non_empty_indices =  gt_rois_t.ne(0).any(dim=1).nonzero().view(-1)
            n_elems = non_empty_indices.nelement()            
            # print('non_empty_indices :',non_empty_indices)
            # if gt_tubes_r[i,0,5] - gt_tubes_r[i,0,2 ] < 12 and gt_tubes_r[i,0,5] - gt_tubes_r[i,0,2 ] > 0:
            #     print('tubes_t.cpu().numpy() :',tubes_t[:5].detach().cpu().numpy())
            #     print('sgl_rois_bbox_pred.cpu().numpy() :',sgl_rois_bbox_pred[i,:5].detach().cpu().numpy())
            #     print('tubes_.detach.cpu().numpy() :',tubes_[i,:5].detach().cpu().numpy())
            #     print('gt_rubes_r[i] :',gt_tubes_r[i])
            #     exit(-1)

            if gt_max_overlaps_sgl[0] > 0.5 and gt_rois_t[0,-4:].sum()==0:
                print('max_indices :',max_indices, max_indices.shape, gt_max_overlaps_sgl )
                print('tubes_t[max_indices[0]] :',tubes_t[max_indices[0]])
                print('gt_rois_t[0] :',gt_rois_t[0])

            gt_max_overlaps_sgl = torch.where(gt_max_overlaps_sgl > iou_thresh, gt_max_overlaps_sgl, torch.zeros_like(gt_max_overlaps_sgl).type_as(gt_max_overlaps_sgl))

            sgl_detected =  gt_max_overlaps_sgl[non_empty_indices].ne(0).sum()

            sgl_true_pos += sgl_detected
            sgl_false_neg += n_elems - sgl_detected

        # if step == 0:
        #     break
        #     # exit(-1)


    recall = float(sgl_true_pos)  / (float(sgl_true_pos)  + float(sgl_false_neg))

    print(' -----------------------')
    print('| Validation Epoch: {: >3} | '.format(epoch+1))
    print('|                       |')
    print('| Proposed Action Tubes |')
    print('|                       |')
    print('| Single frame          |')
    print('|                       |')
    print('| In {: >6} steps    :  |'.format(step))
    print('|                       |')
    print('| True_pos   --> {: >6} |\n| False_neg  --> {: >6} | \n| Recall     --> {: >6.4f} |'.format(
        sgl_true_pos, sgl_false_neg, recall))


    print(' -----------------------')

if __name__ == '__main__':

    # torch.cuda.device_count()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    dataset_folder = '/gpu-data/sgal/JHMDB-act-detector-frames'
    splt_txt_path =  '/gpu-data/sgal/splits'
    boxes_file = './poses.json'

    sample_size = 112
    sample_duration = 16  # len(images)

    batch_size = 1
    n_threads = 4

    mean = [103.29825354, 104.63845484,  90.79830328]  # jhmdb from .png

    # generate model
    last_fc = False

    # classes = ['basketballdunk', 'basketballshooting','cliffdiving', 'cricketbowling', 'fencing', 'floorgymnastics',
    #            'icedancing', 'longjump', 'polevault', 'ropeclimbing', 'salsaspin', 'skateboarding',
    #            'skiing', 'skijet', 'surfing', 'biking', 'diving', 'golfswing', 'horseriding',
    #            'soccerjuggling', 'tennisswing', 'trampolinejumping', 'volleyballspiking', 'walking']

    classes = ['brush_hair', 'clap', 'golf', 'kick_ball', 'pour',
               'push', 'shoot_ball', 'shoot_gun', 'stand', 'throw', 'wave',
               'catch','climb_stairs', 'jump', 'pick', 'pullup', 'run', 'shoot_bow', 'sit',
               'swing_baseball', 'walk' ]

    cls2idx = {classes[i]: i for i in range(0, len(classes))}

    spatial_transform = Compose([Scale(sample_size),  # [Resize(sample_size),
                                 ToTensor(),
                                 Normalize(mean, [1, 1, 1])])
    temporal_transform = LoopPadding(sample_duration)

    data = Video(dataset_folder, frames_dur=sample_duration, spatial_transform=spatial_transform,
                 temporal_transform=temporal_transform, json_file = boxes_file,
                 split_txt_path=splt_txt_path, mode='train', classes_idx=cls2idx)
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                              shuffle=True, num_workers=n_threads, pin_memory=True)

    n_classes = len(classes)
    resnet_shortcut = 'A'

    ## ResNet 34 init
    model = resnet34(num_classes=400, shortcut_type=resnet_shortcut,
                     sample_size=sample_size, sample_duration=sample_duration,
                     last_fc=last_fc)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)

    model_data = torch.load('../temporal_localization/resnet-34-kinetics.pth')
    model.load_state_dict(model_data['state_dict'])
    model.eval()

    lr = 0.001

    # Init rpn1
    rpn_model = _RPN(256).cuda()

    params = []
    for key, value in dict(rpn_model.named_parameters()).items():
        # print(key, value.requires_grad)
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params':[value],'lr':lr*(True + 1), \
                            'weight_decay': False and 0.0005 or 0}]
            else:
                params += [{'params':[value],'lr':lr, 'weight_decay': 0.0005}]

    lr = lr * 0.1
    optimizer = torch.optim.Adam(params)

    # epochs = 20
    epochs = 10
    for epoch in range(epochs):
        print(' ============\n| Epoch {:0>2}/{:0>2} |\n ============'.format(epoch, epochs))

        loss_temp = 0
        # start = time.time()
        rpn_model.train()
        model.eval()

        # ## 2 rois : 1450
        # for step, data  in enumerate(data_loader):
        #     # print('&&&&&&&&&&')
        #     print('step -->\t',step)
        #     # clips,  (h, w), gt_tubes, gt_rois = data
        #     clips,  (h, w), gt_tubes_r, n_actions = data
        #     clips = clips.to(device)
        #     gt_tubes_r = gt_tubes_r.to(device)

        #     inputs = Variable(clips)
        #     # print('gt_tubes.shape :',gt_tubes.shape )
        #     # print('gt_rois.shape :',gt_rois.shape)
        #     outputs = model(inputs)
        #     print('outputs: ', outputs[0][0][0][0])
        #     im_info = torch.Tensor([[sample_size, sample_size, sample_duration]] * gt_tubes_r.size(1)).to(device)

        #     rois, rpn_loss_cls, rpn_loss_box = rpn_model(outputs,
        #                                                  im_info,
        #                                                  gt_tubes_r, None, n_actions)
        #     loss = rpn_loss_cls.mean() + rpn_loss_box.mean()
        #     loss_temp += loss.item()

        #     # backw\ard
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()

        # print('loss_temp :',loss_temp)
        # print('Train Epoch: {} \tLoss: {:.6f}\t'.format(
        #     epoch,loss_temp/step))
        # if ( epoch + 1 ) % 5 == 0:
        #     torch.save(rpn_model.state_dict(), "rpn_model_16fr.pwf".format(epoch+1))

        if (epoch + 1) % (5) == 0:

            print(' ============\n| Validation {:0>2}/{:0>2} |\n ============'.format(epoch+1, epochs))
            validation(epoch, device, act_model, dataset_folder, sample_duration, spatial_transform, temporal_transform, boxes_file, split_txt_path, cls2idx, n_devs, 0)

        torch.save(rpn_model.state_dict(), "rpn_model_pre_16fr.pwf".format(epoch))
