import os
import numpy as np
import json

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from resnet_3D import resnet34
from create_video_id import get_vid_dict
from net_utils import adjust_learning_rate
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, ToTensor, Resize)
from temporal_transforms import LoopPadding
# from model import Model
from model_all_svm import Model
from resize_rpn import resize_rpn, resize_tube
from jhmdb_dataset import Video, video_names

from box_functions import bbox_transform, tube_transform_inv, clip_boxes, tube_overlaps
from mAP_function import calculate_mAP
from plot import plot_tube_with_gt
import imdetect
np.random.seed(42)

def validation(epoch, device, model, dataset_folder, sample_duration, spatial_transform, temporal_transform, boxes_file, splt_txt_path, cls2idx, batch_size, n_threads):

    iou_thresh = 0.5 # Intersection Over Union thresh
    iou_thresh_4 = 0.4 # Intersection Over Union thresh
    iou_thresh_3 = 0.3 # Intersection Over Union thresh

    # confidence_thresh = 0.5
    confidence_thresh = 0.05
    # test_folder_path = './JHMDB-features-256-7-test'
    # test_folder_path = './JHMDB-features-256-mod7-test'
    test_folder_path = './JHMDB-features-256-7ver2-test'
    # test_folder_path = './JHMDB-features-64-7-test'
    vid_name_loader = video_names(dataset_folder, split_txt_path, boxes_file, vid2idx, mode='test', classes_idx=cls2idx, plot=True)
    # data_loader = torch.utils.data.DataLoader(vid_name_loader, batch_size=n_devs, num_workers=8*n_devs, pin_memory=True,
    #                                           shuffle=True)    # reset learning rate
    data_loader = torch.utils.data.DataLoader(vid_name_loader, batch_size=1, num_workers=8*n_devs, pin_memory=True,
                                              shuffle=True)    # reset learning rate
    # data_loader = torch.utils.data.DataLoader(vid_name_loader, batch_size=1, num_workers=8*n_devs, pin_memory=True,
    #                                           shuffle=False)    # reset learning rate

    true_pos = 0
    false_neg = 0

    true_pos_4 = 0
    false_neg_4 = 0

    true_pos_3 = 0
    false_neg_3 = 0
    
    correct = 0
    false   = 0
    correct_all = 0
    false_all   = 0
    correct_per_class = [0 for i in range(22)]
    correct_class = 0
    false_class = 0
    correct_preds = torch.zeros(1).long().to(device)
    n_preds = torch.zeros(1).long().to(device)
    preds = torch.zeros(1).long().to(device)

    rnn_path = './'
    svmWeights = torch.from_numpy(np.loadtxt(os.path.join(rnn_path,'svmWeights.txt')))
    svmBias = torch.from_numpy(np.loadtxt(os.path.join(rnn_path,'svmBias.txt')))
    svmFeatScale = torch.from_numpy(np.loadtxt(os.path.join(rnn_path,'svmFeatScale.txt')))


    for step, data  in enumerate(data_loader):


        # vid_id, clips, boxes, n_frames, n_actions, h, w, target =data
        vid_id, clips, boxes, n_frames, n_actions, h, w, target, clips_plot =data
        vid_id = vid_id.int()
        print('vid_id :',vid_id)
        print('vid_name_loader[vid_id] :',vid_names[vid_id])
        features = torch.load(os.path.join( test_folder_path, vid_names[vid_id], 'feats.pt'))
        labels = torch.load(os.path.join(test_folder_path,vid_names[vid_id], 'labels.pt'))
        scores = imdetect.scoreTubes(features,\
                            svmWeights.type_as(features), svmBias.type_as(features),\
                            svmFeatScale.type_as(features), decisionThreshold=0.5)

        _, cls_int = torch.max(scores[:,],1)
        
        cls_int = cls_int 
        print('labels :',labels)
        print('cls_int :',cls_int)

        # _, cls_int = torch.max(scores[:,1:],1)
        
        # cls_int = cls_int + 1
        # print('labels :',labels)
        # print('cls_int :',cls_int)

        compare = labels == cls_int.type_as(labels)
        if compare.size(0) > 1:
            correct += compare[1:].eq(1).nonzero().nelement()
            false   += compare[1:].eq(0).nonzero().nelement()
            
        correct_class += compare[0].eq(1).nonzero().nelement()
        if compare[0].eq(1).nonzero().nelement() != 0:
            print('---------->labels :',labels, ' and cls_int :',cls_int)
            correct_per_class[labels[0].int()] += 1
        false_class   += compare[0].eq(0).nonzero().nelement()
        correct_all += compare.eq(1).nonzero().nelement()
        false_all += compare.eq(0).nonzero().nelement()

        print('compare :',compare)
        print('correct_all :',correct_all)
        print('false_allex   :',false_all)
        print('correct :',correct)
        print('false   :',false)
        print('correct_class :',correct_class)
        print('false_class   :',false_class)
        print('correct_per_class :',correct_per_class )
        print('correct_per_class :',[[i,correct_per_class] for i,correct_per_class in enumerate(correct_per_class)])



    
if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    dataset_folder = '../JHMDB-act-detector-frames'
    split_txt_path =  '../splits'
    boxes_file = '../poses.json'


    n_devs = torch.cuda.device_count()
    sample_size = 112
    # sample_duration = 4  # len(images)
    sample_duration = 8  # len(images)
    # sample_duration = 16  # len(images)

    batch_size = 1
    n_threads = 0

    # # get mean

    mean = [103.29825354, 104.63845484,  90.79830328]  # jhmdb from .png

    # generate model
    actions = ['__background__','brush_hair', 'clap', 'golf', 'kick_ball', 'pour',
               'push', 'shoot_ball', 'shoot_gun', 'stand', 'throw', 'wave',
               'catch','climb_stairs', 'jump', 'pick', 'pullup', 'run', 'shoot_bow', 'sit',
               'swing_baseball', 'walk' ]

    cls2idx = {actions[i]: i for i in range(0, len(actions))}

    ### get videos id
    vid2idx,vid_names = get_vid_dict(dataset_folder)

    spatial_transform = Compose([Scale(sample_size),  # [Resize(sample_size),
                                 ToTensor(),
                                 Normalize(mean, [1, 1, 1])])
    temporal_transform = LoopPadding(sample_duration)
    validation(0, device, None, dataset_folder, sample_duration, spatial_transform, temporal_transform, boxes_file, split_txt_path, cls2idx, batch_size, n_threads)
