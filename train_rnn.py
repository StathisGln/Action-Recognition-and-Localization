import os
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from ucf_dataset import  RNN_UCF

from create_video_id import get_vid_dict
from net_utils import adjust_learning_rate

from conf import conf
from act_rnn_wrapper import _RNN_wrapper

np.random.seed(42)


if __name__ == '__main__':


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)
    
    dataset_folder = '../UCF-101-features'
    boxes_file = '../pyannot.pkl'
    split_txt_path = '../UCF101_Action_detection_splits/'

    actions = ['__background__', 'Basketball','BasketballDunk','Biking','CliffDiving','CricketBowling',
               'Diving','Fencing','FloorGymnastics','GolfSwing','HorseRiding','IceDancing',
               'LongJump','PoleVault','RopeClimbing','SalsaSpin','SkateBoarding','Skiing',
               'Skijet','SoccerJuggling','Surfing','TennisSwing','TrampolineJumping',
               'VolleyballSpiking','WalkingWithDog']


    cls2idx = {actions[i]: i for i in range(0, len(actions))}

    ### get videos id
    vid2idx,vid_names = get_vid_dict(dataset_folder)

    # first initialize model
    n_devs = torch.cuda.device_count()

    act_rnn_wrapper =_RNN_wrapper(256,128,len(actions))
    act_rnn_wrapper = nn.DataParallel(act_rnn_wrapper)

    act_rnn_wrapper = act_rnn_wrapper.to(device)

    # init data_loaders
    vid_name_loader = RNN_UCF(dataset_folder, split_txt_path, boxes_file, vid2idx, mode='train')

    data_loader = torch.utils.data.DataLoader(vid_name_loader, batch_size=16*n_devs, num_workers=8*n_devs, pin_memory=True,
                                              shuffle=True)    # reset learning rate

    # data_loader = torch.utils.data.DataLoader(vid_name_loader, batch_size=n_devs, num_workers=0, pin_memory=True,
    #                                           shuffle=False)    # reset learning rate
    
    lr = 0.1
    lr_decay_step = 5
    lr_decay_gamma = 0.1

    # reset learning rate

    params = []
    # for key, value in dict(model.act_rnn.named_parameters()).items():
    for key, value in dict(act_rnn_wrapper.named_parameters()).items():

        # print(key, value.requires_grad)
        if value.requires_grad:
            print('key :',key)
            if 'bias' in key:
                params += [{'params':[value],'lr':lr*(True + 1), \
                            'weight_decay': False and 0.0005 or 0}]
            else:
                params += [{'params':[value],'lr':lr, 'weight_decay': 0.0005}]

    lr = lr * 0.1
    optimizer = torch.optim.Adam(params)

    ##########################
    
    epochs = 40

    for ep in range(epochs):

        act_rnn_wrapper.train()
        loss_temp = 0

        print(' ============\n| Epoch {:0>2}/{:0>2} |\n ============'.format(ep+1, epochs))

        if ep % (lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, lr_decay_gamma)
            lr *= lr_decay_gamma

        for step, data  in enumerate(data_loader):

            # if step == 2:
            #     break

            # print('step :',step)
            f_features, len_tubes, target_lbl  = data

            f_features  = f_features.to(device)
            len_tubes = len_tubes.to(device).long()
            cls_loss = act_rnn_wrapper(f_features, len_tubes, target_lbl)

            loss = cls_loss.mean()

            # backw\ard
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_temp += loss.item()

        print('Train Epoch: {} \tLoss: {:.6f}\t lr : {:.6f}'.format(
        ep+1,loss_temp/step, lr))
        if ( ep + 1 ) % 5 == 0:
            torch.save(act_rnn_wrapper.module.act_rnn.state_dict(), "act_rnn.pwf")
    torch.save(act_rnn_wrapper.module.act_rnn.state_dict(), "act_rnn.pwf")
