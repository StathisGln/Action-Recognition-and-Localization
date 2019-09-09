import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import cv2

def validation():

    filename = './training_rnn.txt'
    # filename = './nohup.out'

    epoch = []
    loss = []
    epoch_recall = []
    recall_05 = []
    recall_04 = []
    recall_03 = []
    with open(filename, 'r') as fp:
        lines = fp.readlines()
        for i in range(len(lines)):
            l = lines[i]
            if l.startswith('Train Epoch'):
                words = l.split()

                epoch.append(int(words[2]))
                loss.append(float(words[4]))
            if l.startswith('| Validation Epoch'):
                epoch_recall.append( int(l.split()[3]))
                recall_05.append( float(lines[i+12].split()[3]))
                recall_04.append( float(lines[i+18].split()[3]))
                recall_03.append( float(lines[i+24].split()[3]))

                
            
    # for e,l in zip(epoch,loss):
    #     print('epoch :',e, ' loss :',l)
    print('epoch_recall :',epoch_recall)
    print('epoch :',epoch)
    fig = plt.figure()
    ax = plt.subplot(111, label='1')
    ax.plot(epoch[2:],loss[2:], 'r')
    ax.plot(epoch_recall,recall_05, 'g')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('Loss.png')

    fig = plt.figure()

    ax2 = plt.subplot(111, label='1')
    ax2.plot(epoch_recall,recall_05, 'g')

    plt.savefig('Validation.png')

def plot_tube( clips, n_frames, tube, out_folder='output_vid'):

    ...

def plot_tube_with_gt(clips, n_frames, tube, gt_tube, out_folder='output_vid'):
    
    print('clips.shape :',clips.shape)
    print('gt_tube.shape :',gt_tube.shape)
    print('tube :',tube.shape)
    
    batch_size = clips.size(0)
    num_actions = gt_tube.size(1)
    num_tubes = tube.size(0)
    tube = tube.contiguous().view(num_tubes,-1,4)
    # tube = tube[:,2:].contiguous().view(num_tubes,-1,4)    


    print('tube :',tube.shape)
    # rewrite folder for outputs
    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)

    os.makedirs(out_folder)

    for z in range(num_tubes):
        os.makedirs(os.path.join(out_folder, 'tube_{:0>3}'.format(z)))

    for i in range(batch_size):
        for j in range(n_frames[i]):

            img = clips[i,j].permute(1,2,0).cpu().numpy().copy()
            
            for z in range(num_actions):
                cv2.rectangle(img,(gt_tube[i,z,j,0].int(),gt_tube[i,z,j,1].int()),(gt_tube[i,z,j,2].int(),gt_tube[i,z,j,3].int()),(255,0,0),1)
            for z in range(num_tubes):

                img_= img.copy()
                cv2.rectangle(img_,(tube[z,j,0].int(),tube[z,j,1].int()),(tube[z,j,2].int(),tube[z,j,3].int()),(0,255,0),1)                
                cv2.imwrite(os.path.join(out_folder,'tube_{:0>3}'.format(z),'img_{:0>3}.jpg'.format(j)),img_)

    exit(-1)
if __name__ == '__main__':


    filename = './training_rnn.txt'
    # filename = './nohup.out'

    epoch = []
    loss = []
    epoch_recall = []
    recall_05 = []
    recall_04 = []
    recall_03 = []
    with open(filename, 'r') as fp:
        lines = fp.readlines()
        for i in range(len(lines)):
            l = lines[i]
            if l.startswith('Train Epoch'):
                words = l.split()

                epoch.append(int(words[2]))
                loss.append(float(words[4]))
            if l.startswith('| Validation Epoch'):
                epoch_recall.append( int(l.split()[3]))
                recall_05.append( float(lines[i+12].split()[3]))
                recall_04.append( float(lines[i+18].split()[3]))
                recall_03.append( float(lines[i+24].split()[3]))

                
            
    # for e,l in zip(epoch,loss):
    #     print('epoch :',e, ' loss :',l)
    print('epoch_recall :',epoch_recall)
    print('epoch :',epoch)
    fig = plt.figure()
    ax = plt.subplot(111, label='1')
    ax.plot(epoch[2:],loss[2:], 'r')
    ax.plot(epoch_recall,recall_05, 'g')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('Loss.png')

    fig = plt.figure()

    ax2 = plt.subplot(111, label='1')
    ax2.plot(epoch_recall,recall_05, 'g')

    plt.savefig('Validation.png')

    
