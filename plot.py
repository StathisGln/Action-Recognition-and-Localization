import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch


def plot(name):

    output_name = name[:-4]
    fig_lbl = output_name.split('/')[1]
    output_name = output_name + '.jpg'
    file = open(name)
    lines = file.readlines()
    epochs = []
    recall = []
    for k in range(len(lines)):
        if lines[k].startswith('| Validation'):
            epochs.append(int(lines[k].split()[3]))
            if k + 7 < len(lines):
                recall.append(float(lines[k+7].split()[3]))
    file.close()
    plt.figure()
    plt.title(fig_lbl)
    plt.plot( epochs, recall)
    plt.savefig(output_name)
    


if __name__ == '__main__':

    files = glob.glob('data/validation*.txt')
    print(files)
    for i in files:
        print('i :',i)
        plot(i)
    # output_name = name[:-4]
    # print('output_name:',output_name)
    # file = open('data/validation_2scores_IoU.txt')
    # # lines = file.readlines()
    # # epochs = []
    # # recall = []
    # # for k in range(len(lines)):
    # #     if lines[k].startswith('| Validation'):
    # #         epochs.append(int(lines[k].split()[3]))
    # #         if k + 7 < len(lines):
    # #             recall.append(float(lines[k+7].split()[3]))

    # # print('epochs :',epochs)
    # # print('recall :',recall)
    # # exit(-1)

    
