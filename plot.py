import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch


def plot(name):

    output_name = name[:-4]
    file = open('data/validation_2scores_IoU.txt')
    lines = file.readlines()
    epochs = []
    recall = []
    for k in range(len(lines)):
        if lines[k].startswith('| Validation'):
            epochs.append(int(lines[k].split()[3]))
            if k + 7 < len(lines):
                recall.append(float(lines[k+7].split()[3]))

    print('epochs :',epochs)
    print('recall :',recall)
    exit(-1)

    




if __name__ == '__main__':

    name = 'data/validation_2scores_IoU.txt'
    plot(name)
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

    
