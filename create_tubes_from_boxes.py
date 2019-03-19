import os
import numpy as np
from itertools import groupby
import torch

import json
import cv2
def create_tube(boxes, im_info_3d, sample_duration):

    # print('boxes.shape :',boxes.shape)
    # print('boxes :',boxes)
    batch_size = boxes.size(0)
    n_actions = boxes.size(1)
    # gt_tubes = torch.Tensor((batch_size, n_actions, 7)).type_as(boxes)
    t1 = torch.zeros(batch_size, n_actions).type_as(boxes)
    t2 = torch.zeros(batch_size, n_actions).type_as(boxes)
    x1 = torch.zeros(batch_size, n_actions).type_as(boxes)
    y1 = torch.zeros(batch_size, n_actions).type_as(boxes)
    x2 = torch.zeros(batch_size, n_actions).type_as(boxes)
    y2 = torch.zeros(batch_size, n_actions).type_as(boxes)
    labels = torch.zeros(batch_size, n_actions).type_as(boxes)
    for i in range(batch_size):
        for j in range(boxes.size(1)):
            k = boxes[i,j].gt(0).nonzero()
            # print('k :', k)
            # print(boxes[i,j])
            if k.nelement() == 0 :
                continue
            else:
                # print(k)
                # print(k.shape)
                t1[i,j] = k[0,0]
                t2[i,j] = k[-1,0]
                labels[i,j] = boxes[i,j,k[0,0],4]

                mins, _ = torch.min(boxes[i,j,k[0,0]:k[-1,0]+1], 0)
                x1[i,j] = mins[0]
                y1[i,j] = mins[1]

                maxs, _ = torch.max(boxes[i,j,k[0,0]:k[-1,0]+1], 0)

                x2[i,j] = maxs[2]
                y2[i,j] = maxs[3]
    # print(im_info_3d)
    for i in range(batch_size):
        x1 = x1.clamp_(min=0, max=im_info_3d[i, 1]-1)
        y1 = y1.clamp_(min=0, max=im_info_3d[i, 0]-1)
        x2 = x2.clamp_(min=0, max=im_info_3d[i, 1]-1)
        y2 = y2.clamp_(min=0, max=im_info_3d[i, 0]-1)

    # print('x1 {} y1 {} t1 {} x2 {} y2 {} t2 {}'.format(x1, y1, t1, x2, y2, t2))
    # print('shapes :x1 {} y1 {} t1 {} x2 {} y2 {} t2 {} labels {}'.format(
    #     x1.shape, y1.shape, t1.shape, x2.shape, y2.shape, t2.shape, labels.shape))
    ret = torch.stack((x1, y1, t1, x2, y2, t2, labels)).permute(1,2,0).type_as(boxes)

    # print('ret.shape :',ret.shape)
    # print('ret :',ret)
    return ret

def create_tube_with_frames(boxes, im_info_3d, sample_duration):

    # print('boxes.shape :',boxes.shape)
    batch_size = boxes.size(0)
    n_actions = boxes.size(1)
    # print('boxes :',boxes.cpu().numpy())
    t1 = torch.zeros(batch_size, n_actions).type_as(boxes)
    t2 = torch.zeros(batch_size, n_actions).type_as(boxes)
    x1 = torch.zeros(batch_size, n_actions).type_as(boxes)
    y1 = torch.zeros(batch_size, n_actions).type_as(boxes)
    x2 = torch.zeros(batch_size, n_actions).type_as(boxes)
    y2 = torch.zeros(batch_size, n_actions).type_as(boxes)
    labels = torch.zeros(batch_size, n_actions).type_as(boxes)
    for i in range(batch_size):
        for j in range(n_actions):
            k = boxes[i,j,:,:4].gt(0).nonzero()
            if k.nelement() == 0 :
                continue
            else:

                labels[i,j] = boxes[i,j,k[0,0],4]
                # if labels[i,j] == -1:
                #     print('boxes[i,j] :',boxes[i,j])
                mins, _ = torch.min(boxes[i,j,k[0,0]:k[-1,0]+1], 0)

                x1[i,j] = mins[0]
                y1[i,j] = mins[1]
                t1[i,j] = mins[-1]
                
                maxs, _ = torch.max(boxes[i,j,k[0,0]:k[-1,0]+1], 0)
                # print('boxes[i,j,k[0,0]:k[-1,0]+1] :',boxes[i,j,k[0,0]:k[-1,0]+1])
                # print('maxs :',maxs)
                x2[i,j] = maxs[2]
                y2[i,j] = maxs[3]
                t2[i,j] = maxs[-1]


    for i in range(batch_size):
        x1 = x1.clamp_(min=0, max=im_info_3d[i, 1]-1)
        y1 = y1.clamp_(min=0, max=im_info_3d[i, 0]-1)
        x2 = x2.clamp_(min=0, max=im_info_3d[i, 1]-1)
        y2 = y2.clamp_(min=0, max=im_info_3d[i, 0]-1)

    # print('x1 {} y1 {} t1 {} x2 {} y2 {} t2 {}'.format(x1, y1, t1, x2, y2, t2))
    # print('shapes :x1 {} y1 {} t1 {} x2 {} y2 {} t2 {} labels {}'.format(
    #     x1.shape, y1.shape, t1.shape, x2.shape, y2.shape, t2.shape, labels.shape))
    ret = torch.stack((x1, y1, t1, x2, y2, t2, labels)).permute(1,2,0).type_as(boxes)

    padding_lines = ret[:,:,-1].lt(1).nonzero()

    for i in padding_lines:
        ret[i[0],i[1]] = torch.zeros((7)).type_as(ret)

    # print('ret.shape :',ret.shape)
    # print('ret :',ret)
    return ret

def create_tube_with_frames_np(boxes, im_info_3d, sample_duration):

    batch_size = boxes.shape[0]
    n_actions = boxes.shape[1]

    t1 = np.zeros((batch_size, n_actions))
    t2 = np.zeros((batch_size, n_actions))
    x1 = np.zeros((batch_size, n_actions))
    y1 = np.zeros((batch_size, n_actions))
    x2 = np.zeros((batch_size, n_actions))
    y2 = np.zeros((batch_size, n_actions))
    labels = np.zeros((batch_size, n_actions))
    for i in range(batch_size):
        for j in range(boxes.shape[1]):
            k = np.where(boxes[i,j,:,:4] > (0))

            if k[0].size == 0 :
                continue
            else:
                labels[i,j] = boxes[i,j,k[0][0],4]
                # if labels[i,j] == -1:
                #     print('boxes[i,j] :',boxes[i,j])
                mins = np.min(boxes[i,j,k[0][0]:k[0][-1]+1], 0)

                x1[i,j] = mins[0]
                y1[i,j] = mins[1]
                t1[i,j] = mins[-1]
                
                maxs = np.max(boxes[i,j,k[0][0]:k[0][-1]+1], 0)

                x2[i,j] = maxs[2]
                y2[i,j] = maxs[3]
                t2[i,j] = maxs[-1]
                
    for i in range(batch_size):
        x1 = np.clip(x1, 0, im_info_3d[i, 1]-1)
        y1 = np.clip(y1, 0, im_info_3d[i, 0]-1)
        x2 = np.clip(x2, 0, im_info_3d[i, 1]-1)
        y2 = np.clip(y2, 0, im_info_3d[i, 0]-1)

    # print('x1 {} y1 {} t1 {} x2 {} y2 {} t2 {}'.format(x1, y1, t1, x2, y2, t2))
    # print('shapes :x1 {} y1 {} t1 {} x2 {} y2 {} t2 {} labels {}'.format(
    #     x1.shape, y1.shape, t1.shape, x2.shape, y2.shape, t2.shape, labels.shape))
    ret = np.concatenate((x1, y1, t1, x2, y2, t2, labels))
    ret = np.transpose(ret, (1,0))
    # print('ret.shape :',ret.shape)
    # print('ret :',ret)

    return ret


def create_tube_from_tubes(boxes):

    mins, _ = torch.min(boxes, 0)
    x1 = mins[0]
    y1 = mins[1]
    t1 = mins[2]

    maxs, _ = torch.max(boxes, 0)
    x2 = maxs[3]
    y2 = maxs[4]
    t2 = maxs[5]
    
    # print('x1 {} y1 {} t1 {} x2 {} y2 {} t2 {}'.format(x1, y1, t1, x2, y2, t2))
    # print('shapes :x1 {} y1 {} t1 {} x2 {} y2 {} t2 {} labels {}'.format(
    #     x1.shape, y1.shape, t1.shape, x2.shape, y2.shape, t2.shape, labels.shape))
    ret = torch.stack((x1, y1, t1, x2, y2, t2)).unsqueeze(0).type_as(boxes)

    return ret


def create_video_tube(boxes):

    batch_size = boxes.size(0)
    n_actions = boxes.size(1)
    # print('n_actions :',n_actions)
    boxes = boxes.clamp_(min=0)

    t1 = torch.zeros((batch_size, n_actions)).type_as(boxes)
    t2 = torch.zeros((batch_size, n_actions)).type_as(boxes)
    x1 = torch.zeros((batch_size, n_actions)).type_as(boxes)
    y1 = torch.zeros((batch_size, n_actions)).type_as(boxes)
    x2 = torch.zeros((batch_size, n_actions)).type_as(boxes)
    y2 = torch.zeros((batch_size, n_actions)).type_as(boxes)
    labels = torch.zeros((batch_size, n_actions)).type_as(boxes)
    # print('boxes.shape :',boxes.shape)
    for i in range(batch_size):
        for j in range(n_actions):
            k = boxes[i,j,:,:4].nonzero()
            # print('boxes[i,j,:,:4] :',boxes[i,j,:,:4])
            # print('boxes[j,:,:4 :',boxes[j,:,:4])
            # print('k :',k)
            # print('k.shape :',k.shape)
            if k.nelement() == 0 :
                continue
            else:
                t1[i,j] = k[0,0]
                t2[i,j] = k[-1,0]
                labels[i,j] = boxes[i,j,k[0,0],4]
                # print('k[0,0] :',k[0,0])
                # print('k[-1,0] :',k[-1,0])
                mins,_ = torch.min(boxes[i,j, k[0,0]:k[-1,0]], 0)
                # print('mins :',mins)
                # print('mins.shape :',mins.shape)
                x1[i,j] = mins[ 0]
                y1[i,j] = mins[ 1]

                maxs,_ = torch.max(boxes[i,j, k[0,0]:k[-1,0]], 0)
                x2[i,j] = maxs[ 2]
                y2[i,j] = maxs[ 3]

    # x1 = x1.clip(0, w-1)
    # y1 = y1.clip(0, h-1)
    # x2 = x2.clip(0, w-1)
    # y2 = y2.clip(0, h-1)
    # t1 = t1.clip(0, sample_duration)
    # t2 = t2.clip(0, sample_duration)
    # print('x1 {} y1 {} t1 {} x2 {} y2 {} t2 {}'.format(x1, y1, t1, x2, y2, t2))
    # print('shapes :x1 {} y1 {} t1 {} x2 {} y2 {} t2 {} labels {}'.format(
    #     x1.shape, y1.shape, t1.shape, x2.shape, y2.shape, t2.shape, labels.shape))
    ret = torch.stack((x1, y1, t1, x2, y2, t2, labels)).permute(1,2,0)
    # print('ret.type() :',ret.type())
    # print('ret.shape :',ret.shape)
    # print('ret :',ret)
    return ret


def create_video_tube_numpy(boxes):

    # print('boxes.shape :',boxes.shape)
    n_actions = boxes.shape[0]

    boxes[np.where(boxes == -1)] = 0

    t1 = np.zeros(( n_actions), dtype=np.int)
    t2 = np.zeros(( n_actions), dtype=np.int)
    x1 = np.zeros(( n_actions), dtype=np.int)
    y1 = np.zeros(( n_actions), dtype=np.int)
    x2 = np.zeros(( n_actions), dtype=np.int)
    y2 = np.zeros(( n_actions), dtype=np.int)
    labels = np.zeros(( n_actions))
    # print('boxes.shape :',boxes.shape)
    for j in range(boxes.shape[0]):
        k = boxes[j].nonzero()
        # print('boxes :',boxes[j])
        # print(k)
        if k[0].size == 0 :
            continue
        else:
            t1[j] = k[0][0]
            t2[j] = k[0][-1]
            labels[j] = boxes[j,k[0][0],4]
            # print('t1[j] :',t1[j])
            # print('t2[j] :',t2[j])
            # print('boxes[j, t1[j]:t2[j]+1] :',boxes[j, t1[j]:t2[j]+1])
            mins = np.min(boxes[j, t1[j]:t2[j]+1], 0)
            x1[j] = mins[ 0]
            y1[j] = mins[ 1]

            maxs = np.max(boxes[j, t1[j]:t2[j]+1], 0)
            x2[j] = maxs[ 2]
            y2[j] = maxs[ 3]

    # x1 = x1.clip(0, w-1)
    # y1 = y1.clip(0, h-1)
    # x2 = x2.clip(0, w-1)
    # y2 = y2.clip(0, h-1)
    # t1 = t1.clip(0, sample_duration)
    # t2 = t2.clip(0, sample_duration)
    # print('x1 {} y1 {} t1 {} x2 {} y2 {} t2 {}'.format(x1, y1, t1, x2, y2, t2))
    # print('shapes :x1 {} y1 {} t1 {} x2 {} y2 {} t2 {} labels {}'.format(
        # x1.shape, y1.shape, t1.shape, x2.shape, y2.shape, t2.shape, labels.shape))
    ret = np.stack((x1, y1, t1, x2, y2, t2, labels)).transpose(1,0)
                       
    # print('ret.shape :',ret.shape)
    # print('ret :',ret)
    return ret


def create_tube_numpy(boxes, im_info_3d, sample_duration):

    batch_size = boxes.shape[0]
    n_actions = boxes.shape[1]

    t1 = np.zeros((batch_size, n_actions))
    t2 = np.zeros((batch_size, n_actions))
    labels = np.zeros((batch_size, n_actions))
    # print('boxes.shape :',boxes.shape)
    for i in range(batch_size):
        for j in range(boxes.shape[1]):
            k = boxes[i,j].nonzero()
            # print('boxes :',boxes.tolist())
            # print(k)
            if k[0].size == 0 :
                continue
            else:
                t1[i,j] = k[0][0]
                t2[i,j] = k[0][-1]
                # print('t1 :', t1, ' t2 :',t2)
                labels[i,j] = boxes[i,j,k[0][0],4]

    mins = np.min(boxes, 2)
    x1 = mins[:, :, 0]
    y1 = mins[:, :, 1]

    maxs  = np.max(boxes, 2)

    x2 = maxs[:, :, 2]
    y2 = maxs[:, :, 3]
    # print(im_info_3d)
    for i in range(batch_size):
        x1 = x1.clip(0, im_info_3d[i, 1]-1)
        y1 = y1.clip(0, im_info_3d[i, 0]-1)
        x2 = x2.clip(0, im_info_3d[i, 1]-1)
        y2 = y2.clip(0, im_info_3d[i, 0]-1)

    # print('x1 {} y1 {} t1 {} x2 {} y2 {} t2 {}'.format(x1, y1, t1, x2, y2, t2))
    # print('shapes :x1 {} y1 {} t1 {} x2 {} y2 {} t2 {} labels {}'.format(
    #     x1.shape, y1.shape, t1.shape, x2.shape, y2.shape, t2.shape, labels.shape))
    ret = np.stack((x1, y1, t1, x2, y2, t2, labels)).transpose(1,2,0)
                       
    # print('ret.shape :',ret.shape)
    # print('ret :',ret)
    return ret



def create_tube_list(rois, im_info_3d, sample_duration):

    
    ret = []
    for p in range(len(rois)): # check for each one person seperately
        p_rois = rois[p]
        for act in p_rois:
            act_rois = act[0]
            act_lbl = act[1]
            if act_lbl < 0: # No action:
                continue
            ## act to torch
            act_tensor = torch.Tensor(act_rois)
            labels = torch.Tensor([act_lbl])

            mins, _ = torch.min(act_tensor, 0)

            x1 = mins[0]
            y1 = mins[1]
            t1 = mins[5]

            x1 = x1.clamp_(min=0, max=im_info_3d[1]-1)
            y1 = y1.clamp_(min=0, max=im_info_3d[0]-1)
            t1 = t1.clamp_(min=0, max=sample_duration-1)

            maxs, _ = torch.max(act_tensor, 0)

            x2 = maxs[2]
            y2 = maxs[3]
            t2 = maxs[5]

            x2 = x2.clamp_(min=0, max=im_info_3d[1]-1)
            y2 = y2.clamp_(min=0, max=im_info_3d[0]-1)
            t2 = t2.clamp_(min=0, max=sample_duration-1)

            x1 = x1.cpu().numpy().tolist()
            y1 = y1.cpu().numpy().tolist()
            t1 = t1.cpu().numpy().tolist()
            x2 = x2.cpu().numpy().tolist()
            y2 = y2.cpu().numpy().tolist()
            t2 = t2.cpu().numpy().tolist()


            # print('x1 {} y1 {} t1 {} x2 {} y2 {} t2 {} label {}'.format(
            #     x1, y1, t1, x2, y2, t2,labels))
            # print('shapes :x1 {} y1 {} t1 {} x2 {} y2 {} t2 {}'.format(
            #     x1.shape, y1.shape, t1.shape, x2.shape, y2.shape, t2.shape))
            tmp_ret = [x1, y1, t1, x2, y2, t2, labels]
            ret.append(tmp_ret)
    ret = torch.Tensor(ret)
    return ret


if __name__ == "__main__":

    t = torch.Tensor([[[  0.,   0.,   0.,   0.,  -1.],
                       [  0.,   0.,   0.,   0.,  -1.],
                       [  0.,   0.,   0.,   0.,  -1.],
                       [  0.,   0.,   0.,   0.,  -1.],
                       [  0.,   0.,   0.,   0.,  -1.],
                       [  0.,   0.,   0.,   0.,  -1.],
                       [  0.,   0.,   0.,   0.,  -1.],
                       [  0.,   0.,   0.,   0.,  -1.],
                       [  0.,   0.,   0.,   0.,  -1.],
                       [ 50.,  19.,  72.,  77.,  14.],
                       [ 50.,  19.,  72.,  77.,  14.],
                       [ 50.,  19.,  72.,  77.,  14.],
                       [ 50.,  19.,  72.,  77.,  14.],
                       [ 50.,  19.,  72.,  77.,  14.],
                       [ 50.,  19.,  72.,  77.,  14.],
                       [ 50.,  19.,  72.,  77.,  14.]],
                       [[  0.,   0.,   0.,   0.,  -1.],
                       [  0.,   0.,   0.,   0.,  -1.],
                       [  0.,   0.,   0.,   0.,  -1.],
                       [  0.,   0.,   0.,   0.,  -1.],
                       [  0.,   0.,   0.,   0.,  -1.],
                       [  0.,   0.,   0.,   0.,  -1.],
                       [  0.,   0.,   0.,   0.,  -1.],
                       [  0.,   0.,   0.,   0.,  -1.],
                       [  0.,   0.,   0.,   0.,  -1.],
                       [ 50.,  19.,  72.,  77.,  14.],
                       [ 50.,  19.,  72.,  77.,  14.],
                       [ 50.,  19.,  72.,  77.,  14.],
                       [ 50.,  19.,  72.,  77.,  14.],
                       [ 50.,  19.,  72.,  77.,  14.],
                       [ 50.,  19.,  72.,  77.,  14.],
                       [ 50.,  19.,  72.,  77.,  14.]]]).cuda()
    print(t.shape)

    tube = create_tube(t.unsqueeze(0), torch.Tensor([[112,112]]), 16)
    print('tube :',tube)
