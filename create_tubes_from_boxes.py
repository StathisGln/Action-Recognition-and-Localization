import os
import numpy as np
from itertools import groupby
import torch

import json
import cv2


# def create_tube(boxes, im_info_3d):

#     batch_size = boxes.size(0)

#     print(boxes.shape)
#     labels = boxes[:, 0, 0, 5].unsqueeze(1).cpu().numpy()
#     print('labels.shape :', labels.shape)
#     mins, _ = torch.min(boxes, 1)
#     print('mins :', mins)
#     x1 = mins[:, :, 0]
#     y1 = mins[:, :, 1]
#     t1 = mins[:, :, 4]

#     maxs, _ = torch.max(boxes, 1)
#     print('maxs :', maxs)
#     x2 = maxs[:, :, 2]
#     y2 = maxs[:, :, 3]
#     t2 = maxs[:, :, 4]
#     print('t2 :', t2)
#     for i in range(batch_size):
#         x1 = x1.clamp_(min=0, max=im_info_3d[i, 1]-1)
#         y1 = y1.clamp_(min=0, max=im_info_3d[i, 0]-1)
#         t1 = t1.clamp_(min=im_info_3d[i, 2], max=im_info_3d[i, 3])
#         x2 = x2.clamp_(min=0, max=im_info_3d[i, 1]-1)
#         y2 = y2.clamp_(min=0, max=im_info_3d[i, 0]-1)
#         t2 = t2.clamp_(min=im_info_3d[i, 2], max=im_info_3d[i, 3])
#     print('t2 :', t2)
#     x1 = x1[:].cpu().numpy()
#     y1 = y1[:].cpu().numpy()
#     print('t1 :', t1, 'im_info_3d.shape ', im_info_3d.shape)
#     t1 = (t1 - im_info_3d)[:, 2].unsqueeze(1).cpu().numpy()
#     print('t1 :', t1)
#     x2 = x2[:].cpu().numpy()
#     y2 = y2[:].cpu().numpy()
#     print('t2 :', t2, 't2  - im_info_3d ', t2 - im_info_3d)
#     t2 = (t2 - im_info_3d)[:, 2].unsqueeze(1).cpu().numpy()
#     print('t2 :', t2, 'im_info_3d.shape ', im_info_3d.shape)

#     print('x1 {} y1 {} t1 {} x2 {} y2 {} t2 {}'.format(x1, y1, t1, x2, y2, t2))
#     print('shapes :x1 {} y1 {} t1 {} x2 {} y2 {} t2 {}'.format(
#         x1.shape, y1.shape, t1.shape, x2.shape, y2.shape, t2.shape))
#     ret = torch.Tensor([x1, y1, t1, x2, y2, t2, labels]
#                        ).permute(1, 2, 0).cuda()
#     return ret
def create_tube(boxes, im_info_3d, sample_duration):

    # print('boxes.shape :',boxes.shape)
    # print('boxes :',boxes)
    batch_size = boxes.size(0)
    n_actions = boxes.size(1)
    # gt_tubes = torch.Tensor((batch_size, n_actions, 7)).type_as(boxes)
    t1 = torch.zeros(batch_size, n_actions).type_as(boxes)
    t2 = torch.zeros(batch_size, n_actions).type_as(boxes)
    labels = torch.zeros(batch_size, n_actions).type_as(boxes)
    for i in range(batch_size):
        for j in range(boxes.size(1)):
            k = boxes[i,j].nonzero()
            # print(boxes[i,j])
            if k.nelement() == 0 :
                continue
            else:
                # print(k)
                # print(k.shape)
                t1[i,j] = k[0,0]
                t2[i,j] = k[-1,0]
                labels[i,j] = boxes[i,j,k[0,0],4]
                
    mins, _ = torch.min(boxes, 2)
    x1 = mins[:, :, 0]
    y1 = mins[:, :, 1]

    maxs, _ = torch.max(boxes, 2)

    x2 = maxs[:, :, 2]
    y2 = maxs[:, :, 3]
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


    n_actions = boxes.shape[0]
    # print('n_actions :',n_actions)
    boxes = boxes.clamp_(min=0)

    t1 = torch.zeros(( n_actions)).type_as(boxes)
    t2 = torch.zeros(( n_actions)).type_as(boxes)
    x1 = torch.zeros(( n_actions)).type_as(boxes)
    y1 = torch.zeros(( n_actions)).type_as(boxes)
    x2 = torch.zeros(( n_actions)).type_as(boxes)
    y2 = torch.zeros(( n_actions)).type_as(boxes)
    labels = torch.zeros(( n_actions)).type_as(boxes)
    # print('boxes.shape :',boxes.shape)
    for j in range(boxes.shape[0]):
        k = boxes[j].nonzero()
        # print('boxes :',boxes[j])
        # print(k)
        if k[0].size == 0 :
            continue
        else:
            t1[j] = k[0,0]
            t2[j] = k[-1,0]
            labels[j] = boxes[j,k[0,0],4]

            mins,_ = torch.min(boxes[j, k[0,0]:k[-1,0]], 0)
            x1[j] = mins[ 0]
            y1[j] = mins[ 1]

            maxs,_ = torch.max(boxes[j, k[0,0]:k[-1,0]], 0)
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
    #     x1.shape, y1.shape, t1.shape, x2.shape, y2.shape, t2.shape, labels.shape))
    ret = torch.stack((x1, y1, t1, x2, y2, t2, labels)).permute(1,0)
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
    # print('ok')
    # # with open('./boxes.json') as fp:
    # #     data = json.load(fp)

    # # key = 'brush_hair/Trannydude___Brushing_SyntheticHair___OhNOES!__those_fukin_knots!_brush_hair_u_nm_np1_fr_goo_0'
    # # boxes = data[key][4:4+8]
    # # boxes = torch.Tensor(boxes)

    # # tube = create_tube(boxes, 4, 4+8-1)
    # gt_boxes = torch.Tensor([[[[160.7641,  70.0822, 242.5207, 175.3398,   5.0000, 1.]],
    #                           [[161.1543,  70.5410, 242.4840, 175.2963,   6.0000, 1.]],
    #                           [[161.1610,  70.5489, 242.4820, 175.2937,   7.0000, 1.]],
    #                           [[161.0852,  70.4499, 243.3874, 176.4665,   8.0000, 1.]],
    #                           [[161.0921,  70.4580, 243.3863, 176.4650,   9.0000, 1.]],
    #                           [[161.5888,  73.5920, 242.9024, 176.6015,   10.0000, 1.]],
    #                           [[161.5971,  73.5839, 242.9018, 177.6026,   11.0000, 1.]],
    #                           [[161.6053,  73.5757, 242.9014, 177.6040,   12.0000, 1.]]],
    #                          [[[160.7641,  70.0822, 242.5207, 175.3398,   5.0000, 1.]],
    #                           [[161.1543,  70.5410, 242.4840, 175.2963,   6.0000, 1.]],
    #                           [[161.1610,  70.5489, 242.4820, 175.2937,   7.0000, 1.]],
    #                           [[161.0852,  70.4499, 243.3874, 176.4665,   8.0000, 1.]],
    #                           [[161.0921,  70.4580, 243.3863, 176.4650,   2.0000, 1.]],
    #                           [[-1161.5888,  -173.5920, 42424.9024, 14244.6015,   42.0000, 1.]],
    #                           [[161.5971,  73.5839, 242.9018, 177.6026,   11.0000, 1.]],
    #                           [[161.6053,  73.5757, 242.9014, 177.6040,   12.0000, 1.]]]]).cuda()
    # gt_boxes = torch.Tensor([[[[56.4268,  38.4016, 263.4687, 339.3752,  21.0000]],
    #                           [[56.8170,  38.3890, 265.5017, 339.3763,  22.0000]],
    #                           [[56.6460,  38.3759, 267.1082, 339.3838,  23.0000]],
    #                           [[56.4268,  38.4016, 263.4687, 339.3752,  21.0000]],
    #                           [[56.8170,  38.3890, 265.5017, 339.3763,  22.0000]],
    #                           [[56.6460,  38.3759, 267.1082, 339.3838,  23.0000]],
    #                           [[56.4268,  38.4016, 263.4687, 339.3752,  21.0000]],
    #                           [[56.8170,  38.3890, 265.5017, 339.3763,  22.0000]]]]).cuda()
    # print(gt_boxes.shape)
    # # im_info_3d = torch.Tensor([[240., 320.,   20., 28., ]]).cuda()
    im_info_3d = torch.Tensor([240., 320.,    5., 12. ]).cuda()
    print(im_info_3d.shape)
    # # print(gt_boxes.shape)
    # # print(im_info_3d.shape)
    # # tube = create_tube(gt_boxes, im_info_3d)
    # # print(tube)
    # # print(tube.shape)
    # print(len(boxes))

    boxes = [[[99, 105, 143, 184]], [[99, 105, 143, 184]], [[99, 105, 143, 184]],
             [[93, 105, 137, 184]], [[93, 105, 137, 184]], [[93, 105, 137, 184]],
             [[93, 105, 137, 184]], [[88, 100, 132, 179]], [[88, 100, 132, 179]],
             [[83, 100, 127, 179]], [[83, 100, 127, 179]], [[80, 104, 124, 183]],
             [[80, 104, 124, 183]], [[77, 104, 121, 183]], [[75, 105, 119, 184]],
             [[76, 92, 119, 185]], [[76, 92, 119, 185]], [[76, 92, 119, 185]],
             [[75, 91, 118, 184]], [[75, 90, 118, 183]], [[74, 87, 117, 180]],
             [[74, 90, 117, 183]], [[73, 91, 116, 184]], [[71, 92, 114, 185]],
             [[70, 98, 113, 191]], [[70, 101, 113, 194]], [[69, 103, 112, 196]],
             [[64, 108, 107, 201]], [[61, 113, 104, 206]], [[61, 113, 104, 206]],
             [[59, 113, 102, 206]], [[52, 117, 95, 210]], [[52, 117, 95, 210]],
             [[49, 118, 92, 211]], [[45, 122, 88, 215]], [[41, 126, 84, 219]],
             [[41, 128, 84, 221]], [], [], [], [], [], [], [], [], [], [], [],
             [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
             [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
             [], [], [], []]
    # ## file :v_VolleyballSpiking_g19_c03
    action_exist =  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                      1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                      1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     #  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     #  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    # rois = [[[89, 89, 137, 182], [89, 88, 137, 181], [89, 88, 137, 181], [91, 87, 139, 180],
    #          [91, 87, 139, 180], [89, 87, 137, 180], [89, 87, 137, 180], [85, 90, 133, 183],
    #          [85, 90, 133, 183], [82, 92, 130, 185], [82, 92, 130, 185], [79, 95, 127, 188],
    #          [79, 95, 127, 188], [79, 100, 127, 193], [79, 100, 127, 193], [83, 102, 131, 195],
    #          [83, 102, 131, 195], [83, 102, 131, 195], [83, 102, 131, 195], [83, 102, 131, 195],
    #          [92, 104, 140, 197], [92, 104, 140, 197], [90, 108, 138, 201], [90, 108, 138, 201],
    #          [89, 110, 137, 203], [89, 110, 137, 203], [83, 89, 146, 206], [83, 89, 146, 206],
    #          [83, 86, 146, 203], [83, 86, 146, 203], [79, 74, 142, 191], [77, 67, 140, 184],
    #          [77, 67, 140, 184], [70, 64, 133, 181], [70, 64, 133, 181], [64, 63, 127, 180],
    #          [64, 63, 127, 180], [57, 68, 120, 185], [54, 71, 117, 188], [53, 78, 116, 195],
    #          [49, 66, 127, 211], [48, 66, 126, 211], [48, 66, 126, 211], [41, 77, 119, 222],
    #          [41, 77, 119, 222], [40, 83, 118, 228], [40, 83, 118, 228], [38, 88, 116, 233],
    #          [38, 88, 116, 233], [35, 95, 113, 240], [35, 95, 113, 240], [35, 95, 113, 240],
    #          [35, 95, 113, 240], [38, 95, 116, 240], [38, 95, 116, 240], [44, 94, 122, 239],
    #          [51, 92, 129, 237], [51, 92, 129, 237], [66, 163, 138, 237]],
    #         [[134, 88, 179, 168], [135, 88, 180, 168], [135, 88, 180, 168], [135, 88, 180, 168],
    #          [135, 88, 180, 168], [131, 87, 176, 167], [131, 87, 176, 167], [125, 90, 170, 170],
    #          [125, 90, 170, 170], [122, 92, 167, 172], [122, 92, 167, 172], [122, 92, 167, 172],
    #          [122, 92, 167, 172], [119, 91, 164, 171], [119, 91, 164, 171], [115, 95, 160, 175],
    #          [111, 99, 156, 179], [111, 99, 156, 179], [109, 99, 154, 179], [109, 99, 154, 179],
    #          [106, 97, 151, 177], [106, 97, 151, 177], [106, 97, 151, 177], [106, 97, 151, 177],
    #          [103, 99, 148, 179], [103, 99, 148, 179], [102, 104, 147, 184], [102, 104, 147, 184],
    #          [101, 107, 146, 187], [101, 107, 146, 187], [96, 109, 141, 189], [94, 111, 139, 191],
    #          [94, 111, 139, 191], [91, 113, 136, 193], [91, 113, 136, 193], [90, 113, 135, 193],
    #          [90, 113, 135, 193], [82, 121, 127, 201], [82, 121, 127, 201], [74, 128, 119, 208],
    #          [74, 128, 119, 208], [72, 133, 117, 213], [72, 133, 117, 213], [69, 138, 114, 218],
    #          [69, 138, 114, 218], [68, 141, 113, 221], [68, 141, 113, 221], [67, 141, 112, 221],
    #          [66, 141, 111, 221], [66, 141, 111, 221], [65, 141, 110, 221], [65, 141, 110, 221],
    #          [70, 146, 115, 226], [70, 146, 115, 226], [74, 144, 119, 224], [74, 144, 119, 224],
    #          [89, 143, 134, 223], [89, 143, 134, 223], [88, 137, 133, 217]],
    #         [[89, 105, 159, 226], [88, 105, 158, 226], [88, 105, 158, 226], [94, 103, 164, 224],
    #          [94, 103, 164, 224], [88, 76, 161, 219], [87, 76, 160, 219], [87, 76, 160, 219],
    #          [86, 76, 159, 219], [86, 76, 159, 219], [81, 77, 154, 220], [81, 77, 154, 220],
    #          [75, 77, 148, 220], [75, 77, 148, 220], [69, 84, 142, 227], [69, 84, 142, 227],
    #          [65, 88, 138, 231], [65, 88, 138, 231], [63, 88, 136, 231], [63, 88, 136, 231],
    #          [63, 88, 136, 231], [57, 90, 130, 233], [57, 90, 130, 233], [57, 91, 130, 234],
    #          [57, 91, 130, 234], [51, 92, 124, 235], [51, 92, 124, 235], [43, 94, 116, 237],
    #          [43, 94, 116, 237], [38, 95, 111, 238], [38, 95, 111, 238], [37, 95, 110, 238],
    #          [37, 95, 110, 238], [17, 95, 90, 238], [17, 95, 90, 238], [9, 94, 82, 237],
    #          [8, 94, 81, 237], [8, 94, 81, 237], [6, 94, 79, 237], [6, 94, 79, 237],
    #          [0, 94, 73, 237], [0, 94, 73, 237], [0, 118, 59, 238], [0, 118, 59, 238],
    #          [0, 118, 59, 238], [0, 118, 59, 238], [-3, 119, 56, 239], [-3, 119, 56, 239],
    #          [-3, 119, 56, 239], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
    #          [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]
    rois = [[[89, 89, 137, 182, -1], [89, 88, 137, 181, -1], [89, 88, 137, 181, -1], [91, 87, 139, 180, -1],
             [91, 87, 139, 180, -1], [89, 87, 137, 180, -1], [89, 87, 137, 180, -1], [85, 90, 133, 183, -1],
             [85, 90, 133, 183, -1], [82, 92, 130, 185, -1], [82, 92, 130, 185, -1], [79, 95, 127, 188, -1],
             [79, 95, 127, 188, -1], [79, 100, 127, 193, -1], [79, 100, 127, 193, -1], [83, 102, 131, 195, -1],
             [83, 102, 131, 195, -1], [83, 102, 131, 195, -1], [83, 102, 131, 195, -1], [83, 102, 131, 195, -1],
             [92, 104, 140, 197, -1], [92, 104, 140, 197, -1], [90, 108, 138, 201, -1], [90, 108, 138, 201, -1],
             [89, 110, 137, 203, -1], [89, 110, 137, 203, -1], [83, 89, 146, 206, -1], [83, 89, 146, 206, 22],
             [83, 86, 146, 203, 22], [83, 86, 146, 203, 22], [79, 74, 142, 191, 2], [77, 67, 140, 184, 2],
             [77, 67, 140, 184, 2], [70, 64, 133, 181, 2], [70, 64, 133, 181, 22], [64, 63, 127, 180, 22],
             [64, 63, 127, 180, 22], [57, 68, 120, 185, 22], [54, 71, 117, 188, 22], [53, 78, 116, 195, 22],
             [49, 66, 127, 211, 22], [48, 66, 126, 211, 22], [48, 66, 126, 211, 22], [41, 77, 119, 222, 22],
             [41, 77, 119, 222, 22], [40, 83, 118, 228, 22], [40, 83, 118, 228, 22], [38, 88, 116, 233, 22],
             [38, 88, 116, 233, 22], [35, 95, 113, 240, 22], [35, 95, 113, 240, -1], [35, 95, 113, 240, -1],
             [35, 95, 113, 240, -1], [38, 95, 116, 240, -1], [38, 95, 116, 240, -1], [44, 94, 122, 239, -1],
             [51, 92, 129, 237, -1], [51, 92, 129, 237, -1], [66, 163, 138, 237, -1]],
            [[134, 88, 179, 168, -1], [135, 88, 180, 168, -1], [135, 88, 180, 168, -1], [135, 88, 180, 168, -1],
             [135, 88, 180, 168, -1], [131, 87, 176, 167, -1], [131, 87, 176, 167, -1], [125, 90, 170, 170, -1],
             [125, 90, 170, 170, -1], [122, 92, 167, 172, -1], [122, 92, 167, 172, -1], [122, 92, 167, 172, -1],
             [122, 92, 167, 172, -1], [119, 91, 164, 171, -1], [119, 91, 164, 171, -1], [115, 95, 160, 175, -1],
             [111, 99, 156, 179, -1], [111, 99, 156, 179, -1], [109, 99, 154, 179, -1], [109, 99, 154, 179, -1],
             [106, 97, 151, 177, -1], [106, 97, 151, 177, -1], [106, 97, 151, 177, -1], [106, 97, 151, 177, -1],
             [103, 99, 148, 179, -1], [103, 99, 148, 179, -1], [102, 104, 147, 184, -1], [102, 104, 147, 184, -1],
             [101, 107, 146, 187, -1], [101, 107, 146, 187, -1], [96, 109, 141, 189, -1], [94, 111, 139, 191, -1],
             [94, 111, 139, 191, -1], [91, 113, 136, 193, -1], [91, 113, 136, 193, -1], [90, 113, 135, 193, -1],
             [90, 113, 135, 193, -1], [82, 121, 127, 201, -1], [82, 121, 127, 201, -1], [74, 128, 119, 208, -1],
             [74, 128, 119, 208, -1], [72, 133, 117, 213, -1], [72, 133, 117, 213, -1], [69, 138, 114, 218, -1],
             [69, 138, 114, 218, -1], [68, 141, 113, 221, -1], [68, 141, 113, 221, -1], [67, 141, 112, 221, -1],
             [66, 141, 111, 221, -1], [66, 141, 111, 221, -1], [65, 141, 110, 221, -1], [65, 141, 110, 221, -1],
             [70, 146, 115, 226, -1], [70, 146, 115, 226, -1], [74, 144, 119, 224, -1], [74, 144, 119, 224, -1],
             [89, 143, 134, 223, -1], [89, 143, 134, 223, -1], [88, 137, 133, 217, -1]],
            [[89, 105, 159, 226, -1], [88, 105, 158, 226, -1], [88, 105, 158, 226, -1], [94, 103, 164, 224, -1],
             [94, 103, 164, 224, -1], [88, 76, 161, 219, -1], [87, 76, 160, 219, -1], [87, 76, 160, 219, -1],
             [86, 76, 159, 219, -1], [86, 76, 159, 219, -1], [81, 77, 154, 220, -1], [81, 77, 154, 220, -1],
             [75, 77, 148, 220, -1], [75, 77, 148, 220, -1], [69, 84, 142, 227, -1], [69, 84, 142, 227, -1],
             [65, 88, 138, 231, -1], [65, 88, 138, 231, -1], [63, 88, 136, 231, -1], [63, 88, 136, 231, -1],
             [63, 88, 136, 231, -1], [57, 90, 130, 233, -1], [57, 90, 130, 233, -1], [57, 91, 130, 234, -1],
             [57, 91, 130, 234, -1], [51, 92, 124, 235, -1], [51, 92, 124, 235, -1], [43, 94, 116, 237, -1],
             [43, 94, 116, 237, -1], [38, 95, 111, 238, -1], [38, 95, 111, 238, -1], [37, 95, 110, 238, -1],
             [37, 95, 110, 238, -1], [17, 95, 90, 238, -1], [17, 95, 90, 238, -1], [9, 94, 82, 237, -1],
             [8, 94, 81, 237, -1], [8, 94, 81, 237, -1], [6, 94, 79, 237, -1], [6, 94, 79, 237, -1],
             [0, 94, 73, 237, -1], [0, 94, 73, 237, -1], [0, 118, 59, 238, -1], [0, 118, 59, 238, -1],
             [0, 118, 59, 238, -1], [0, 118, 59, 238, -1], [-3, 119, 56, 239, -1], [-3, 119, 56, 239, -1],
             [-3, 119, 56, 239, -1], [0, 0, 0, 0, -1], [0, 0, 0, 0, -1], [0, 0, 0, 0, -1], [0, 0, 0, 0, -1],
             [0, 0, 0, 0, -1], [0, 0, 0, 0, -1], [0, 0, 0, 0, -1], [0, 0, 0, 0, -1], [0, 0, 0, 0, -1],
             [0, 0, 0, 0, -1]]]

    sample_duration = 8
    time_index = 30
    act_idx = 4
    frames = np.array(range(time_index, time_index+ sample_duration))

    # frames = np.array(range(26, 26+sample_duration))
    print(len(frames))
    print(frames)
    rois = torch.Tensor(rois)
    print(rois.shape)
    print(rois[:,frames,:].shape)
    rois = rois[:,frames,:].tolist()
    print(rois)
    rois = [[z+[j] for j,z in enumerate(rois[i])] for i in range(len(rois))]
    print(rois)
    final_rois =[[[list(g),i] for i,g in groupby(w, key=lambda x: x[:][4])] for w in rois] # [person, [action, class]
    # print(len(l))
    # print(len(l[0]),len(l[1]),len(l[2]))
    # print('l[0] :', l[0])
    # print('l[0][0] :', l[0][0])
    # print('l[0][0][0] :', l[0][0][0])
    # print('l[0] :', l[0])
    # print('l[1] :', l[1])
    # print('l[2] :', l[2])
    # tmp_coords = []
    # print(len(action_exist))
    # tmp_rois = [[] for i in range( sample_duration)]
    # print('tmp_rois :', tmp_rois)
    # action_exist_np = np.array(action_exist)

    # final_rois = []
    
    # for i in range(len(action_exist)):
    #     pos = np.where(action_exist_np[i,frames-1]==1)
    #     pos = pos[0]
    #     print(pos, action_exist_np[i,frames-1])
    #     rois_np = np.array(rois[i])
    #     print('pos :',pos)
    #     if len(pos) != 0:
            
    #         # print(rois_np[pos+f], list(enumerate(rois[i])))
    #         tmp_rois = [ rois[i][j] + [j, act_idx] for j in pos]
    #         print(tmp_rois)
    #         # tmp_rois = np.expand_dims(tmp_rois,axis=1)
    #         final_rois.append(tmp_rois)

    # print('final rois:', final_rois)
    ret = create_tube_list(final_rois, im_info_3d, sample_duration)
    # print(ret)
    # for i in each_frame:
    #     print(len(i))
    #     if len(i) > 1:
    #         print('edw perissoteres :', i)
            
# # print(im_info_3d.shape)
# # tube = create_tube(gt_boxes, im_info_3d)
# # print(tube)
# # print(tube.shape)
# print(len(boxes))

# gt_boxes = boxes[4:4+16]
# print(len(gt_boxes))
# print(gt_boxes)
