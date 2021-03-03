'''
A file containing usefull functions to be used during loading data
'''

import os
import numpy as np
import functools
import itertools
from PIL import Image
import torch
# from lib.utils.resize_rpn import resize_boxes_np
from .resize_rpn import resize_boxes_np


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        raise('No accimage available, please implement accimage image loader...')
    else:
        return pil_loader

def video_loader(video_dir_path, frame_indices, image_loader, image_temp='image_{:05d}.jpg'):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, image_temp.format(i))
        # image_path = os.path.join(video_dir_path, '{:05d}.jpg'.format(i))
        # image_path = os.path.join(video_dir_path, '{:05d}.png'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            print('image_path {} doesn\'t exist'.format(image_path))
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def preprocess_boxes(boxes, h, w, sample_size, add_frames=False):

    for i in range(boxes.shape[0]):
        boxes[i] = resize_boxes_np(np.expand_dims( boxes[i], axis=0), h,w,sample_size)

    if add_frames:

        fr_tensor = np.expand_dims( np.expand_dims( np.arange(0,boxes.shape[-2]), axis=1), axis=0)
        fr_tensor = np.repeat(fr_tensor, boxes.shape[0], axis=0)
        boxes = np.concatenate((boxes, fr_tensor), axis=-1)

    return boxes

def get_all_actions_in_current_clip(boxes, sample_duration):

    ### code for getting all seperate actions
    rois_sample = boxes.tolist()
    rois_fr = [[z + [j] for j, z in enumerate(rois_sample[i])] for i in range(len(rois_sample))]
    rois_gp = [[[list(g), i] for i, g in itertools.groupby(w, key=lambda x: x[:][4])] for w in
               rois_fr]  # [person, [action, class]

    final_rois_list = []
    for p in rois_gp:  # for each person
        for r in p:
            if r[1] > -1:
                final_rois_list.append(r[0])

    num_actions = len(final_rois_list)

    if num_actions == 0:
        final_rois = torch.zeros(1, sample_duration, 5)
        gt_tubes = torch.zeros(1, 7)
    else:
        final_rois = torch.zeros((num_actions, sample_duration, 5))  # num_actions x [x1,y1,x2,y2,label]
        for i in range(num_actions):
            # for every action:
            for j in range(len(final_rois_list[i])):
                # for every rois
                # print('final_rois_list[i][j][:5] :',final_rois_list[i][j][:5])
                pos = final_rois_list[i][j][5]
                final_rois[i, pos, :] = torch.Tensor(final_rois_list[i][j][:5])

        # # print('final_rois :',final_rois)
        # gt_tubes = create_tube_list(rois_gp, [w, h],
        #                             self.sample_duration)  ## problem when having 2 actions simultaneously
        #
    return final_rois