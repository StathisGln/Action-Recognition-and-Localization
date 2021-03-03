import os
import pickle
import json
import torch
import numpy as np
import torch.utils.data as data
from itertools import groupby

from lib.utils.dataloader_utils import  *
from torchvision.transforms import  Compose, Normalize, ToTensor


def get_file_names(mode):

    TRAIN_PEOPLE_ID = {11, 12, 13, 14, 15, 16, 17, 18}
    DEV_PEOPLE_ID = {19, 20, 21, 23, 24, 25, 1, 4}
    TEST_PEOPLE_ID = {22, 2, 3, 5, 6, 7, 8, 9, 10}

    if mode.upper() == 'TRAIN':
        people_id = TRAIN_PEOPLE_ID
    elif mode.upper() == 'VAL':
        people_id = DEV_PEOPLE_ID
    else:
        people_id = TEST_PEOPLE_ID
    return people_id

def make_correct_dataset(dataset_path, boxes_file, split_txt_path, mode='train'):

    dataset = []
    classes = next(os.walk(dataset_path, True))[1]
    assert classes != (None), 'classes must not be None, Check dataset path'

    with open(boxes_file, 'rb') as fp:
        type = os.path.splitext(boxes_file)[1]
        if type.endswith('json'):
            boxes_data = json.load(fp)
        else:
            boxes_data = pickle.load(fp)


    max_sim_actions = -1
    max_frames = -1
    people_ids = get_file_names( mode )

    for vid_name, values in boxes_data.items():
        # print('vid :',vid)
        person_id = int(vid_name.split('_')[0][-2:])
        if person_id not in people_ids:
            continue
        name = vid_name.split('/')[-1]
        n_frames = values['numf']
        annots = values['annotations']
        n_actions = len(annots)

        # find max simultaneous actions
        if n_actions > max_sim_actions:
            max_sim_actions = n_actions

        # find max number of frames
        if n_frames > max_frames:
            max_frames = n_frames

        rois = np.zeros((n_actions, n_frames, 5))
        rois[:, :, 4] = -1
        cls = values['label']

        for k in range(n_actions):
            sample = annots[k]
            s_frame = sample['sf']
            e_frame = sample['ef']
            s_label = sample['label']
            boxes = sample['boxes']
            rois[k, s_frame:e_frame, :4] = boxes
            rois[k, s_frame:e_frame, 4] = s_label

        # name = vid.split('/')[-1].split('.')[0]
        video_sample = {
            'video_name': name,
            'abs_path': os.path.join(dataset_path, values['abs_path']),
            'class': cls,
            'n_frames': n_frames,
            'boxes': rois
        }
        dataset.append(video_sample)

    print('len(dataset) :', len(dataset))
    print('max_sim_actions :', max_sim_actions)
    print('max_frames :', max_frames)
    return dataset, max_sim_actions, max_frames


class Video_Dataset(data.Dataset):

    def __init__(self, video_path, frames_dur=16, split_txt_path=None, sample_size=112,
                 spatial_transform=None, temporal_transform=None, bboxes_file=None, scale=1,
                 get_loader=get_default_video_loader, mode='train', classes_idx=None):

        self.mode = mode
        self.data, self.max_sim_actions, self.max_frames = make_correct_dataset(
                    video_path, bboxes_file, split_txt_path, self.mode)
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = get_loader()
        self.sample_duration = frames_dur
        self.sample_size = sample_size
        self.bboxes_file = bboxes_file
        self.classes_idx = classes_idx
        self.scale = scale

    def __len__(self):

        return  len(self.data)

    def __getitem__(self, index):

        vid_name  = self.data[index]['video_name']
        boxes = self.data[index]['boxes']
        path      = self.data[index]['abs_path']
        class_num = self.data[index]['class']
        n_frames  = self.data[index]['n_frames']

        
        ## get  random frames from the video

        ## TODO change way to get time index
        ## TODO create function to do so

        time_index = np.random.randint(
            0, n_frames - self.sample_duration - 1) + 1
        frame_indices = list(
            range(time_index, time_index + self.sample_duration))  # get the corresponding frames

        clip = self.loader(path, frame_indices)
        w, h = clip[0].size
        clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        ## get bboxes and create gt_tubes
        boxes = preprocess_boxes(boxes, h, w, self.sample_size)
        boxes = boxes[:, np.array(frame_indices), :5]

        final_rois =  get_all_actions_in_current_clip(boxes=boxes, sample_duration=self.sample_duration)

        n_acts = final_rois.shape[0]
        # ret_tubes = torch.zeros(self.max_sim_actions, 7)
        # ret_tubes[:n_acts, :] = gt_tubes

        f_rois = torch.zeros(self.max_sim_actions, self.sample_duration, 5)
        f_rois[:n_acts, :, :] = final_rois[:n_acts]

        im_info = torch.Tensor([self.sample_size, self.sample_size, self.sample_duration])

        sample = {
            'clip' : clip,
            'h' : np.array([h], dtype=np.int),
            'w' : np.array([w], dtype=np.int),
            'gt_tubes' : torch.zeros(1,7),
            'bboxes' : f_rois,
            'n_acts' : np.array([n_acts], dtype=np.int),
            'n_frames' : np.array([n_frames], dtype=np.int),
            'im_info' : im_info,
        }
        return  sample


if __name__ == '__main__':

    # from IPython import embed
    import sys
    sys.path.append('../')
    from utils.get_dataset_mean import get_dataset_mean_and_std

    np.random.seed(42)
    dataset_folder = '../../dataset'
    boxes_file = '../../dataset_actions_annots.json'
    split_txt = '../../00sequences.txt'



    scale = 1
    rev_scale = 255 if scale == 1 else 1
    mean, std = get_dataset_mean_and_std('kth', scale = 1)

    std = (0.5,0.5,0.5)
    mean = (0.5,0.5,0.5)
    print(f'mean {mean}, {std} std')
    spatial_transform = Compose([ToTensor(),
                                 Normalize(mean, std)])
    data = Video_Dataset(video_path=dataset_folder, bboxes_file=boxes_file,
                         split_txt_path=split_txt, spatial_transform=spatial_transform, scale=scale)

    ret = data[5]
    print(f'ret : {ret}')