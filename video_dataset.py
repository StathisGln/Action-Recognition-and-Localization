import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import os
import math
import functools
import copy
import glob
import json
from itertools import groupby
from create_tubes_from_boxes import create_tube_list

np.random.seed(42)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        import accimage
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
        # image_path = os.path.join(video_dir_path, '{:05d}.jpg'.format(i))
        # image_path = os.path.join(video_dir_path, '{:05d}.png'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            if subset == 'testing':
                video_names.append('test/{}'.format(key))
            else:
                label = value['annotations']['label']
                video_names.append('{}/{}'.format(label, key))
                annotations.append(value['annotations'])

    return video_names, annotations


def create_tcn_dataset(split_txt_path, json_path, classes, mode):

    videos = []
    dataset = []
    txt_files = glob.glob(split_txt_path+'/*1.txt')  # 1rst split
    for txt in txt_files:
        class_name = txt.split('/')[-1][:-16]
        class_idx = classes.index(class_name)
        with open(txt, 'r') as fp:
            lines = fp.readlines()
        for l in lines:
            spl = l.split()
            if spl[1] == '1' and mode == 'train':  # train video
                vid_name = spl[0][:-4]
                videos.append(vid_name)
            elif spl[1] == '2' and mode == 'test':  # train video
                vid_name = spl[0][:-4]
                videos.append(vid_name)

        with open(os.path.join(json_path, class_name+'.json'), 'r') as fp:
            data = json.load(fp)
        for feat in data.keys():
            if feat in videos:
                sample = {
                    'video': feat,
                    'class': class_name,
                    'class_idx': class_idx
                }
                dataset.append(sample)
    print(len(dataset))
    return dataset


def make_dataset(dataset_path,  boxes_file, mode='train'):
    dataset = []
    classes = next(os.walk(dataset_path, True))[1]

    with open(boxes_file, 'r') as fp:
        boxes_data = json.load(fp)

    assert classes != (None), 'classes must not be None, Check dataset path'

    for vid, data in boxes_data.items():
        # print(vid)
        f_coords = data['f_coords']
        each_frame = data['each_frame']
        rois = data['rois']
        action_exist = data['action_exist']
        cls = data['class']
        n_frames = data['n_frames']
        name = vid.split('/')[-1].split('.')[0]
        video_sample = {
            'video_name' : name,
            'abs_path' : os.path.join(dataset_path, cls, name),
            'class' : cls,
            'n_frames' : n_frames,
            'f_coords' : f_coords,
            'action_exist' : action_exist,
            'each_frame' : each_frame,
            'rois' : rois
            
            }
        dataset.append(video_sample)

    print('len(dataset) :',len(dataset))
    return dataset
            
            
            

    # for idx, cls in enumerate(classes):

    #     class_path = os.path.join(dataset_path, cls)
    #     videos = []
    #     # # for each video
    #     # videos =  next(os.walk(class_path,True))[1]
    #     for vid in videos:

    #         video_path = os.path.join(dataset_path, cls, vid)
    #         n_frames = len(glob.glob(video_path+'/*.png'))
    #         begin_t = 1
    #         end_t = n_frames

    #         video_sample = {
    #             'video': vid,
    #             'class': cls,
    #             'abs_path': video_path,
    #             'begin_t': begin_t,
    #             'end_t': n_frames

    #         }

    #         dataset.append(video_sample)
    print(len(dataset))
    return dataset


class Video(data.Dataset):
    def __init__(self, video_path, frames_dur=8, 
                 spatial_transform=None, temporal_transform=None, json_file=None,
                 sample_duration=16, get_loader=get_default_video_loader, mode='train', classes_idx=None):

        self.mode = mode
        self.data = make_dataset(
            video_path, json_file, self.mode)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = get_loader()
        self.sample_duration = frames_dur
        self.json_file = json_file
        self.classes_idx = classes_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        # print(self.data[index]['video'])
        name = self.data[index]['video_name']   # video path
        cls  = self.data[index]['class']
        path = self.data[index]['abs_path']
        rois = self.data[index]['rois']
        action_exist = self.data[index]['action_exist']
        print(name)
        # print('action_exist: ',action_exist)
        # print('rois :', rois)
        n_frames = self.data[index]['n_frames']
        each_frame = self.data[index]['each_frame']

        # print(list(enumerate(each_frame)))
        # print(each_frame)
        ## get 8 random frames from the video
        time_index = np.random.randint(
            0, n_frames - self.sample_duration - 1) + 1

        frame_indices = list(
            range(time_index, time_index + self.sample_duration))  # get the corresponding frames

        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices)
        # print('clip.shape :',clip[0].shape )
        # print(len(clip))
        # print(clip[0].size)

        ## get original height and width
        w, h = clip[0].size
        if self.spatial_transform is not None:
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        
        ## get bboxes and create gt tubes
        rois_Tensor = torch.Tensor(rois)
        print('rois_Tensor.shape :', rois_Tensor.shape)
        rois_sample = rois_Tensor[:,np.array(frame_indices),:].tolist()
        
        rois_fr = [[z+[j] for j,z in enumerate(rois_sample[i])] for i in range(len(rois_sample))]
        
        rois_gp =[[[list(g),i] for i,g in groupby(w, key=lambda x: x[:][4])] for w in rois_fr] # [person, [action, class]

        final_rois = []
        for p in rois_gp: # for each person
            for r in p:
                if r[1] > -1 :
                    final_rois.append(r)
        # print('frame_indices :', frame_indices)
        # print('final_rois :', final_rois)
        
        gt_tubes = create_tube_list(rois_gp,[w,h], self.sample_duration)
        
        # class_int = self.classes_idx[cls.lower().replace('_','').replace(' ','')]

        print(gt_tubes)
        if final_rois == []: # empty ==> only background, no gt_tube exists
            print('rois_gp :',rois_gp)
            final_rois = [0] * 7
            gt_tubes = torch.Tensor([[0,0,0,0,0,0,0]])
        print(' final_rois: ', final_rois)
        print('gt_tubes :', gt_tubes )
        print('Sending...')
        if self.mode == 'train':
            return clip,  (h, w),  gt_tubes, final_rois
        else:
            return clip,  (h, w),  gt_tubes, final_rois,  self.data[index]['abs_path']

    def __len__(self):
        return len(self.data)


class TCN_Dataset(data.Dataset):
    def __init__(self, split_txt_path, json_path, classes, mode):

        self.split_txt_path = split_txt_path
        self.json_path = json_path
        self.mode = mode
        if mode == 'train':
            self.data = create_tcn_dataset(
                split_txt_path, json_path, classes, mode)
        elif mode == 'test':
            self.data = create_tcn_dataset(
                split_txt_path, json_path, classes, mode)
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (features, target) where target is class_index of the target class.
        """
        path = self.data[index]['video']
        target = self.data[index]['class_idx']
        class_name = self.data[index]['class']
        with open(os.path.join(self.json_path, class_name+'.json'), 'r') as fp:
            data = json.load(fp)
        feats = data[path]['clips']
        final_feats = np.zeros((len(feats), 512))
        for f in range(len(feats)):
            final_feats[f, :] = feats[f]['features']

        return final_feats, path, target

    def __len__(self):
        return len(self.data)


# def check_boxes:

#     for
class Pics(data.Dataset):
    def __init__(self, video_path, frames_dur=8, split_txt_path=None,
                 spatial_transform=None, temporal_transform=None, json_file=None,
                 sample_duration=16, get_loader=get_default_video_loader, mode='train', classes_idx=None):

        self.mode = mode
        self.data = make_dataset(
            video_path, split_txt_path, json_file, self.mode)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = get_loader()
        self.sample_duration = frames_dur
        self.json_file = json_file
        self.classes_idx = classes_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        # print(self.data[index]['video'])
        path = self.data[index]['abs_path']   # video path
        start_t = self.data[index]['begin_t']  # starting frame
        end_t = self.data[index]['end_t']     # ending   frame
        # get 8 random frames from the video
        time_index = np.random.randint(
            start_t, end_t - self.sample_duration) + 1

        frame_indices = list(
            range(time_index, time_index + self.sample_duration))  # get the corresponding frames

        # print('frame_indices :', frame_indices)
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices)

        # get original height and width
        w, h = clip[0].size
        if self.spatial_transform is not None:
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        # get bbox
        cls = self.data[index]['class']
        name = self.data[index]['video']
        json_key = os.path.join(cls, name)

        with open(self.json_file, 'r') as fp:
            data = json.load(fp)[json_key]

        # frames = list(range(time_index, ))

        boxes = [data[i] for i in frame_indices]
        # print('len(boxes) {}, len(boxes[0] {}'.format(
        #     len(boxes), len(boxes[0])))

        class_int = self.classes_idx[cls]
        target = torch.IntTensor([class_int])
        print('target : ', target)
        gt_bboxes = torch.Tensor([boxes[i] + [class_int]
                                  for i in range(len(boxes))])

        print('gt_bboxes ', gt_bboxes)
        if self.mode == 'train':
            return clip, target, (h, w),  gt_bboxes
        else:
            return clip, target, (h, w),  gt_bboxes, self.data[index]['abs_path']

    def __len__(self):
        return len(self.data)
