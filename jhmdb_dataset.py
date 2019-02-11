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
from create_tubes_from_boxes import create_tube

from spatial_transforms import (Compose, Normalize, Scale, CenterCrop, ToTensor, Resize)
from temporal_transforms import LoopPadding

np.random.seed(42) #

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
        # image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
        # image_path = os.path.join(video_dir_path, '{:05d}.jpg'.format(i))
        image_path = os.path.join(video_dir_path, '{:05d}.png'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            print('image_path {} doesn\'t exist'.format( image_path))
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

def create_tcn_dataset(split_txt_path,json_path, classes, mode):

    videos = []
    dataset = []
    txt_files = glob.glob(split_txt_path+'/*1.txt') # 1rst split
    for txt in txt_files:
        class_name = txt.split('/')[-1][:-16]
        class_idx = classes.index(class_name)
        with open(txt, 'r') as fp:
            lines=fp.readlines()
        for l in lines:
            spl = l.split()
            if spl[1] == '1' and mode == 'train': # train video
                vid_name = spl[0][:-4]
                videos.append(vid_name)
            elif spl[1] == '2' and mode == 'test': # train video
                vid_name = spl[0][:-4]
                videos.append(vid_name)

        with open(os.path.join(json_path,class_name+'.json'), 'r') as fp:
            data = json.load(fp)
        for feat in data.keys():
            if feat in videos:
                sample = {
                    'video' : feat,
                    'class' : class_name,
                    'class_idx' : class_idx
                }
                dataset.append(sample)
    print(len(dataset))
    return dataset
    
def make_dataset(dataset_path, split_txt_path, boxes_file, mode='train'):
    dataset = []
    classes = next(os.walk(dataset_path, True))[1]

    with open(boxes_file, 'r') as fp:
        boxes_data = json.load(fp)
        
    assert classes != (None), 'classes must not be None, Check dataset path'
    
    for idx, cls in enumerate(classes):
        
        class_path = os.path.join(dataset_path, cls)
        videos = []
        with open(os.path.join(split_txt_path,'{}_test_split1.txt'.format(cls)), 'r') as fp:
            lines=fp.readlines()
        for l in lines:
            spl = l.split()
            if spl[1] == '1' and mode == 'train': # train video
                vid_name = spl[0][:-4]
                b_key =  os.path.join(cls,vid_name)
                if b_key in boxes_data:
                    videos.append(vid_name)
                else:
                    print ( '2', b_key)
            elif spl[1] == '2' and mode == 'test': # train video
                vid_name = spl[0][:-4]

                videos.append(vid_name)

        # # for each video
        # videos =  next(os.walk(class_path,True))[1]
        for vid in videos:

            video_path = os.path.join(dataset_path, cls, vid)
            n_frames = len(glob.glob(video_path+'/*.png'))
            begin_t = 1
            end_t = n_frames

            video_sample = {
                'video': vid,
                'class': cls,
                'abs_path' : video_path,
                'begin_t' : begin_t,
                'end_t' : n_frames

            }

            dataset.append(video_sample)
    print(len(dataset))
    return dataset


class Video(data.Dataset):
    def __init__(self, video_path, frames_dur=8, split_txt_path=None,
                 spatial_transform=None, temporal_transform=None, json_file = None,
                 sample_duration=16, get_loader=get_default_video_loader, mode='train', classes_idx=None):

        self.mode = mode
        self.data = make_dataset(video_path, split_txt_path, json_file, self.mode)

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
        path = self.data[index]['abs_path']
        end_t = self.data[index]['end_t']

        n_frames = self.data[index]['end_t']

        if n_frames < 17:
            time_index  = 1
            frame_indices = list(
            range(time_index, n_frames+1))  # get the corresponding frames
        else:
            time_index = np.random.randint(
                0, n_frames - self.sample_duration+1 ) + 1
            frame_indices = list(
                range(time_index, time_index + self.sample_duration))  # get the corresponding frames
        # print('path :',path ,' n_frames :',n_frames, ' index :', index, ' time_index :', time_index)


        # print(frame_indices)
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices)

        # # get original height and width
        # print('clip.size :', clip[0].size)
        w, h = clip[0].size
        
        if self.spatial_transform is not None:
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        # get bbox
        cls = self.data[index]['class']
        name = self.data[index]['video']
        json_key = os.path.join(cls, name)

        # print(json_key)
        with open(self.json_file, 'r') as fp:
            data = json.load(fp)[json_key]
            
        boxes = data[time_index-1:time_index +self.sample_duration-1] # because time_index starts from 1
        
        class_int = self.classes_idx[cls]
        # frames = list(range( time_index, time_index + self.sample_duration))
        target = torch.IntTensor([class_int] * self.sample_duration)
        # print(len(boxes)
        gt_bboxes = torch.Tensor([boxes[i] + [ class_int] for i in range(len(boxes))]).clamp_(min=0)
        gt_bboxes_tube = torch.Tensor([boxes[i] + [i, class_int] for i in range(len(boxes))]).unsqueeze(0)
        gt_bboxes = torch.round(gt_bboxes)
        # print('gt_bboxes.shape :',gt_bboxes.shape)
        # im_info_tube = torch.Tensor([[w,h,frames[0],frames[-1]]*gt_bboxes.size(0)])
        im_info_tube = torch.Tensor([[w,h,]*gt_bboxes.size(0)])
        # print('im_info_tube :',im_info_tube)
        gt_tubes = create_tube(gt_bboxes_tube.unsqueeze(2),im_info_tube,self.sample_duration)
        gt_tubes = torch.round(gt_tubes)
        
        # print(gt_bboxes)
        if self.mode == 'train':
            return clip, (h,w), gt_tubes, gt_bboxes
        else:
            return clip, (h,w), gt_tubes, gt_bboxes, self.data[index]['abs_path'], frame_indices
        
        
    def __len__(self):
        return len(self.data)

class TCN_Dataset(data.Dataset):
    def __init__(self, split_txt_path, json_path,classes, mode):

        self.split_txt_path = split_txt_path
        self.json_path = json_path
        self.mode = mode
        if mode == 'train':
            self.data = create_tcn_dataset(split_txt_path,json_path, classes, mode)
        elif mode == 'test':
            self.data = create_tcn_dataset(split_txt_path,json_path, classes, mode)
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
            data= json.load(fp)
        feats = data[path]['clips']
        final_feats = np.zeros((len(feats),512))
        for f in range(len(feats)):
            final_feats[f,:]=feats[f]['features']
        
        return final_feats, path, target

    def __len__(self):
        return len(self.data)



# def check_boxes:

#     for 

if __name__ == "__main__":

    classes = ['brush_hair', 'clap', 'golf', 'kick_ball', 'pour',
               'push', 'shoot_ball', 'shoot_gun', 'stand', 'throw', 'wave',
               'catch','climb_stairs', 'jump', 'pick', 'pullup', 'run', 'shoot_bow', 'sit',
               'swing_baseball', 'walk'
               ]
    cls2idx = { classes[i] : i for i in range(0, len(classes)) }


    dataset_folder = '/gpu-data/sgal/JHMDB-act-detector-frames'
    splt_txt_path =  '/gpu-data/sgal/splits'
    boxes_file = './poses.json'
    sample_size = 112
    sample_duration = 16 #len(images)

    batch_size = 1
    n_threads = 2
    
    mean = [103.29825354, 104.63845484,  90.79830328] # jhmdb from .png

    spatial_transform = Compose([Scale(sample_size), # [Resize(sample_size),
                                 # CenterCrop(sample_size),
                                 ToTensor(),
                                 Normalize(mean, [1, 1, 1])])
    temporal_transform = LoopPadding(sample_duration)

    
    


    data = Video(dataset_folder, frames_dur=sample_duration, spatial_transform=spatial_transform,
                 temporal_transform=temporal_transform, json_file = boxes_file,
                 split_txt_path=splt_txt_path, mode='train', classes_idx=cls2idx)
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                              shuffle=True, num_workers=n_threads, pin_memory=True)
    clips, (h,w), gt_tubes, gt_bboxes = next(data_loader.__iter__())
    # print('gt_bboxes.shape :',gt_bboxes)
    # print('gt_bboxes.shape :',gt_bboxes.shape)
    # print('gt_tubes :',gt_tubes)
    # print('clips.shape :',clips.shape)
