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
from resize_rpn import resize_rpn, resize_tube

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
            elif spl[1] == '2' and( mode == 'test' or mode == 'val'): # train video
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
    
    max_sim_actions = -1

    for idx, cls in enumerate(classes):
        
        class_path = os.path.join(dataset_path, cls)
        videos = []
        with open(os.path.join(split_txt_path,'{}_test_split1.txt'.format(cls)), 'r') as fp:
            lines=fp.readlines()
        for l in lines:
            spl = l.split()
            if spl[1] == '1' and mode == 'train' : # train video
                vid_name = spl[0][:-4]
                b_key =  os.path.join(cls,vid_name)
                if b_key in boxes_data:
                    videos.append(vid_name)
                else:
                    print ( '2', b_key)
            elif spl[1] == '2' and (mode == 'test' or mode == 'val'): # train video
                vid_name = spl[0][:-4]

                videos.append(vid_name)

        # # for each video
        # videos =  next(os.walk(class_path,True))[1]
        max_frames = -1
        # max_actions = -1

        for vid in videos:

            video_path = os.path.join(dataset_path, cls, vid)
            n_frames = len(glob.glob(video_path+'/*.png'))

            begin_t = 1

            json_key = os.path.join(cls,vid)
            boxes = boxes_data[json_key]

            if n_frames > max_frames:
                max_frames = n_frames

            # if n_actions > max_actions:
            #     max_actions = n_actions

            # boxes = [boxes[i]+[cls] for i in range(len(boxes))]
            end_t = min(n_frames,len(boxes))

            video_sample = {
                'video': vid,
                'class': cls,
                'abs_path' : video_path,
                'begin_t' : begin_t,
                'end_t' : end_t,
                'boxes' : np.array(boxes)
            }

            dataset.append(video_sample)
    print(len(dataset))
    print('max_frames :', max_frames)
    return dataset, max_frames


class Video(data.Dataset):
    def __init__(self, video_path, frames_dur=8, split_txt_path=None, sample_size= 112,
                 spatial_transform=None, temporal_transform=None, json_file = None,
                 sample_duration=16, get_loader=get_default_video_loader, mode='train', classes_idx=None):

        self.mode = mode
        self.data, self.max_frames = make_dataset(video_path, split_txt_path, json_file, self.mode)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = get_loader()
        self.sample_duration = frames_dur
        self.sample_size = sample_size
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
        begin_t = self.data[index]['begin_t']
        end_t = self.data[index]['end_t']
        boxes = self.data[index]['boxes']

        n_frames = self.data[index]['end_t']
        if n_frames < 17:
            time_index = 1
            frame_indices = list(range(time_index, min(n_frames, time_index + self.sample_duration)))  # get the corresponding frames
        else:
            try:
                time_index = np.random.randint(0, n_frames - self.sample_duration ) + 1
            except:
                print('n_frames :',n_frames)
                print(' n_frames - self.sample_duration - 1 :', n_frames - self.sample_duration )
                raise
            frame_indices = list(
            range(time_index, time_index + self.sample_duration))  # get the corresponding frames

        # print(frame_indices)
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)

        # if n_frames < 17:
        #     print('frame_indices :',frame_indices)
        clip = self.loader(path, frame_indices)

        # # get original height and width
        w, h = clip[0].size
        
        if self.spatial_transform is not None:
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        # if n_frames < 17:
        #     print('clip.shape :',clip.shape)

        # get bbox
        cls = self.data[index]['class']
        name = self.data[index]['video']
        json_key = os.path.join(cls, name)

        class_int = self.classes_idx[cls]
        gt_bboxes = torch.from_numpy(boxes )
        gt_bboxes[:,[0,2]] = gt_bboxes[:,[0,2]].clamp_(min=0,max=w)
        gt_bboxes[:,[1,3]] = gt_bboxes[:,[1,3]].clamp_(min=0,max=h)
        gt_bboxes = torch.round(gt_bboxes)
        gt_bboxes_r = resize_rpn(gt_bboxes.unsqueeze(0), h,w,self.sample_size).squeeze(0)
        gt_bboxes_r = gt_bboxes_r[frame_indices]

        ## add gt_bboxes_r class_int
        gt_bboxes_r = torch.cat((gt_bboxes_r[:,:4],torch.Tensor( [[ class_int] for i in range(len(frame_indices))])\
                                 .type_as(gt_bboxes_r)),dim=1).unsqueeze(0)
        im_info_tube = torch.Tensor([[w,h,]*gt_bboxes_r.size(0)])
        gt_tubes_seg = create_tube(gt_bboxes_r.unsqueeze(0), torch.Tensor([[self.sample_size,self.sample_size]]), self.sample_duration).squeeze(0)

        # if self.mode == 'train' or self.mode == 'val':

        #     if n_frames < 17:
        #         indexes = [0]
        #     else:
        #         indexes = range(0, (n_frames -self.sample_duration  ), int(self.sample_duration/2))
        #     gt_tubes = torch.zeros(len(indexes),7)

        #     for i in indexes :
        #         lim = min(i+self.sample_duration-1, (n_frames-1))
        #         if (n_frames < i+self.sample_duration):
        #             print('lim :',lim)
        #         vid_indices = torch.arange(i,lim-1).long()
        #         gt_rois_seg = gt_bboxes_r[:, vid_indices]
                
        #         gt_tubes_seg = create_tube(gt_rois_seg.unsqueeze(0), torch.Tensor([[self.sample_size,self.sample_size]]), self.sample_duration)                
        #     gt_tubes_seg[:,:,2] = i
        #     gt_tubes_seg[:,:,5] = i+self.sample_duration-1
        #     gt_tubes[int(i/self.sample_duration*2)] = gt_tubes_seg

                
        #     gt_tubes = torch.round(gt_tubes)

        # else:
        #     gt_tubes =  create_tube(gt_bboxes_r.unsqueeze(0), torch.Tensor([[self.sample_size,self.sample_size]]), n_frames)                

        im_info = torch.Tensor([self.sample_size, self.sample_size, self.sample_duration])

        if self.mode == 'train':
            return clip, h, w, gt_tubes_seg, gt_bboxes_r, class_int, n_frames, im_info
        elif self.mode == 'val':
            return clip, h, w, gt_tubes_seg, gt_bboxes_r, class_int, n_frames, im_info
        else:
            return clip, h, w, gt_tubes_seg, gt_bboxes_r, class_int, n_frames, im_info
        
        
    def __len__(self):
        return len(self.data)

if __name__ == "__main__":

    classes = ['brush_hair', 'clap', 'golf', 'kick_ball', 'pour',
               'push', 'shoot_ball', 'shoot_gun', 'stand', 'throw', 'wave',
               'catch','climb_stairs', 'jump', 'pick', 'pullup', 'run', 'shoot_bow', 'sit',
               'swing_baseball', 'walk'
               ]
    cls2idx = { classes[i] : i for i in range(0, len(classes)) }


    dataset_folder = '/gpu-data/sgal/JHMDB-act-detector-frames'
    splt_txt_path =  '/gpu-data/sgal/splits'
    boxes_file = '../temporal_localization/poses.json'
    sample_size = 112
    sample_duration = 16 #len(images)

    batch_size = 1
    n_threads = 0
    
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
                                              shuffle=False, num_workers=n_threads, pin_memory=True)
    # clips, (h,w), gt_tubes, gt_bboxes, n_actions, n_frames = next(data_loader.__iter__())
    # for i in data:
    #     clips, (h,w), gt_tubes, gt_bboxes, n_actions, n_frames = i
    #     # print('gt_bboxes.shape :',gt_bboxes)
    #     # print('gt_bboxes.shape :',gt_bboxes.shape)
    #     # print('gt_tubes :',gt_tubes)
    #     # print('clips.shape :',clips.shape)

    #     # print('n_frames :',n_frames)
    #     if (n_frames != gt_bboxes.size(1)):
    #         print('probleeeemm', ' n_frames :',n_frames, ' gt_bboxes :',gt_bboxes.size(1))
    clips, (h,w), gt_tubes, gt_bboxes, n_actions, n_frames = data[108]
    if (n_frames != gt_bboxes.size(1)):
        print('probleeeemm', ' n_frames :',n_frames, ' gt_bboxes :',gt_bboxes.size(1))
