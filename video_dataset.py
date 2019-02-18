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
import pickle
from itertools import groupby
from create_tubes_from_boxes import create_tube_list
from resize_rpn import resize_rpn_multirois, resize_tube

from spatial_transforms import (Compose, Normalize, Scale, CenterCrop, ToTensor, Resize)
from temporal_transforms import LoopPadding


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


def make_correct_ucf_dataset(dataset_path,  boxes_file, mode='train'):
    dataset = []
    classes = next(os.walk(dataset_path, True))[1]

    with open(boxes_file, 'rb') as fp:
        boxes_data = pickle.load(fp)

    assert classes != (None), 'classes must not be None, Check dataset path'

    max_sim_actions = -1
    for vid, values in boxes_data.items():
        name = vid.split('/')[-1]
        n_frames = values['numf']
        annots = values['annotations']
        n_actions = len(annots)
        if n_actions > max_sim_actions:
            max_sim_actions = n_actions
        rois = np.zeros((n_actions,n_frames,5))
        rois[:,:,4] = -1 
        cls = values['label']

        for k  in range(n_actions):
            sample = annots[k]
            s_frame = sample['sf']
            e_frame = sample['ef']
            s_label = sample['label']
            boxes   = sample['boxes']
            rois[k,s_frame:e_frame,:4] = boxes
            rois[k,s_frame:e_frame,4]  = s_label

        # name = vid.split('/')[-1].split('.')[0]
        video_sample = {
            'video_name' : name,
            'abs_path' : os.path.join(dataset_path, vid),
            'class' : cls,
            'n_frames' : n_frames,
            'rois' : rois
            }
        dataset.append(video_sample)

    print('len(dataset) :',len(dataset))
    print('max_sim_actions :',max_sim_actions)
    return dataset, max_sim_actions


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



class Video(data.Dataset):
    def __init__(self, video_path, frames_dur=8, 
                 spatial_transform=None, temporal_transform=None, json_file=None, sample_size = 112,
                 sample_duration=16, get_loader=get_default_video_loader, mode='train', classes_idx=None):

        self.mode = mode
        self.data, self.max_sim_actions = make_correct_ucf_dataset(
                    video_path, json_file, self.mode)

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
        name = self.data[index]['video_name']   # video path
        cls  = self.data[index]['class']
        path = self.data[index]['abs_path']
        rois = self.data[index]['rois']

        
        n_frames = len(glob.glob(path+ '/*.jpg'))
        # print('n_frames :',n_frames, ' files ', sorted(glob.glob(path+ '/*.jpg')))
        # print('path :',path,  ' n_frames :', n_frames, 'index :',index)        

        ## get  random frames from the video 
        time_index = np.random.randint(
            0, n_frames - self.sample_duration - 1) + 1

        frame_indices = list(
            range(time_index, time_index + self.sample_duration))  # get the corresponding frames
        # print('frame_indices :', frame_indices)

        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices)
        # print('clip : ',len(clip))
        ## get original height and width
        w, h = clip[0].size
        if self.spatial_transform is not None:
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        ## get bboxes and create gt tubes
        rois_Tensor = torch.Tensor(rois)
        # rois_Tensor[:,[0,2]] = rois_Tensor[:,[0,2]].clamp_(min=0,max=w)
        # rois_Tensor[:,[1,3]] = rois_Tensor[:,[1,3]].clamp_(min=0,max=h)
        
        # print('rois.shape :',rois.shape)
        # print('rois :', rois)
        # print('rois_Tensor.shape :',rois_Tensor.shape)
        rois_sample_tensor = rois_Tensor[:,np.array(frame_indices),:]
        # print('rois_sample_tensor :',rois_sample_tensor)
        # print('rois_sample_tensor.shape :',rois_sample_tensor.shape)
        # print('rois_sample_tensor :',rois_sample_tensor)
        rois_sample_tensor[:,:,2] = rois_sample_tensor[:,:,0] + rois_sample_tensor[:,:,2]
        rois_sample_tensor[:,:,3] = rois_sample_tensor[:,:,1] + rois_sample_tensor[:,:,3]
        rois_sample_tensor[:,:,[0,2]] =  rois_sample_tensor[:,:,[0,2]].clamp_(min=0, max=w)
        rois_sample_tensor[:,:,[1,3]] =  rois_sample_tensor[:,:,[1,3]].clamp_(min=0, max=w)
        # resize them to (112,112)
        rois_sample_tensor = resize_rpn_multirois(rois_sample_tensor, h,w,self.sample_size)
        rois_sample = rois_sample_tensor.tolist()

        rois_fr = [[z+[j] for j,z in enumerate(rois_sample[i])] for i in range(len(rois_sample))]
        rois_gp =[[[list(g),i] for i,g in groupby(w, key=lambda x: x[:][4])] for w in rois_fr] # [person, [action, class]

        final_rois_list = []
        for p in rois_gp: # for each person
            for r in p:
                if r[1] > -1 :
                    final_rois_list.append(r[0])

        num_actions = len(final_rois_list)

        if num_actions == 0:
            final_rois = torch.zeros(1,16,5)
            gt_tubes = torch.zeros(1,7)
        else:
            final_rois = torch.zeros((num_actions,16,5)) # num_actions x [x1,y1,x2,y2,label]
            for i in range(num_actions):
                # for every action:
                for j in range(len(final_rois_list[i])):
                    # for every rois
                    # print('final_rois_list[i][j][:5] :',final_rois_list[i][j][:5])
                    pos = final_rois_list[i][j][5]
                    final_rois[i,pos,:]= torch.Tensor(final_rois_list[i][j][:5])

            # # print('final_rois :',final_rois)
            gt_tubes = create_tube_list(rois_gp,[w,h], self.sample_duration) ## problem when having 2 actions simultaneously

        ret_tubes = torch.zeros(self.max_sim_actions,7)
        n_acts = gt_tubes.size(0)
        ret_tubes[:n_acts,:] = gt_tubes
        print('ret_tubes.shape :',ret_tubes.shape)
        ## f_rois
        f_rois = torch.zeros(self.max_sim_actions,self.sample_duration,5)
        f_rois[:n_acts,:,:] = final_rois[:n_acts]
        print('f_rois.shape :',f_rois.shape)
        # print('gt_tubes :',gt_tubes)
        # print(type(gt_tubes))
        # print('gt_tubes.shape :',gt_tubes.shape)
        # print('final_rois.shape :', final_rois.shape)
        if self.mode == 'train':
            # print('clips.shape :',clip.shape)
            # print('h {} w{}'.format(h,w))
            # print('gt_tubes :',gt_tubes)
            # return clip,  h, w,  gt_tubes, final_rois
            return clip,  h, w,  ret_tubes, f_rois, n_acts
        else:
            # return clip,  (h, w),  gt_tubes, final_rois,  self.data[index]['abs_path']
            return clip,  h, w,  gt_tubes, final_rois, n_acts, self.data[index]['abs_path'], frame_indices

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
        # print('target : ', target)
        gt_bboxes = torch.Tensor([boxes[i] + [class_int]
                                  for i in range(len(boxes))])

        # print('gt_bboxes ', gt_bboxes)
        if self.mode == 'train':
            return clip, target, (h, w),  gt_bboxes
        else:
            return clip, target, (h, w),  gt_bboxes, self.data[index]['abs_path']

    def __len__(self):
        return len(self.data)

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    dataset_folder = '/gpu-data/sgal/UCF-101-frames'
    boxes_file = './pyannot.pkl'
    # boxes_file = '/gpu-data/sgal/UCF-bboxes.json'
    # dataset_folder = '../UCF-101-frames'
    # boxes_file = '../UCF-101-frames/UCF-bboxes.json'

    sample_size = 112
    sample_duration = 16  # len(images)

    batch_size = 1
    n_threads = 0

    # # get mean
    # mean =  [103.75581543 104.79421473  91.16894564] # jhmdb
    # mean = [103.29825354, 104.63845484,  90.79830328]  # jhmdb from .png
    mean = [112.07945832, 112.87372333, 106.90993363]  # ucf-101 24 classes
    # generate model
    last_fc = False

    actions = ['Basketball','BasketballDunk','Biking','CliffDiving','CricketBowling',
               'Diving','Fencing','FloorGymnastics','GolfSwing','HorseRiding','IceDancing',
               'LongJump','PoleVault','RopeClimbing','SalsaSpin','SkateBoarding','Skiing',
               'Skijet','SoccerJuggling','Surfing','TennisSwing','TrampolineJumping',
               'VolleyballSpiking','WalkingWithDog']

    cls2idx = {actions[i]: i for i in range(0, len(actions))}

    spatial_transform = Compose([Scale(sample_size),  # [Resize(sample_size),
                                 ToTensor(),
                                 Normalize(mean, [1, 1, 1])])
    temporal_transform = LoopPadding(sample_duration)

    data = Video(dataset_folder, frames_dur=sample_duration, spatial_transform=spatial_transform,
                 temporal_transform=temporal_transform, json_file=boxes_file, sample_size = 112,
                 mode='train', classes_idx=cls2idx)
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                              shuffle=True, num_workers=n_threads, pin_memory=True)
    clips, h,w, gt_tubes, gt_bboxes, n_acts = next(data_loader.__iter__())

    # print(gt_bboxes)
