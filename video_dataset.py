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
from create_tubes_from_boxes import create_tube_list,create_tube_with_frames

from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, ToTensor, Resize)
from temporal_transforms import LoopPadding
from resize_rpn import resize_boxes, resize_tube

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

def get_file_names(spt_path, mode, split_number=1):
        
    file_name =  '{}list{:0>2}.txt'.format(mode,split_number)

    with open(os.path.join(spt_path, file_name)) as fp:

        lines = fp.readlines()
        files = [i.split()[0][:-4] for i in lines]

    data =  [ i.split('/') for i in files]
    file_names = [i[1] for i in data]
    classes = list(set([i[0] for i in data]))

    return file_names

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
    max_frames = -1
    for vid, values in boxes_data.items():
        # print('vid :',vid)
        name = vid.split('/')[-1]
        n_frames = values['numf']
        annots = values['annotations']
        n_actions = len(annots)

        # find max simultaneous actions
        if n_actions > max_sim_actions:
            max_sim_actions = n_actions

        # find max number of frames
        if n_frames > max_frames:
            max_frames = n_frames

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
    print('max_frames :', max_frames)
    return dataset, max_sim_actions, max_frames

def prepare_samples (vid_names, vid_id, boxes, sample_duration, step):
    dataset = []

    video_path = vid_names[vid_id]
    name = video_path.split('/')[-1]
    n_actions = boxes.shape[0]
    n_frames = boxes.shape[1]

    begin_t = 1
    end_t = n_frames
    sample = {
        'video_path': video_path,
        'video_name' : name,
        'segment': [begin_t, end_t],
        'n_frames': n_frames,
    }
    for i in range(1, (n_frames - sample_duration + 1), step):
        sample_i = copy.deepcopy(sample)
        sample_i['frame_indices'] = list(range(i, i + sample_duration))
        sample_i['segment'] = torch.IntTensor([i, i + sample_duration - 1])
        sample_i['boxes'] = boxes[:,range(i, i + sample_duration)]
        sample_i['start_fr'] = i-1
        dataset.append(sample_i)

    return dataset, n_actions, n_frames


def make_dataset(dataset_path, spt_path, boxes_file, mode):
    dataset = []

    classes = next(os.walk(dataset_path, True))[1]

    with open(boxes_file, 'rb') as fp:
        boxes_data = pickle.load(fp)

    file_names = get_file_names(spt_path, mode, split_number=1)
    max_frames = -1
    max_actions = -1
    for cls in classes:
    # for cls in ['RopeClimbing']:
        videos = next(os.walk(os.path.join(dataset_path,cls), True))[1]
        for vid in videos:

            video_path = os.path.join(cls,vid)
            if video_path not in boxes_data or not(vid in file_names):
                # print('OXI to ',video_path)
                continue
            # print('video_path :',video_path)
            #  TrampolineJumping/v_TrampolineJumping_g17_c03
            # if(video_path == 'IceDancing/v_IceDancing_g06_c04'):

            #     print('edsdfes')
            #     print('n_frames :',n_frames)

            values= boxes_data[video_path]
            n_frames = values['numf']
            annots = values['annotations']
            n_actions = len(annots)
            if n_frames > max_frames:
                max_frames = n_frames

            if n_actions > max_actions:
                max_actions = n_actions
            # # pos 0 --> starting frame, pos 1 --> ending frame
            # s_e_fr = np.zeros((n_actions, 2)) 
            rois = np.zeros((n_actions,n_frames,5))
            rois[:,:,4] = -1 

            for k  in range(n_actions):
                sample = annots[k]
                s_frame = sample['sf']
                e_frame = sample['ef']
                # s_e_fr[k,0] = s_frame
                # s_e_fr[k,1] = e_frame
                s_label = sample['label']
                boxes   = sample['boxes']
                rois[k,s_frame:e_frame,:4] = boxes
                rois[k,s_frame:e_frame,4]  = s_label

            sample_i = {
                'video': video_path,
                'n_actions' : n_actions,
                'boxes' : rois,
                'n_frames' : n_frames,
                # 's_e_fr' : s_e_fr
            }
            dataset.append(sample_i)

    print(len(dataset))
    print('max_frames :',max_frames)
    return dataset, max_frames, max_actions

class video_names(data.Dataset):
    def __init__(self, dataset_folder, spt_path,  boxes_file, vid2idx, mode='train',get_loader=get_default_video_loader,):

        self.dataset_folder = dataset_folder
        self.boxes_file = boxes_file
        self.vid2idx = vid2idx
        self.mode = mode
        self.data, self.max_frames, self.max_actions = make_dataset( dataset_folder, spt_path, boxes_file, mode)
        self.loader = get_loader()
        
    def __getitem__(self, index):

        vid_name = self.data[index]['video']
        n_persons = self.data[index]['n_actions']
        boxes = self.data[index]['boxes']
        n_frames = self.data[index]['n_frames']
        
        ## read one image for h,w

        abs_path = os.path.join(self.dataset_folder, vid_name)
        print('vid_name :',vid_name)
        clip = self.loader(abs_path, [1])
        w, h = clip[0].size

        boxes_lst = boxes.tolist()
        rois_fr = [[z+[j] for j,z in enumerate(boxes_lst[i])] for i in range(len(boxes_lst))]
        rois_gp =[[[list(g),i] for i,g in groupby(w, key=lambda x: x[:][4])] for w in rois_fr] # [person, [action, class]

        new_rois = []
        for i in rois_gp:
            # print(len(i))
            for k in i:
                # print(k[1])
                if k[1] != -1.0 : # not background
                    tube_rois = np.zeros((n_frames,5))
                    tube_rois[:,4] = -1 
                    s_f = k[0][0][-1]
                    e_f = k[0][-1][-1] + 1
                    tube_rois[s_f : e_f] = np.array([k[0][i][:5] for i in range(len(k[0]))])
                    new_rois.append(tube_rois.tolist())
        new_rois_np = np.array(new_rois)

        n_actions = new_rois_np.shape[0]

        final_boxes = np.zeros((self.max_actions, self.max_frames, new_rois_np.shape[2]))
        final_boxes[:n_actions,:n_frames, :] = new_rois_np

        vid_id = np.array([self.vid2idx[vid_name]],dtype=np.int64)
        n_frames_np = np.array([n_frames], dtype=np.int64)
        n_actions_np = np.array([n_actions], dtype=np.int64)
        # 'IceDancing/v_IceDancing_g06_c04'
        # if vid_name == 'IceDancing/v_IceDancing_g06_c04': # 'TrampolineJumping/v_TrampolineJumping_g17_c03':
        #     print('index :',index)
        
        # if vid_id == 895:
        #     print('index :',index)
        return vid_id, final_boxes, n_frames_np, n_actions_np, h, w
    
    def __len__(self):

        return len(self.data)


class single_video(data.Dataset):
    def __init__(self, dataset_folder, vid_names, vid_id,frames_dur=16, sample_size=112,
                 spatial_transform=None, temporal_transform=None, boxes=None,
                 get_loader=get_default_video_loader, mode='train', classes_idx=None):

        self.mode = mode
        self.dataset_folder = dataset_folder
        self.data, self.n_actions, n_frames = prepare_samples(
                    vid_names, vid_id, boxes, frames_dur, int(frames_dur/2))

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = get_loader()
        self.sample_duration = frames_dur
        self.sample_size = sample_size
        self.classes_idx = classes_idx

        self.tensor_dim = len(range(0, n_frames-self.sample_duration, int(self.sample_duration/2)))
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        name = self.data[index]['video_name']   # video path
        path = self.data[index]['video_path']
        rois = self.data[index]['boxes']
        start_fr = self.data[index]['start_fr']

        n_frames = self.data[index]['n_frames']
        frame_indices = self.data[index]['frame_indices']
        abs_path = os.path.join(self.dataset_folder, path)

        # if path.startswith('CricketBowling/v_CricketBowling'):
        #     print(rois)
        
        # if self.temporal_transform is not None:
        #     frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(abs_path, frame_indices)

        ## get original height and width
        w, h = clip[0].size
        if self.spatial_transform is not None:
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        ## get bboxes and create gt tubes
        rois_Tensor = rois
        rois_indx = np.array(frame_indices) - frame_indices[0]
        rois_sample_tensor = rois_Tensor[:,rois_indx,:]
        rois_sample_tensor[:,:,2] = rois_sample_tensor[:,:,0] + rois_sample_tensor[:,:,2]
        rois_sample_tensor[:,:,3] = rois_sample_tensor[:,:,1] + rois_sample_tensor[:,:,3]
        rois_sample_tensor_r = resize_boxes(rois_sample_tensor, h,w,self.sample_size)

        fr_tensor = torch.Tensor(frame_indices).type_as(rois_sample_tensor_r)-1
        fr_tensor = fr_tensor.unsqueeze(1).unsqueeze(0).expand(rois_sample_tensor_r.size(0),fr_tensor.size(0), 1)
        rois_fr = torch.cat((rois_sample_tensor_r,fr_tensor ), dim=2)
        
        # print('rois_sample_tensor_r :',rois_sample_tensor_r)
        # rois_sample = rois_sample_tensor_r.tolist()
        tubes = create_tube_with_frames(rois_fr.unsqueeze(0), torch.Tensor([[h,w]*rois_sample_tensor_r.size(0)]), self.sample_duration)
        tubes = tubes.squeeze(0)
        # print('tubes :',tubes)
        padding_lines = tubes[:,-1].squeeze(0).lt(0).nonzero().view(-1)
        for i in padding_lines:
            tubes[i] = torch.zeros((7))
        ## im_info
        im_info = torch.Tensor([self.sample_size, self.sample_size, self.sample_duration] )
        return clip,  (h, w),  tubes, rois_sample_tensor_r, im_info, rois_sample_tensor_r.size(0), start_fr

    def __len__(self):
        return len(self.data)

    def __max_sim_actions__(self):
        return self.n_actions



if __name__ == '__main__':

    dataset_folder = '/gpu-data2/sgal/UCF-101-frames'
    boxes_file = '/gpu-data/sgal/pyannot.pkl'

    data = video_names(dataset_folder=dataset_folder, boxes_file=boxes_file)
    # ret = data[40]
    ret = data[500]
    # # dataset_folder = '/gpu-data/sgal/UCF-101-frames'
    # boxes_file = '/gpu-data/sgal/pyannot.pkl'

    # sample_size = 112
    # sample_duration = 16  # len(images)

    # batch_size = 10
    # n_threads = 0

    # # # get mean
    # mean = [112.07945832, 112.87372333, 106.90993363]  # ucf-101 24 classes


    # actions = ['Basketball','BasketballDunk','Biking','CliffDiving','CricketBowling',
    #            'Diving','Fencing','FloorGymnastics','GolfSwing','HorseRiding','IceDancing',
    #            'LongJump','PoleVault','RopeClimbing','SalsaSpin','SkateBoarding','Skiing',
    #            'Skijet','SoccerJuggling','Surfing','TennisSwing','TrampolineJumping',
    #            'VolleyballSpiking','WalkingWithDog']

    # cls2idx = {actions[i]: i for i in range(0, len(actions))}

    # spatial_transform = Compose([Scale(sample_size),  # [Resize(sample_size),
    #                              ToTensor(),
    #                              Normalize(mean, [1, 1, 1])])
    # temporal_transform = LoopPadding(sample_duration)

    # # data = Video_UCF(dataset_folder, frames_dur=sample_duration, spatial_transform=spatial_transform,
    # #              temporal_transform=temporal_transform, json_file=boxes_file,
    # #              mode='train', classes_idx=cls2idx)
    # dataset_folder = '/gpu-data2/sgal/UCF-101-frames'
    # vid_path = 'PoleVault/v_PoleVault_g06_c02'
    # prepare_samples(vid_path, boxes_file,16,8)
    # data = single_video(dataset_folder, vid_path, 16, sample_size, spatial_transform=spatial_transform,
    #                     temporal_transform=temporal_transform, json_file=boxes_file,
    #                     mode='train', classes_idx=cls2idx)

    # data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
    #                                           shuffle=False, num_workers=n_threads, pin_memory=True)

    # for step, dt in enumerate(data_loader):

    #      clip,  (h, w),  ret_tubes, f_rois, im_info, n_acts = dt
    #      print('clip.shape :',clip.shape)
    #      print('h :',h)
    #      print('w :',w)
    #      print('ret_tubes :',ret_tubes)
    #      print('ret_tubes.shape :',ret_tubes.shape)
    #      print('f_rois.shape :',f_rois.shape)
    #      print('n_acts :',n_acts)
    #      print('im_info.shape :',im_info.shape)
    #      print('im_info :',im_info)
