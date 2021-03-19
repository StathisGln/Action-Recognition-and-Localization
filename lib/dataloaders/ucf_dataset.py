import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import os
import functools
import copy
import json
import pickle
from itertools import groupby

from lib.utils.dataloader_utils import  *

from lib.utils.create_tubes_from_boxes import create_tube_list

from lib.utils.spatial_transforms import (
    Compose, Normalize, Scale, ToTensor)
from lib.utils.temporal_transforms import LoopPadding_still
from lib.utils.resize_rpn import resize_boxes_np

np.random.seed(42)

def get_classes():
    '''
    :return: a list of all the classes in UCF dataset
    '''
    return ['__background__', 'Basketball', 'BasketballDunk', 'Biking', 'CliffDiving', 'CricketBowling',
    'Diving', 'Fencing', 'FloorGymnastics', 'GolfSwing', 'HorseRiding', 'IceDancing',
    'LongJump', 'PoleVault', 'RopeClimbing', 'SalsaSpin', 'SkateBoarding', 'Skiing',
    'Skijet', 'SoccerJuggling', 'Surfing', 'TennisSwing', 'TrampolineJumping',
    'VolleyballSpiking', 'WalkingWithDog']

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


def make_correct_ucf_dataset(dataset_path,  boxes_file, split_txt_path, mode='train'):
    dataset = []
    classes = next(os.walk(dataset_path, True))[1]

    with open(boxes_file, 'rb') as fp:
        boxes_data = pickle.load(fp)

    assert classes != (None), 'classes must not be None, Check dataset path'

    max_sim_actions = -1
    max_frames = -1
    file_names = get_file_names(split_txt_path, mode, split_number=1)
        
    for vid, values in boxes_data.items():
        # print('vid :',vid)
        vid_name = vid.split('/')[-1]
        if vid_name not in file_names:
            continue
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

def prepare_samples (vid_names, vid_id, sample_duration, step, n_frames):

    """
    function preparing sample for single_video class
    """
    dataset = []

    video_path = vid_names[vid_id]
    name = video_path.split('/')[-1]
    # n_actions = boxes.shape[0]
    # n_frames = boxes.shape[1]

    begin_t = 1
    end_t = n_frames
    sample = {
        'video_path': video_path,
        'video_name' : name,
        'segment': [begin_t, end_t],
        'n_frames': n_frames,
    }

    if n_frames < 17:
        sample_i = copy.deepcopy(sample)
        sample_i['frame_indices'] = list(range(1, min( sample_duration, n_frames)+1))
        sample_i['start_fr'] = 0
        dataset.append(sample_i)
    else:
        for i in range(1, (n_frames ), step):
            sample_i = copy.deepcopy(sample)
            sample_i['frame_indices'] = list(range(i, min(i + sample_duration, n_frames+1)))
            sample_i['start_fr'] = i-1
            dataset.append(sample_i)
    return dataset


def make_dataset(dataset_path, spt_path, boxes_file, mode):

    """
    function preparing dataset containing all available videos
    """
    dataset = []

    classes = next(os.walk(dataset_path, True))[1]

    with open(boxes_file, 'rb') as fp:
        boxes_data = pickle.load(fp)

    file_names = get_file_names(spt_path, mode, split_number=1)
    max_frames = -1
    max_actions = -1
    #TrampolineJumping/v_TrampolineJumping_g10_c01
    for cls in classes:
    # for cls in ['Biking']:
    # for cls in ['GolfSwing']:
    # for cls in ['TrampolineJumping']:
    # for cls in ['Basketball']:
        videos = next(os.walk(os.path.join(dataset_path,cls), True))[1]
        for vid in videos:
        # for vid in [ 'v_Biking_g22_c03','v_Biking_g03_c03','v_Biking_g13_c03','v_Biking_g23_c02','v_Biking_g12_c03']:
        # for vid in ['v_TrampolineJumping_g21_c02','v_TrampolineJumping_g10_c01','v_TrampolineJumping_g20_c02','v_TrampolineJumping_g09_c05' , 'v_TrampolineJumping_g10_c06','v_TrampolineJumping_g11_c05']:
        # for vid in ['v_TrampolineJumping_g11_c05','v_TrampolineJumping_g21_c02','v_TrampolineJumping_g10_c01','v_TrampolineJumping_g20_c02','v_TrampolineJumping_g09_c05' , 'v_TrampolineJumping_g10_c06']:
        # for vid in ['v_VolleyballSpiking_g14_c03','v_TrampolineJumping_g20_c02']: # empty rois

        # for vid in [' v_VolleyballSpiking_g23_c01']:
        # for vid in ['v_TrampolineJumping_g10_c06']:
        # for vid in ['v_VolleyballSpiking_g14_c03','v_VolleyballSpiking_g23_c01']:
        # for vid in ['v_GolfSwing_g17_c05','v_GolfSwing_g15_c05','v_GolfSwing_g14_c05','v_GolfSwing_g18_c05','v_GolfSwing_g19_c05','v_GolfSwing_g12_c05','v_GolfSwing_g11_c05']:

            video_path = os.path.join(cls,vid)
            if video_path not in boxes_data or not(vid in file_names):

                # print('OXI to ',video_path)
                continue

            values= boxes_data[video_path]
            n_frames = values['numf']
            annots = values['annotations']
            n_actions = len(annots)
            # if n_actions < 2:
            #     continue
            # if n_frames != 200:
            #     continue
            if n_frames > max_frames:
                max_frames = n_frames

            # if n_frames < 800:
            #     continue
            # if n_frames > 100:
            #     continue
            # if n_actions < 2:
            #     continue

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
                s_label = sample['label']+1 ## because annotations start from 0 without background
                boxes   = sample['boxes']
                rois[k,s_frame:e_frame,:4] = boxes
                rois[k,s_frame:e_frame,4]  = s_label

            sample_i = {
                'video': video_path,
                'n_actions' : n_actions,
                'boxes' : rois,
                'n_frames' : n_frames,
                'class' : s_label
                # 's_e_fr' : s_e_fr
            }
            dataset.append(sample_i)

    print(len(dataset))
    print('max_frames :',max_frames)
    return dataset, max_frames, max_actions

class Video_Dataset_whole_video(data.Dataset):
    def __init__(self, video_path, split_txt_path=None, bboxes_file=None, vid2idx=None,
                 mode='train', get_loader=get_default_video_loader,
                 sample_size=112, classes_idx=None, spatial_transform=None):

        self.dataset_folder = video_path
        self.sample_size = sample_size
        self.boxes_file = bboxes_file
        self.vid2idx = vid2idx
        self.mode = mode
        self.data, self.max_frames, self.max_actions = make_dataset(video_path, split_txt_path, bboxes_file, mode)
        self.loader = get_loader()
        self.classes_idx = classes_idx
        self.spatial_transform=spatial_transform
        # self.loader = get_loader()
        
    def __getitem__(self, index):

        vid_name = self.data[index]['video']
        n_persons = self.data[index]['n_actions']
        boxes = self.data[index]['boxes']
        n_frames = self.data[index]['n_frames']
        cls = self.data[index]['class']
        w, h = 320, 240

        boxes = preprocess_boxes(boxes, h, w, self.sample_size, add_frames=True)

        boxes_lst = boxes.tolist()
        rois_fr = [[z+[j] for j,z in enumerate(boxes_lst[i])] for i in range(len(boxes_lst))]
        rois_gp =[[[list(g),i] for i,g in groupby(w, key=lambda x: x[:][4])] for w in rois_fr] # [person, [action, class]

        new_rois = []
        for i in rois_gp:
            for k in i:
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

        frame_indices= list(
            range( 1, n_frames+1))
        path = os.path.join(self.dataset_folder, vid_name)
        clip = self.loader(path, frame_indices)
        clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        f_clips = torch.zeros(self.max_frames,3,self.sample_size,self.sample_size)
        f_clips[:n_frames] = clip.permute(1,0,2,3)

        ## add frame to final_boxes
        
        fr_tensor = np.expand_dims( np.expand_dims( np.arange(0,final_boxes.shape[-2]), axis=1), axis=0)
        fr_tensor = np.repeat(fr_tensor, final_boxes.shape[0], axis=0)
        final_boxes = np.concatenate((final_boxes, fr_tensor), axis=-1)
        return vid_id, f_clips, final_boxes, n_frames_np, n_actions_np, h, w, cls
    
    def __len__(self):

        return len(self.data)

class single_video(data.Dataset):
    def __init__(self, dataset_folder, h, w, vid_names, vid_id,frames_dur=16, sample_size=112, 
                 classes_idx=None, n_frames=-1, json_file=None):

        self.h = h
        self.w = w
        self.dataset_folder = dataset_folder
        self.data = prepare_samples(
                    vid_names, vid_id, frames_dur, int(frames_dur/2), n_frames)
        vid_path = vid_names[vid_id]
        self.temporal_transform = LoopPadding_still(frames_dur)
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
        start_fr = self.data[index]['start_fr']
        n_frames = self.data[index]['n_frames']
        frame_indices = self.data[index]['frame_indices']
        abs_path = os.path.join(self.dataset_folder, path)
        frame_indices = np.array(frame_indices) - 1
        frame_indices = np.array(self.temporal_transform(frame_indices.tolist()))
        
        im_info = np.array([self.sample_size,self.sample_size, self.sample_duration])
        # return clip, frame_indices, im_info, start_fr
        return frame_indices, im_info, start_fr


    def __len__(self):
        return len(self.data)

class Video_Dataset_small_clip(data.Dataset):

    def __init__(self, video_path, frames_dur=16, split_txt_path=None, sample_size=112,
                 spatial_transform=None, temporal_transform=None, bboxes_file=None,
                 get_loader=get_default_video_loader, mode='train', classes_idx=None):

        self.mode = mode
        self.data, self.max_sim_actions, max_frames = make_correct_ucf_dataset(
                    video_path, bboxes_file, split_txt_path, self.mode)
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = get_loader()
        self.sample_duration = frames_dur
        self.sample_size = sample_size
        self.json_file = bboxes_file
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
        n_frames = self.data[index]['n_frames']


        ## get  random frames from the video 
        time_index = np.random.randint(
            0, n_frames - self.sample_duration - 1) + 1
        # print('times_index :',time_index)
        # print('n_frames :',n_frames)
        frame_indices = list(
            range(time_index, time_index + self.sample_duration))  # get the corresponding frames

        clip = self.loader(path, frame_indices)
        ## get original height and width
        w, h = clip[0].size
        clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        ## get bboxes and create gt tubes
        boxes = preprocess_boxes(rois,h, w,self.sample_size)
        boxes = boxes[:,np.array(frame_indices),:5]
        rois_sample = boxes.tolist()

        rois_fr = [[z+[j] for j,z in enumerate(rois_sample[i])] for i in range(len(rois_sample))]
        rois_gp =[[[list(g),i] for i,g in groupby(w, key=lambda x: x[:][4])] for w in rois_fr] # [person, [action, class]

        final_rois_list = []
        for p in rois_gp: # for each person
            for r in p:
                if r[1] > -1 :
                    final_rois_list.append(r[0])

        num_actions = len(final_rois_list)

        if num_actions == 0:
            final_rois = torch.zeros(1,self.sample_duration,5)
            gt_tubes = torch.zeros(1,7)
        else:
            final_rois = torch.zeros((num_actions,self.sample_duration,5)) # num_actions x [x1,y1,x2,y2,label]
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

        f_rois = torch.zeros(self.max_sim_actions,self.sample_duration,5)
        f_rois[:n_acts,:,:] = final_rois[:n_acts]

        im_info = torch.Tensor([self.sample_size, self.sample_size, self.sample_duration])

        return clip,  np.array([h], dtype=np.int64), np.array([w],dtype=np.int64),  ret_tubes, f_rois, np.array([n_acts],dtype=np.int64), np.array([n_frames],dtype=np.int64), im_info

    def __len__(self):
        return len(self.data)

class RNN_UCF(data.Dataset):

    def __init__(self, dataset_folder, spt_path,  boxes_file, vid2idx, mode='train',get_loader=get_default_video_loader, \
                 max_n_tubes = 19, max_len_tubes = 73):

        self.dataset_folder = dataset_folder
        self.max_n_tubes = max_n_tubes
        self.max_len_tubes = max_len_tubes
        self.boxes_file = boxes_file
        self.vid2idx = vid2idx
        self.mode = mode
        self.data, self.max_frames, self.max_actions = make_dataset( dataset_folder, spt_path, boxes_file, mode)
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data[index]['video']   # video path

        # f_features = torch.zeros(self.max_n_tubes,self.max_len_tubes, 64, 16) - 1 
        f_features = torch.zeros(self.max_n_tubes, 64, 16) - 1 
        len_tubes = torch.zeros(self.max_n_tubes) 
        f_target_lbl = torch.zeros(self.max_n_tubes) - 1

        # f_features = np.zeros((self.max_n_tubes, 64, 16)) - 1 
        # len_tubes = np.zeros((self.max_n_tubes))
        # f_target_lbl = np.zeros((self.max_n_tubes)) - 1

        features    = torch.load(os.path.join(self.dataset_folder,path, 'feats.pt'),map_location='cpu')
        target_lbl  = torch.load(os.path.join(self.dataset_folder,path, 'labels.pt'),map_location='cpu')
        n_tubes = features.size(0)
        # for b in range(features.size(0)):

        for b in range(features.size(0)):

            # f_features[b,:feat_len] = features[b]
            f_features[b] = features[b]
            f_target_lbl[b] = target_lbl[b]
            # for j in range(features.size(1)):
            #     len_tubes[b] += 1
            #     if final_tubes[b,j,0] == -1:
            #         len_tubes[b] -= 1
            #         break

        # return f_features, len_tubes,  f_target_lbl,
        return f_features, n_tubes,   f_target_lbl,

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':

    dataset_folder = '/gpu-data2/sgal/UCF2-101-frames'
    boxes_file = '/gpu-data/sgal/pyannot.pkl'

    data = Video_Dataset_whole_video(video_path=dataset_folder, bboxes_file=boxes_file)
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
