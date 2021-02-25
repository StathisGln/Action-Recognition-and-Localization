import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import os
import functools
import glob
import json
from itertools import groupby
from create_tubes_from_boxes import create_tube
from resize_rpn import resize_rpn

from lib.utils.spatial_transforms import (Compose, Normalize, Scale, ToTensor)
from lib.utils.temporal_transforms import LoopPadding

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
    return dataset
    
def make_dataset(dataset_path, split_txt_path, boxes_file, mode='train'):
    dataset = []
    classes = next(os.walk(dataset_path, True))[1]

    with open(boxes_file, 'r') as fp:
        boxes_data = json.load(fp)
        
    assert classes != (None), 'classes must not be None, Check dataset path'
    
    max_sim_actions = -1
    max_frames = -1
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

class video_names(data.Dataset):
    def __init__(self, dataset_folder, spt_path,  boxes_file, vid2idx, mode='train',get_loader=get_default_video_loader,
                 sample_size=112,  classes_idx=None):

        self.dataset_folder = dataset_folder
        self.sample_size = sample_size
        self.boxes_file = boxes_file
        self.vid2idx = vid2idx
        self.mode = mode
        self.data, self.max_frames, self.max_actions = make_dataset_names( dataset_folder, spt_path, boxes_file, mode)
        self.loader = get_loader()
        self.classes_idx = classes_idx
        # mean = [112.07945832, 112.87372333, 106.90993363]  # ucf-101 24 classes
        mean = [103.29825354, 104.63845484,  90.79830328]  # jhmdb from .png
        spatial_transform = Compose([Scale(sample_size),  # [Resize(sample_size),
                                     ToTensor(),
                                     Normalize(mean, [1, 1, 1])])
        self.spatial_transform=spatial_transform
        # self.loader = get_loader()
        
    def __getitem__(self, index):

        vid_name = self.data[index]['video']
        n_persons = self.data[index]['n_actions']
        boxes = self.data[index]['boxes']
        n_frames = self.data[index]['n_frames']
        cls = self.data[index]['class']
        class_int = self.classes_idx[cls]



        
        # abs_path = os.path.join(self.dataset_folder, vid_name)
        w, h = 320, 240
        gt_bboxes = torch.from_numpy(boxes )
        # gt_bboxes = torch.cat([gt_bboxes, torch.ones(1,gt_bboxes.size(1),1).type_as(gt_bboxes) * class_int], dim=2)
        gt_bboxes[:, :,[0,2]] = gt_bboxes[:, :, [0,2]].clamp_(min=0,max=w)
        gt_bboxes[:, :,[1,3]] = gt_bboxes[:, :, [1,3]].clamp_(min=0,max=h)
        gt_bboxes = torch.round(gt_bboxes)
        gt_bboxes_r = resize_rpn(gt_bboxes, h,w,self.sample_size)
        gt_bboxes_r = torch.cat([gt_bboxes_r, torch.ones(1,gt_bboxes.size(1),1).type_as(gt_bboxes_r) * class_int], dim=2)

        boxes_lst = gt_bboxes_r.tolist()
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

        # clips = torch.load(os.path.join(self.dataset_folder,vid_name,'images.pt'))
        # # print('clips.shape :',clips.shape)
        frame_indices= list(
            range( 1, n_frames+1))
        path = os.path.join(self.dataset_folder, vid_name)
        clip = self.loader(path, frame_indices)
        clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        f_clips = torch.zeros(self.max_frames,3,self.sample_size,self.sample_size)
        # print('f_clips.shape :',f_clips.shape)
        # print('n_frames :',n_frames )
        f_clips[:n_frames] = clip.permute(1,0,2,3)

        ## add frame to final_boxes
        
        fr_tensor = np.expand_dims( np.expand_dims( np.arange(0,final_boxes.shape[-2]), axis=1), axis=0)
        fr_tensor = np.repeat(fr_tensor, final_boxes.shape[0], axis=0)
        final_boxes = np.concatenate((final_boxes, fr_tensor), axis=-1)
        return vid_id, f_clips, final_boxes, n_frames_np, n_actions_np, h, w, class_int
    
    def __len__(self):

        return len(self.data)


def make_dataset_names(dataset_path, spt_path, boxes_file, mode):

    """
    function preparing dataset containing all available videos
    """
    dataset = []

    classes = next(os.walk(dataset_path, True))[1]

    with open(boxes_file, 'r') as fp:
        boxes_data = json.load(fp)

    assert classes != (None), 'classes must not be None, Check dataset path'

    max_frames = -1
    max_actions = -1
    for idx, cls in enumerate(classes):

        videos = []
        with open(os.path.join(spt_path,'{}_test_split1.txt'.format(cls)), 'r') as fp:
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


        for vid in videos:

            video_path = os.path.join(cls,vid)
            n_frames = len(glob.glob(os.path.join(dataset_path,video_path+'/*.png')))


            json_key = os.path.join(cls,vid)
            boxes = boxes_data[json_key]

            begin_t = 1
            end_t = min(n_frames,len(boxes))

            # if n_frames > 15:
            #     continue
            
            if n_frames > max_frames:
                max_frames = n_frames

            n_actions = 1
            rois = np.expand_dims(np.array(boxes), axis=0)
            sample_i = {
                'video': video_path,
                'n_actions' : 1,
                'boxes' : rois,
                'class' : cls,
                'n_frames' : n_frames,
            }
            dataset.append(sample_i)

    print(len(dataset))
    print('max_frames :',max_frames)
    return dataset, max_frames, 1


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
        im_info = torch.Tensor([self.sample_size, self.sample_size, self.sample_duration])

        if self.mode == 'train':
            return clip, h, w, gt_tubes_seg, gt_bboxes_r, class_int, n_frames, im_info
        elif self.mode == 'val':
            return clip, h, w, gt_tubes_seg, gt_bboxes_r, class_int, n_frames, im_info
        else:
            return clip, h, w, gt_tubes_seg, gt_bboxes_r, class_int, n_frames, im_info
        
        
    def __len__(self):
        return len(self.data)

class RNN_JHMDB(data.Dataset):

    def __init__(self, dataset_folder, spt_path,  boxes_file, vid2idx, mode='train',get_loader=get_default_video_loader, \
                 max_n_tubes = 25, max_len_tubes = 73):

        self.dataset_folder = dataset_folder
        self.max_n_tubes = max_n_tubes
        self.max_len_tubes = max_len_tubes
        self.boxes_file = boxes_file
        self.vid2idx = vid2idx
        self.mode = mode
        self.data, self.max_frames, self.max_actions = make_dataset_names( dataset_folder, spt_path, boxes_file, mode)
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
        features = features.mean(1)
        target_lbl  = torch.load(os.path.join(self.dataset_folder,path, 'labels.pt'),map_location='cpu')
        n_tubes = features.size(0)
        # for b in range(features.size(0)):

        for b in range(features.size(0)):

            # f_features[b,:feat_len] = features[b]
            # print('f_features.shape :',f_features.shape)
            # print('features.shape :',features.shape)
            # print('f_features[b].shape :',f_features[b].shape)
            # print('features[b].shape :',features[b].shape)
            # exit(-1)

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
