import os
import pickle
import json
import torch
import torch.utils.data as data
from PIL import Image
import functools


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

def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
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
            'abs_path': os.path.join(dataset_path, vid),
            'class': cls,
            'n_frames': n_frames,
            'rois': rois
        }
        dataset.append(video_sample)

    print('len(dataset) :', len(dataset))
    print('max_sim_actions :', max_sim_actions)
    print('max_frames :', max_frames)
    return dataset, max_sim_actions, max_frames


class Video_Dataset(data.Dataset):

    def __init__(self, video_path, frames_dur=16, split_txt_path=None, sample_size=112,
                 spatial_transform=None, temporal_transform=None, bboxes_file=None,
                 get_loader=get_default_video_loader, mode='train', classes_idx=None):

        self.mode = mode
        self.data, self.max_sim_actions, max_frames = make_correct_dataset(
                    video_path, bboxes_file, split_txt_path, self.mode)
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = get_loader()
        self.sample_duration = frames_dur
        self.sample_size = sample_size
        self.bboxes_file = bboxes_file
        self.classes_idx = classes_idx

if __name__ == '__main__':

    # from IPython import embed
    dataset_folder = '../../dataset'
    boxes_file = '../../dataset_annots.json'

    data = Video_Dataset(video_path=dataset_folder, bboxes_file=boxes_file)