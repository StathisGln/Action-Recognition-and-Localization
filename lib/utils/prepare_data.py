import os
import glob
from PIL import Image
import functools

import torch

from lib.utils.spatial_transforms import (
    Compose, Normalize, Scale, ToTensor)

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

if __name__ == '__main__':

    dataset_path = '/gpu-data2/sgal/UCF-101-frames'
    output_path =  '/gpu-data2/sgal/UCF-101-pickle'
    
    classes = next(os.walk(dataset_path, True))[1]
    loader = get_default_video_loader()
    
    sample_size = 112
    mean = [112.07945832, 112.87372333, 106.90993363]  # ucf-101 24 classes

    spatial_transform = Compose([Scale(sample_size),  # [Resize(sample_size),
                                 ToTensor(),
                                 Normalize(mean, [1, 1, 1])])

    for cls in classes:

        videos = next(os.walk(os.path.join(dataset_path, cls), True))[1]

        if not os.path.exists(os.path.join(output_path, cls)):
            os.mkdir(os.path.join(output_path, cls))
        for vid in videos:
            video_path = os.path.join(cls,vid)
            path = os.path.join(dataset_path,video_path)
            print(path)            
            if not os.path.exists(os.path.join(output_path,cls,vid)):
                os.mkdir(os.path.join(output_path,cls,vid))

            frame_indices = sorted([ int(i.split('/')[-1][-9:-4]) for i in glob.glob(path+'/*.jpg')])
            clip = loader(path, frame_indices)

            if spatial_transform is not None:
                clip = [spatial_transform(img) for img in clip]
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

            torch.save(clip, os.path.join(output_path,cls,vid,'images.pt'))
