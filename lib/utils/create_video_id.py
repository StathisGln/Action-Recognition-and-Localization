import os
import json

def get_vid_dict(dataset_path):

    vid_names = []
    
    classes = next(os.walk(dataset_path, True))[1]

    for cls in classes:
        videos = next(os.walk(os.path.join(dataset_path,cls), True))[1]

        for vid in videos:
            video_path = os.path.join(cls,vid)
            vid_names.append(video_path)

    vid2idx = {vid_names[i]: i for i in range(0, len(vid_names))}
    return vid2idx, vid_names


if __name__ == '__main__':

    dataset_path =  '/gpu-data2/sgal/UCF-101-frames'

    vid2idx = get_vid_dict(dataset_path)
    id = vid2idx['CricketBowling/v_CricketBowling_g12_c04']
    print(id)
