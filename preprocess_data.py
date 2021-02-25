import os
import numpy as np
import json
import subprocess
import numpy as np
import glob
from shutil import rmtree

def extract_frames_from_path(video_path, output_path):

    print(f'VIDEO PATH :{video_path} | OUTPUT FOLDER {output_path}')
    if os.path.exists(output_path):
        rmtree(output_path)

    subprocess.call('mkdir {}'.format(os.path.join(output_path)), shell=True)
    subprocess.call('ffmpeg -i {} {}/image_%05d.jpg'.format(video_path, output_path),
                    shell=True)

def extract_frames_from_folder(dataset,output_folder, type='avi',  classes=None):

    _,video_folders,_ = next(os.walk(dataset_root_path).__iter__())
    if classes is None:
        classes = video_folders

    for cl in classes:
        print(f'cl : {cl}')
        curr_path = os.path.join(dataset_root_path, cl)
        _, cur_subfolders, cur_videos = next(os.walk(curr_path).__iter__())
        cur_videos = [vid for vid in cur_videos if vid.endswith(type)]
        for vid in cur_videos:
            extract_frames_from_path(os.path.join(curr_path, vid), \
                                     os.path.join(output_folder, os.path.splitext(vid)[0]))


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Set folder and video type for converting videos to sequences of '\
                                     'frames for each video')

    parser.add_argument('--path', '-p', default=None,  help='add the rootfolder of videos')
    parser.add_argument('--type', '-t', default='avi', choices=['avi','mp4',], \
                        help='Video file ending') ## TODO add more choices for video
    parser.add_argument('--classes_file','-c', default=None, help='a txt| json file containing all the dataset classes')
    parser.add_argument('--output_folder','-o', default=None, help='add the output folder of the frames')
    args = parser.parse_args()

    if args.path is None:
        print('Please set the root folder path')
        exit(-1)

    dataset_root_path = args.path
    video_prefix = args.type
    output_folder = args.output_folder

    if args.classes_file is not None:
        # TODO check if reading is ok
        with open(args.clases_files, 'r') as fp:
            if args.classes_file.endswith('txt'):
                classes = fp.readlines().split()
            elif args.classes_file.endswith('json'):
                classes = json.load(fp)
            else:
                print('Unknown classes type')
                classes = None
    else:
        classes = None

    print('----------------------\n'\
    f'DATASET_ROOT_PATH : {dataset_root_path} \n'\
    f'VIDEO ENDIND : {video_prefix}\n'\
    f'OUTPUT FOLDER PATH : {output_folder}\n'\
    '----------------------')


    ## STEP 1
    # extract_frames_from_folder(dataset=dataset_root_path, output_folder=output_folder, type=video_prefix, )

