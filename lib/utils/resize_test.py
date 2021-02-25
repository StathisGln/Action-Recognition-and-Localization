import torch

# from video_dataset import Video
from lib.dataloaders.video_dataset import Video
from lib.utils.spatial_transforms import (
    Compose, Scale, ToTensor)
from lib.utils.temporal_transforms import LoopPadding

import cv2

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)


    sample_size = 112
    sample_duration = 16  # len(images)
    batch_size = 1
    n_threads = 0

    # # get mean
    # mean =  [103.75581543 104.79421473  91.16894564] # jhmdb
    mean = [103.29825354, 104.63845484,  90.79830328]  # jhmdb from .png

    # generate model
    last_fc = False

    scale_size = [sample_size,sample_size]
    spatial_transform = Compose([Scale(sample_size),  # [Resize(sample_size),
                                 ToTensor()])
    temporal_transform = LoopPadding(sample_duration)

    spatial_transform2 = Compose([  # [Resize(sample_size),
                                 ToTensor()])

    ## UCF code
    dataset_folder = '/gpu-data/sgal/UCF-101-frames'
    boxes_file = './pyannot.pkl'
    actions = ['Basketball','BasketballDunk','Biking','CliffDiving','CricketBowling',
               'Diving','Fencing','FloorGymnastics','GolfSwing','HorseRiding','IceDancing',
               'LongJump','PoleVault','RopeClimbing','SalsaSpin','SkateBoarding','Skiing',
               'Skijet','SoccerJuggling','Surfing','TennisSwing','TrampolineJumping',
               'VolleyballSpiking','WalkingWithDog']

    cls2idx = {actions[i]: i for i in range(0, len(actions))}


    data = Video(dataset_folder, frames_dur=sample_duration, spatial_transform=spatial_transform,
                 temporal_transform=temporal_transform, json_file=boxes_file,
                 mode='train', classes_idx=cls2idx, sample_size=112)
    # data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
    #                                           shuffle=True, num_workers=n_threads, pin_memory=True)

    # ## JHMDB code


    # dataset_folder = '/gpu-data/sgal/JHMDB-act-detector-frames'
    # splt_txt_path =  '/gpu-data/sgal/splits'
    # boxes_file = './poses.json'

    # classes = ['brush_hair', 'clap', 'golf', 'kick_ball', 'pour',
    #            'push', 'shoot_ball', 'shoot_gun', 'stand', 'throw', 'wave',
    #            'catch','climb_stairs', 'jump', 'pick', 'pullup', 'run', 'shoot_bow', 'sit',
    #            'swing_baseball', 'walk' ]


    # cls2idx = {classes[i]: i for i in range(0, len(classes))}

    # data = Video(dataset_folder, frames_dur=sample_duration, spatial_transform=spatial_transform,
    #              temporal_transform=temporal_transform, json_file = boxes_file,
    #              split_txt_path=splt_txt_path, mode='test', classes_idx=cls2idx)
    # data2 = Video(dataset_folder, frames_dur=sample_duration, spatial_transform=spatial_transform2,
    #              temporal_transform=temporal_transform, json_file = boxes_file,
    #              split_txt_path=splt_txt_path, mode='test', classes_idx=cls2idx)


    # clips,  (h, w), gt_tubes, final_rois = data[906]
    # clips,  (h, w), gt_tubes, final_rois = data[905]
    # clips, (h,w), gt_tubes, gt_rois, path,frame_indices = data[1024]

    ## UCF-25
    clips, h,w, gt_tubes_r, gt_rois_r, n_actions = data[150]
    # ## JHMDB
    # clips, (h,w), gt_tubes_r,n_actions, path,frame_indices = data[150]
    # clips2, (h,w), gt_tubes_r,n_actions, path,frame_indices = data2[150]

    print('gt_tubes_r.shape :',gt_tubes_r.shape)
    print('gt_rois_r.shape :',gt_rois_r.shape)
    colors = [ (255,0,0), (0,255,0), (0,0,255)]
    clips = clips.squeeze().permute(1,2,3,0)

    print(n_actions)
    for i in range(16):
        # img = cv2.imread(os.path.join(path, 'image_{:0>5}.jpg'.format(frame_indices[i])))
        # img = cv2.imread(os.path.join(path, '{:0>5}.png'.format(frame_indices[i])))
        img = clips[i].cpu().numpy()
        # print(img.shape)
        img_tmp = img.copy()
        img_tmp1 = img.copy()
        img_tmp2 = img.copy()
        for j in range(n_actions):
            cv2.rectangle(img_tmp,(int(gt_tubes_r[j,0]),int(gt_tubes_r[j,1])),(int(gt_tubes_r[j,3]),int(gt_tubes_r[j,4])), (0,255,0),1)
            cv2.rectangle(img_tmp2,(int(gt_tubes_r[j,0]),int(gt_tubes_r[j,1])),(int(gt_tubes_r[j,3]),int(gt_tubes_r[j,4])), (0,255,0),1)
            cv2.rectangle(img_tmp2,(int(gt_rois_r[j,i,0]),int(gt_rois_r[j,i,1])),(int(gt_rois_r[j,i,2]),int(gt_rois_r[j,i,3])), (255,0,0),1)
            cv2.rectangle(img_tmp2,(int(gt_rois_r[j,i,0]),int(gt_rois_r[j,i,1])),(int(gt_rois_r[j,i,2]),int(gt_rois_r[j,i,3])), (255,0,0),1)

        cv2.imwrite('./out_frames/tube_{:0>3}.jpg'.format(i), img_tmp)
        cv2.imwrite('./out_frames/rois_{:0>3}.jpg'.format(i), img_tmp1)
        cv2.imwrite('./out_frames/rois_{:0>3}.jpg'.format(i), img_tmp2)


