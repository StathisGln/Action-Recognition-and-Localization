import torch
import torch.nn as nn

# from video_dataset import Video
from lib.dataloaders.jhmdb_dataset import Video
from lib.utils.spatial_transforms import (
    Compose, Normalize, Scale, ToTensor)
from lib.utils.temporal_transforms import LoopPadding
from lib.utils.resize_rpn import resize_tube

from lib.models.action_net import ACT_net
from lib.utils.bbox_transform import bbox_transform_inv, clip_boxes
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
                                 ToTensor(),
                                 Normalize(mean, [1, 1, 1])])
    temporal_transform = LoopPadding(sample_duration)

    # ## UCF code
    # dataset_folder = '/gpu-data/sgal/UCF-101-frames'
    # boxes_file = './pyannot.pkl'
    # actions = ['Basketball','BasketballDunk','Biking','CliffDiving','CricketBowling',
    #            'Diving','Fencing','FloorGymnastics','GolfSwing','HorseRiding','IceDancing',
    #            'LongJump','PoleVault','RopeClimbing','SalsaSpin','SkateBoarding','Skiing',
    #            'Skijet','SoccerJuggling','Surfing','TennisSwing','TrampolineJumping',
    #            'VolleyballSpiking','WalkingWithDog']

    # cls2idx = {actions[i]: i for i in range(0, len(actions))}


    # data = Video(dataset_folder, frames_dur=sample_duration, spatial_transform=spatial_transform,
    #              temporal_transform=temporal_transform, json_file=boxes_file,
    #              mode='test', classes_idx=cls2idx, scale_size = scale_size)
    # # data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
    # #                                           shuffle=True, num_workers=n_threads, pin_memory=True)

    ## JHMDB code

    dataset_folder = '/gpu-data/sgal/JHMDB-act-detector-frames'
    splt_txt_path =  '/gpu-data/sgal/splits'
    boxes_file = '../temporal_localization/poses.json'

    classes = ['brush_hair', 'clap', 'golf', 'kick_ball', 'pour',
               'push', 'shoot_ball', 'shoot_gun', 'stand', 'throw', 'wave',
               'catch','climb_stairs', 'jump', 'pick', 'pullup', 'run', 'shoot_bow', 'sit',
               'swing_baseball', 'walk' ]


    cls2idx = {classes[i]: i for i in range(0, len(classes))}

    data = Video(dataset_folder, frames_dur=sample_duration, spatial_transform=spatial_transform,
                 temporal_transform=temporal_transform, json_file = boxes_file,
                 split_txt_path=splt_txt_path, mode='test', classes_idx=cls2idx)

    spatial_transform = Compose([Scale(sample_size),  # [Resize(sample_size),
                                 ToTensor()])
    temporal_transform = LoopPadding(sample_duration)


    data2 = Video(dataset_folder, frames_dur=sample_duration, spatial_transform=spatial_transform,
                 temporal_transform=temporal_transform, json_file = boxes_file,
                 split_txt_path=splt_txt_path, mode='test', classes_idx=cls2idx)

    clips, (h,w), gt_tubes,n_actions, path,frame_indices = data[100]
    clips2, (h,w), gt_tubes,n_actions, path,frame_indices = data2[100]
    print(h,w)
    print('path :',path)
    print('clips.shape :',clips.shape)
    clips = clips.unsqueeze(0)
    gt_tubes = gt_tubes.unsqueeze(0)
    print('gt_tubes.shape :',gt_tubes.shape )
    clips = clips.to(device)
    gt_tubes_r = resize_tube(gt_tubes, torch.Tensor([h]),torch.Tensor([w]),sample_size).to(device)
    gt_tubes_r = gt_tubes_r.to(device)
    im_info = torch.Tensor([[sample_size, sample_size, sample_duration]] * gt_tubes_r.size(1)).to(device)

    n_classes = len(classes)
    resnet_shortcut = 'A'

    # Init action_net
    model = ACT_net(classes)
    model.create_architecture()
    data = model.act_rpn.RPN_cls_score.weight.data.clone()

    model_data = torch.load('../temporal_localization/jmdb_model.pwf')
    # model_data = torch.load('../temporal_localization/jmdb_model_020.pwf')
    # # model_data = torch.load('../temporal_localization/r')

    model.load_state_dict(model_data)

    model = nn.DataParallel(model)
    model.to(device)

    model.eval()
    print('im_info :',im_info)
    print('-----Starts-----')
    tubes,  bbox_pred, rois, bbox_pred_s  = model(clips,
                                                 im_info,
                                                 None, None, None)
    # print('bbox_pred_s :',bbox_pred_s)
    # print('bbox_pred :',bbox_pred)
    # rpn_loss_bbox,  act_loss_bbox, rois_label = model(clips,
    #                                                   torch.Tensor([[h,w]] * gt_tubes.size(1)).to(device),
    #                                                   gt_tubes, gt_rois,
    #                                                   torch.Tensor(len(gt_tubes)).to(device))
    print('-----Eksww-----')
    # print('rois :',rois.shape)
    # print('rois :',rois)
    # print('tubes :',tubes)
    print('bbox_pred.shape :',bbox_pred.shape)
    print('bbox_pred_s.shape :',bbox_pred_s.shape)
    bbox_pred_s = bbox_pred_s.view(1,10,16,4).permute(0,2,1,3).contiguous().view(-1,10,4)
    print('bbox_pred_s.shape :',bbox_pred_s.shape)

    thresh = 0.05
    bbox_normalize_means = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    bbox_normalize_stds = (0.1, 0.1, 0.1, 0.2, 0.2, 0.1)

    print('rois :',rois)
    rois = rois[:,:,1:]

    # # rois = rois.data
    # scores = cls_prob.data
    # print('scores :', scores)
    bbox_normalize_means_s = (0.0, 0.0, 0.0, 0.0 )
    bbox_normalize_stds_s = (0.1, 0.1, 0.2, 0.2)

    
    # box_deltas = bbox_pred.view(-1, 4) * torch.FloatTensor(bbox_normalize_stds).to(device) \
    #                            + torch.FloatTensor(bbox_normalize_means).to(device)

    # box_deltas = box_deltas.view(1,-1,4)
    # pred_boxes = bbox_transform_inv_3d(tubes, box_deltas, 1)
    # pred_boxes = clip_boxes_3d(pred_boxes, im_info.data, 1)
    # pred_boxes = pred_boxes.view(1,rois.size(1),1,6)

    box_deltas_s = bbox_pred_s.view(-1, 4) * torch.FloatTensor(bbox_normalize_stds_s).to(device) \
                               + torch.FloatTensor(bbox_normalize_means_s).to(device)
    print('box_deltas_s :', box_deltas_s.shape)
    box_deltas_s = box_deltas_s.view(16,10,4)

    print(im_info.data)
    im_info_s = torch.Tensor([[112,112]] * 16).to(device)
    print('im_info_s.shape :',im_info_s)
    pred_boxes_s = bbox_transform_inv(rois, bbox_pred_s, 16)
    pred_boxes_s = clip_boxes(pred_boxes_s,im_info_s , 16)
    pred_boxes_s = pred_boxes_s.view(16,rois.size(1),1,4)

    print('pred_boxes_s :', pred_boxes_s.shape)
    print('pred_boxes_s.shape :', pred_boxes_s)
    # print('bbox_pred.shape :',pred_boxes.shape)
    
    # print(scores)
    # pred_boxes = pred_boxes.data
    # print(pred_boxes_s)
    colors = [ (255,0,0), (0,255,0), (0,0,255)]
    clips2 = clips2.squeeze().permute(1,2,3,0)

    print('rois.shape  :',rois.shape )
    for i in range(16): # frame
        # img = cv2.imread(os.path.join(path, 'image_{:0>5}.jpg'.format(frame_indices[i])))
        # img = cv2.imread(os.path.join(path, '{:0>5}.png'.format(frame_indices[i])))
        img = clips2[i].cpu().numpy()
        print(img.shape)
        # if img.all():
        #     print('Image {} not found '.format(os.path.join(path, 'image_{:0>5}.jpg'.format(frame_indices[i]))))
        #     break

        for j in range(10): #rois
            # img_tmp = img.copy()
            # cv2.rectangle(img_tmp,(int(rois[i,j,0]),int(rois[i,j,1])),(int(rois[i,j,2]),int(rois[i,j,3])), (255,0,0),3)

            # # print('out : ./out/{:0>3}.jpg'.format(i))
            # cv2.imwrite('./out_frames/action_tube_{}_{:0>3}.jpg'.format(j,i), img_tmp)

            img_tmp = img.copy()
            # cv2.rectangle(img_tmp,(int(pred_boxes_s[i,j,0,0]),int(pred_boxes_s[i,j,0,1])),(int(pred_boxes_s[i,j,0,2]),int(pred_boxes_s[i,j,0,3])), (255,0,0),3)
            cv2.rectangle(img_tmp,(int(rois[i,j,0]),int(rois[i,j,1])),(int(rois[i,j,2]),int(rois[i,j,3])), (255,0,0),3)
            # print('out : ./out/{:0>3}.jpg'.format(i))
            cv2.imwrite('./out_frames/action_rois_{}_{:0>3}.jpg'.format(j,i), img_tmp)
            img_tmp = img.copy()
            cv2.rectangle(img_tmp,(int(pred_boxes_s[i,j,0,0]),int(pred_boxes_s[i,j,0,1])),(int(pred_boxes_s[i,j,0,2]),int(pred_boxes_s[i,j,0,3])), (255,0,0),3)
            # print('out : ./out/{:0>3}.jpg'.format(i))
            cv2.imwrite('./out_frames/action_regboxes_{}_{:0>3}.jpg'.format(j,i), img_tmp)
