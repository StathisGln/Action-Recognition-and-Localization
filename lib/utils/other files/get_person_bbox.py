import os
import torch
import numpy as np
import torchvision
from PIL import Image
from IPython import embed
import json
from tqdm import tqdm
'''
ONLY FOR KTH DATASET

This file is because I was experimenting firstly with KTH action dataset,
which didn't contained any bbox information regarding the person
'''

def detect_multi_image(img_path, prefix_path,  thresh=0.5, class_num=1, batch_size=64):


    trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    for step, first_pos in enumerate(range(0,len(images),batch_size)):
        print(f'step {step}, first_pos {first_pos}, last_item {first_pos+batch_size}')
        img_list = []

        for pos in range(first_pos, min(len(images), first_pos+batch_size)):
            print(f'pos : {pos})')
            img = Image.open(os.path.join(prefix_path, img_path[pos]))
            img = trans(img)
            img = img.to('cuda')
            img_list.append(img)
        images_tensor = torch.stack(img_list)
        exit(-1)

    with torch.no_grad():
        pred = model([img])

    labels = pred[0]['labels']
    positions_lbl = ((labels==class_num)*1).nonzero().flatten()
    
    cof_score = pred[0]['scores']
    positions_scr =((cof_score>thresh)*1).nonzero().flatten()

    positions =  torch.tensor(np.intersect1d(positions_lbl.cpu().numpy(), positions_scr.cpu().numpy()))

    bboxes = pred[0]['boxes']
    bboxes = bboxes[positions].tolist()

    return bboxes

def detect_one_image(img_path, thresh=0.5, class_num=1):

    trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    img = Image.open(img_path)
    img = trans(img)
    img = img.to('cuda')


    with torch.no_grad():
        pred = model([img])

    labels = pred[0]['labels']
    positions_lbl = ((labels==class_num)*1).nonzero().flatten()
    
    cof_score = pred[0]['scores']
    positions_scr =((cof_score>thresh)*1).nonzero().flatten()

    positions =  torch.tensor(np.intersect1d(positions_lbl.cpu().numpy(), positions_scr.cpu().numpy()))

    bboxes = pred[0]['boxes']
    bboxes = bboxes[positions].tolist()

    return bboxes

def find_best_seq(bboxes_list, pos_list):
    '''

    :param bboxes_list: a list containing all bboxes
    :param pos_list: a list containing the positions of these bboxes
    :return: cleaned bboxes_list, pos_list
    '''

    lists = []
    starting_pos = 0
    ending_pos = 0

    for i in range(0, len(pos_list) - 1):

        if pos_list[i] + 1 == pos_list[i + 1]:
            ending_pos += 1
        else:
            lists.append([starting_pos, ending_pos + 1])
            starting_pos = i + 1
            ending_pos = i + 1

    lists.append([starting_pos, ending_pos + 1])
    lengths = np.array([ls[1] - ls[0] for ls in lists])
    max_list = np.argmax(lengths)
    positions = np.arange(lists[max_list][0], lists[max_list][1], 1)
    pos_list = [pos_list[i] for i in positions]
    bboxes_list = [bboxes_list[i] for i in positions]

    return bboxes_list, pos_list

def create_json_with_actions_for_each_video(json_file, annotations_file, output_json_file_path):
    '''
    :param json_file: file containing bboxes for each video frame
    :param annotations_file: txt file containing time stamps for each action
    :param output_json_file_path: file path for each action
    '''


    with open(json_file, 'r') as fp:
        bbox_data = json.load(fp)

    with open(annotations_file, 'r') as fp:
        txt_file_lines = fp.readlines()
        txt_data = [line.split() for line in txt_file_lines if line.startswith('person')]
        txt_dict = {el[0]: el[2:] for el in txt_data}

    final_dict = {}
    for vid, values in bbox_data.items():
        cur_key = '_'.join(vid.split('_')[:-1])
        frame_limits = txt_dict[cur_key]
        vid_dirname = os.path.dirname(list(values.values())[0]['img_path'])
        vid_name = os.path.basename(vid_dirname)
        vid_folder = os.path.basename(os.path.dirname(vid_dirname))
        vid_path = os.path.join(vid_folder, vid_name)
        this_video_actions_list = []

        for lim in frame_limits:

            if lim[-1] ==',':
                lim = lim[:-1]

            sf, ef = lim.split('-')
            starting_pos = int(sf)
            ending_pos = int(ef)
            this_action_bboxes_list = []
            debug_list = []
            ## changing limits in order to have the actual first and last frame
            bbox_found_list = []

            for pos, id in enumerate(range(starting_pos, ending_pos+1)):

                cur_img_name = f'image_{id:05d}.jpg'
                cur_img_data = values[cur_img_name]

                if cur_img_data['bbox'] != []:
                    cur_bbox = cur_img_data['bbox'][0]
                    this_action_bboxes_list.append(cur_bbox)
                    bbox_found_list.append(pos)

                    debug_list.append(id)


            this_action_bboxes_list, bbox_found_list = find_best_seq(this_action_bboxes_list, bbox_found_list)
            sf_pos, ef_pos = min(bbox_found_list), max(bbox_found_list)

            ending_pos = starting_pos + ef_pos
            starting_pos += sf_pos


            cur_action_dict = {
                'boxes' : this_action_bboxes_list,
                'sf'    : starting_pos-1,
                'ef'    : ending_pos,
                'label' : 1,
            }
            this_video_actions_list.append(cur_action_dict)
            if len(this_action_bboxes_list ) != cur_action_dict['ef'] - cur_action_dict['sf'] :
                print('-------------> ', vid, len(this_action_bboxes_list ),
                      cur_action_dict['ef'] - cur_action_dict['sf'] + 1)
        final_dict[vid] = {
            'numf' : len(values),
            'annotations' : this_video_actions_list,
            'label' : 1,
            'abs_path' : vid_path
        }

    with open(output_json_file_path, 'w') as fp:
        json.dump(final_dict, fp)

if __name__ == '__main__':

    generate_bboxes = False

    bboxes_output_file_path = '../../../dataset_annots.json'
    output_file_path  = '../../../dataset_actions_annots.json'
    actions_txt = '../../00sequences.txt'
    thresh = 0.5
    class_num = 1

    annots_dict = {}
    output_temp = '../../{}.json'

    dataset_root_path = '../../../dataset_frames/walking/'
    _, folders, _ = next(os.walk(dataset_root_path).__iter__())
    num_folders = len(folders)

    if generate_bboxes:

        # model = torchvision.models.VGG(123)
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        model = model.to('cuda')
        model.eval()

        print('GETTING BBOXES')
        for i, cur_fld in enumerate(folders):
            print(f'[{i+1}]/[{num_folders}] CURRENT FOLDER : {cur_fld}\n')

            cur_fld_path = os.path.join(dataset_root_path, cur_fld)
            _,_,images = next(os.walk(cur_fld_path).__iter__())

            # cur_dict = detect_multi_image(images, prefix_path=cur_fld_path)
            # annots_dict.update(cur_dict)

            curr_dict = {}
            for step, cur_img in enumerate(tqdm(images)):

                img_path = os.path.join(cur_fld_path, cur_img)
                bboxes_list = detect_one_image(img_path, thresh=thresh, class_num=class_num)
                curr_dict[cur_img] = {'img_path':img_path, 'bbox' : bboxes_list}
            with open(output_temp.format(cur_fld),'w') as fp:
                json.dump(curr_dict, fp)

            annots_dict[cur_fld] = curr_dict

        with open(bboxes_output_file_path, 'w') as fp:
            json.dump(annots_dict,fp)

    if True:

        create_json_with_actions_for_each_video(
            json_file=bboxes_output_file_path,
            annotations_file=actions_txt,
            output_json_file_path=output_file_path)