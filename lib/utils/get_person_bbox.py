import os
import torch
import numpy as np
import torchvision
from PIL import Image
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from IPython import embed
import json
from tqdm import tqdm
'''
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


if __name__ == '__main__':


    output_file_path = '../../dataset_annots.json'
    thresh = 0.5
    class_num = 1

    # model = torchvision.models.VGG(123)
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model = model.to('cuda')
    model.eval()

    annots_dict = {}
    output_temp = '../../{}.json'

    dataset_root_path = '../../dataset_frames/walking/'
    _, folders, _ = next(os.walk(dataset_root_path).__iter__())
    num_folders = len(folders)
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

    with open(output_file_path, 'w') as fp:
        json.dump(annots_dict,fp)
            
