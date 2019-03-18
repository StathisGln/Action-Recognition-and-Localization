import numpy as np
import torch

def resize_rpn(gt_rois_old, h,w, sample_size):
    '''
    Input: a torch.Tensor
    size shape is [h,w]
    '''
    target_h = sample_size
    target_w = sample_size

    gt_rois =torch.zeros(gt_rois_old.size()).type_as(gt_rois_old)

    w_bigger_h = w > h
    scale = sample_size / max(h,w)

    if w_bigger_h:
        target_h = int(np.round( h * float(sample_size) / float(w)))
    else:
        target_w = int(np.round( w * float(sample_size) / float(h)))

    top = int(max(0, np.round(((sample_size) - target_h) / 2)))
    left = int(max(0, np.round(((sample_size) - target_w) / 2)))

    gt_rois[:,:,0] = (torch.round(gt_rois_old[:,:,0]  * float(scale)) + left)
    gt_rois[:,:,1] = (torch.round(gt_rois_old[:,:,1]  * float(scale)) + top )
    gt_rois[:,:,2] = (torch.round(gt_rois_old[:,:,2]  * float(scale)) + left)
    gt_rois[:,:,3] = (torch.round(gt_rois_old[:,:,3]  * float(scale)) + top)

    return gt_rois

def resize_rpn_np(gt_rois, h,w, sample_size):
    '''
    Input: a torch.Tensor
    size shape is [h,w]
    '''
    target_h = sample_size
    target_w = sample_size

    w_bigger_h = w > h
    scale = sample_size / max(h,w)
    
    if w_bigger_h:
        target_h = int(np.round( h * float(sample_size) / float(w)))
    else:
        target_w = int(np.round( w * float(sample_size) / float(h)))

    top = int(max(0, np.round((sample_size - target_h) / 2)))
    left = int(max(0, np.round((sample_size - target_w) / 2)))
    gt_rois_new = np.zeros(gt_rois.shape)
    # top = int(max(0, np.round((h - sample_size) / 2)))
    # left = int(max(0, np.round((w - sample_size) / 2)))
    # bottom = height - top - target_h
    # right = width - left - target_w
    # print('gt_rois.shape :',gt_rois.shape)
    # print('top {}, left {}'.format(top,left))
    # print('w {}, h {}, sample {} w.type {} h.type {}'.format(w, h, sample_size, w.type(), h.type()))
    gt_rois_new[:,0] = (np.round(gt_rois[:,0]  * scale) + left)
    gt_rois_new[:,1] = (np.round(gt_rois[:,1]  * scale) + top )
    gt_rois_new[:,2] = (np.round(gt_rois[:,2]  * scale) + left)
    gt_rois_new[:,3] = (np.round(gt_rois[:,3]  * scale) + top)
    # print('gt_Rois :',gt_rois)
    return gt_rois_new

def resize_boxes(gt_rois_old, h,w, sample_size):
    '''
    Input: a torch.Tensor
    size shape is [h,w]
    '''

    target_h = sample_size
    target_w = sample_size

    gt_rois =torch.zeros(gt_rois_old.size()).type_as(gt_rois_old)

    w_bigger_h = w > h

    scale = (sample_size / max(h,w).float()).type_as(gt_rois_old)
    # print('scale.type() :',scale.type())
    # print('gt_rois_old.type() :',gt_rois_old.type())
    if w_bigger_h:
        target_h = int(np.round( h * float(sample_size) / float(w)))
    else:
        target_w = int(np.round( w * float(sample_size) / float(h)))

    top = int(max(0, np.round(((sample_size) - target_h) / 2)))
    left = int(max(0, np.round(((sample_size) - target_w) / 2)))

    gt_rois[:,:,:, 0] = (torch.round(gt_rois_old[:,:,:, 0]  * float(scale)) + left)
    gt_rois[:,:,:, 1] = (torch.round(gt_rois_old[:,:,:, 1]  * float(scale)) + top )
    gt_rois[:,:,:, 2] = (torch.round(gt_rois_old[:,:,:, 2]  * float(scale)) + left)
    gt_rois[:,:,:, 3] = (torch.round(gt_rois_old[:,:,:, 3]  * float(scale)) + top)
    gt_rois[:,:,:, 4] = gt_rois_old[:,:,:, 4]

    for i in range(gt_rois.size(0)):
        for j in range(gt_rois.size(1)): # num_action
            padding_pos = gt_rois[i,j,:,-1].lt(1).nonzero()
            if padding_pos.nelement() != 0:
                gt_rois[i,j,padding_pos] = torch.zeros((5)).type_as(gt_rois)
    return gt_rois

def resize_boxes_np(gt_rois_old, h,w, sample_size):
    '''
    Input: a torch.Tensor
    size shape is [h,w]
    '''
    target_h = sample_size
    target_w = sample_size

    gt_rois =np.zeros(gt_rois_old.shape)

    w_bigger_h = w > h
    scale = (sample_size * 1.0 )/ max(h,w) 
    # print('scale :',scale)
    if w_bigger_h:
        target_h = int(np.round( h * float(sample_size) / float(w)))
    else:
        target_w = int(np.round( w * float(sample_size) / float(h)))

    top = int(max(0, np.round(((sample_size) - target_h) / 2)))
    left = int(max(0, np.round(((sample_size) - target_w) / 2)))

    padding_pos = np.where(gt_rois_old[:,:,4] < 0)
    # print('gt_rois_old.shape :',gt_rois_old.shape)
    # print('gt_rois.shape :',gt_rois.shape)
    gt_rois[:,:,0] = (np.round(gt_rois_old[:,:,0]  * float(scale)) + left)
    gt_rois[:,:,1] = (np.round(gt_rois_old[:,:,1]  * float(scale)) + top )
    gt_rois[:,:,2] = (np.round(gt_rois_old[:,:,2]  * float(scale)) + left)
    gt_rois[:,:,3] = (np.round(gt_rois_old[:,:,3]  * float(scale)) + top)
    gt_rois[:,:,4] = gt_rois_old[:,:,4]

    # print('padding_pos :',padding_pos)
    for i in range(padding_pos[0].shape[0]):
        # print(i)
        # print('padding_pos[0][i] :',padding_pos[0][i])
        # print('gt_rois[padding_pos[i,0],padding_pos[i,1]] :',gt_rois[padding_pos[0][i],padding_pos[1][i]])
        # print('gt_rois :',gt_rois)
        gt_rois[padding_pos[0][i],padding_pos[1][i]] = np.array([[0,0,0,0,-1]])
    return gt_rois


def resize_tube(gt_rois, h_tensor,w_tensor, sample_size):
    '''
    Input: a torch.Tensor
    size shape is [h,w]
    '''
    batch_size = gt_rois.size(0)

    for i in range(batch_size):
        h = h_tensor[i].float()
        w = w_tensor[i].float()

        gt_rois[i,:,0] = gt_rois[i,:,0].clamp_(min = 0, max=w)
        gt_rois[i,:,1] = gt_rois[i,:,1].clamp_(min = 0, max=h)
        gt_rois[i,:,3] = gt_rois[i,:,3].clamp_(min = 0, max=w)
        gt_rois[i,:,4] = gt_rois[i,:,4].clamp_(min = 0, max=h)
        target_h = sample_size
        target_w = sample_size
        w_bigger_h = w > h
        scale = (sample_size / max(h,w)).type_as(gt_rois)
        # print(scale)
        # print('w_bigger_h :',w_bigger_h)
        # print(w)

        if w_bigger_h:
            target_h = int(np.round( h * float(sample_size) / float(w)))
        else:
            target_w = int(np.round( w * float(sample_size) / float(h)))
        
        # print('target_h :',target_h, ' target_w :',target_w)
        # print('h :',h, ' w :',w)
        top = int(max(0, np.round((sample_size - target_h) / 2)))
        left = int(max(0, np.round((sample_size - target_w) / 2)))

        # print('top {}, left {}'.format(top,left))
        # print('w {}, h {}, sample {} w.type {} h.type {}'.format(w, h, sample_size, w.type(), h.type()))
        # print( 'gt_rois[i,:,0] :',gt_rois[i,:,0])
        # print( 'gt_rois[i,:,0] * scale :',gt_rois[i,:,0] * scale )
        # print( 'gt_rois[i,:,0] -left :',gt_rois[i,:,0] * scale -left)
        # print( 'gt_rois[i,:,1]   :',gt_rois[i,:,1] )
        # print( 'gt_rois[i,:,1] * sample_size/h  :',gt_rois[i,:,1] * sample_size/h )
        # print( 'gt_rois[i,:,1] * sample_size/h -top  :',gt_rois[i,:,1] * sample_size/h +top)
        gt_rois[i,:,0] = (np.round(gt_rois[i,:,0]  * scale) + left)
        gt_rois[i,:,1] = (np.round(gt_rois[i,:,1]  * scale) + top )
        gt_rois[i,:,3] = (np.round(gt_rois[i,:,3]  * scale) + left)
        gt_rois[i,:,4] = (np.round(gt_rois[i,:,4]  * scale) + top)
        # print('gt_rois :', gt_rois)

    return gt_rois
