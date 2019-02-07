import numpy as np

def resize_rpn(gt_rois, h,w, sample_size):
    '''
    Input: a torch.Tensor
    size shape is [h,w]
    '''
    h = h.float()
    w = w.float()

    target_h = h
    target_w = w
    scale = w > h
    if scale:
        target_h = int(np.round(float(sample_size) * h / w))
    else:
        target_w = int(np.round(float(ample_size) * w / h))

    top = int(max(0, np.round((sample_size - target_h) / 2)))
    left = int(max(0, np.round((sample_size - target_w) / 2)))

    # top = int(max(0, np.round((h - sample_size) / 2)))
    # left = int(max(0, np.round((w - sample_size) / 2)))
    # bottom = height - top - target_h
    # right = width - left - target_w

    # print('top {}, left {}'.format(top,left))
    # print('w {}, h {}, sample {} w.type {} h.type {}'.format(w, h, sample_size, w.type(), h.type()))
    gt_rois[:,:,0] = (( gt_rois[:,:,0]-left ) * sample_size/w).clamp_(min=0)
    gt_rois[:,:,1] = (( gt_rois[:,:,1]-top  ) * sample_size/h).clamp_(min=0)
    gt_rois[:,:,2] = (( gt_rois[:,:,2]-left ) * sample_size/w).clamp_(min=0)
    gt_rois[:,:,3] = (( gt_rois[:,:,3]-top  ) * sample_size/h).clamp_(min=0)

    return gt_rois

def resize_tube(gt_rois, h,w, sample_size):
    '''
    Input: a torch.Tensor
    size shape is [h,w]
    '''
    h = h.float()
    w = w.float()
    target_h = h
    target_w = w
    scale = w > h
    if scale:
        target_h = int(np.round(float(sample_size) * w / w))
    else:
        target_w = int(np.round(float(sample_size) * h / h))

    top = int(max(0, np.round((sample_size - target_h) / 2)))
    left = int(max(0, np.round((sample_size - target_w) / 2)))

    # print('top {}, left {}'.format(top,left))
    # print('w {}, h {}, sample {} w.type {} h.type {}'.format(w, h, sample_size, w.type(), h.type()))

    gt_rois[:,:,0] = (( gt_rois[:,:,0]-left ) * sample_size/w).clamp_(min=0)
    gt_rois[:,:,1] = (( gt_rois[:,:,1]-top  ) * sample_size/h).clamp_(min=0)
    gt_rois[:,:,3] = (( gt_rois[:,:,3]-left ) * sample_size/w).clamp_(min=0)
    gt_rois[:,:,4] = (( gt_rois[:,:,4]-top  ) * sample_size/h).clamp_(min=0)

    return gt_rois
