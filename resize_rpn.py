import numpy as np

def resize_rpn(gt_rois, h,w, sample_size):
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

    # top = int(max(0, np.round((h - sample_size) / 2)))
    # left = int(max(0, np.round((w - sample_size) / 2)))
    # bottom = height - top - target_h
    # right = width - left - target_w
    # print('gt_rois.shape :',gt_rois.shape)
    # print('top {}, left {}'.format(top,left))
    # print('w {}, h {}, sample {} w.type {} h.type {}'.format(w, h, sample_size, w.type(), h.type()))
    gt_rois[:,0] = (np.round(gt_rois[:,0]  * scale) + left)
    gt_rois[:,1] = (np.round(gt_rois[:,1]  * scale) + top )
    gt_rois[:,2] = (np.round(gt_rois[:,2]  * scale) + left)
    gt_rois[:,3] = (np.round(gt_rois[:,3]  * scale) + top)
    # print('gt_Rois :',gt_rois)
    return gt_rois

def resize_rpn_multirois(gt_rois, h,w, sample_size):
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

    # top = int(max(0, np.round((h - sample_size) / 2)))
    # left = int(max(0, np.round((w - sample_size) / 2)))
    # bottom = height - top - target_h
    # right = width - left - target_w
    # print('top {}, left {}'.format(top,left))
    # print('w {}, h {}, sample {} w.type {} h.type {}'.format(w, h, sample_size, w.type(), h.type()))
    gt_rois[:,:,0] = (np.round(gt_rois[:,:,0]  * scale) + left)
    gt_rois[:,:,1] = (np.round(gt_rois[:,:,1]  * scale) + top )
    gt_rois[:,:,2] = (np.round(gt_rois[:,:,2]  * scale) + left)
    gt_rois[:,:,3] = (np.round(gt_rois[:,:,3]  * scale) + top)
    return gt_rois

# def resize_tube(gt_rois, h_tensor,w_tensor, sample_size):
#     '''
#     Input: a torch.Tensor
#     size shape is [h,w]
#     '''
#     batch_size = h_tensor.size(0)
#     for i in range(batch_size):
#         h = h_tensor[i].float()
#         w = w_tensor[i].float()
#         target_h = h
#         target_w = w
#         scale = w > h
#         # print(scale)
#         # print(w)
#         print('target_h :',target_h, ' target_w :',target_w)
#         if scale:
#             target_h = int(np.round(float(sample_size) * h / w))
#         else:
#             target_w = int(np.round(float(sample_size) * w / h))
#         top = int(max(0, np.round((sample_size - target_h) / 2)))
#         left = int(max(0, np.round((sample_size - target_w) / 2)))

#         print('top {}, left {}'.format(top,left))
#         print('w {}, h {}, sample {} w.type {} h.type {}'.format(w, h, sample_size, w.type(), h.type()))

#         gt_rois[i,:,0] = (( gt_rois[i,:,0]-left ) * sample_size/w).clamp_(min=0, max=target_w)
#         gt_rois[i,:,1] = (( gt_rois[i,:,1]-top  ) * sample_size/h).clamp_(min=0, max=target_h)
#         gt_rois[i,:,3] = (( gt_rois[i,:,3]-left ) * sample_size/w).clamp_(min=0, max=target_w)
#         gt_rois[i,:,4] = (( gt_rois[i,:,4]-top  ) * sample_size/h).clamp_(min=0, max=target_h)

#     return gt_rois

def resize_tube(gt_rois, h_tensor,w_tensor, sample_size):
    '''
    Input: a torch.Tensor
    size shape is [h,w]
    '''
    batch_size = h_tensor.size(0)
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
        scale = sample_size / max(h,w)
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
