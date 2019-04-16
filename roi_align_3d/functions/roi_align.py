import torch
from torch.autograd import Function
from .._ext import roi_align_3d
import json

# TODO use save_for_backward instead
class RoIAlignFunction(Function):
    def __init__(self, aligned_height, aligned_width, time_dim, spatial_scale, temp_scale=1):
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.time_dim = int(time_dim)
        self.spatial_scale = float(spatial_scale)
        self.temp_scale = float(temp_scale)
        self.rois = None
        self.feature_size = None

    def forward(self, features, rois):
        self.rois = rois
        self.feature_size = features.size()
        batch_size, num_channels, data_time, data_height, data_width = features.size()
        num_rois = rois.size(0)
        # print('rois :',rois.cpu().numpy())
        # print('num_rois :',num_rois)
        # print('num_channels :',num_channels)
        # print('data_time :',data_time)
        # print('data_height  :',data_height )
        # print('data_width :',data_width)

        # print('rois :',rois.type())
        # print('features.type() :',features.type())
        # print('features.shape :',features.shape)

        # print('num_rois {}, num_channels {}, data_time {}, self.aligned_height {}, self.aligned_width {}'.format(
            # num_rois, num_channels, data_time, self.aligned_height, self.aligned_width))
        output = features.new( num_rois, num_channels, self.time_dim, self.aligned_height, self.aligned_width).zero_()

        roi_align_3d.roi_align_forward_cuda(self.aligned_height,
                                            self.aligned_width,
                                            self.time_dim,
                                            self.spatial_scale, self.temp_scale, features.cuda(),
                                            rois.cuda(), output.cuda())

        if torch.isnan(output).any():
            print('exw mono sto output nan :', output)
            exit(-1)
        return output

    def backward(self, grad_output):
        assert(self.feature_size is not None and grad_output.is_cuda)

        batch_size, num_channels, data_time, data_height, data_width = self.feature_size

        grad_input = self.rois.new(batch_size, num_channels, data_time, data_height,
                                  data_width).zero_()
        roi_align_3d.roi_align_backward_cuda(self.aligned_height,
                                          self.aligned_width,
                                          self.time_dim,
                                          self.spatial_scale, self.temp_scale, grad_output,
                                          self.rois, grad_input)
        # else:
        #     roi_align.roi_align_backward(self.aligned_height,
        #                                  self.aligned_width,
        #                                  self.time_dim,
        #                                  self.spatial_scale, self.temp_scale, grad_output,
        #                                  self.rois, grad_input)

        # print grad_input

        return grad_input, None



