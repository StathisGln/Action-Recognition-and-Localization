import torch
from torch.autograd import Function
from .._ext import roi_align_3d
import json

# TODO use save_for_backward instead
class RoIAlignFunction(Function):
    def __init__(self, aligned_height, aligned_width, time_dim, spatial_scale):
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.time_dim = int(time_dim)
        self.spatial_scale = float(spatial_scale)
        self.rois = None
        self.feature_size = None

    def forward(self, features, rois):
        self.rois = rois
        self.feature_size = features.size()
        batch_size, num_channels, data_time, data_height, data_width = features.size()
        print('features.size() :',features.size())
        offset_batch = torch.arange(batch_size)
        offset = torch.arange(data_time)

        print('offset_batch :',offset_batch)
        print('offset :', offset )
        
        # exit(-1)
        num_rois = rois.size(0)
        # features = features.permute(0,2,1,3,4).contiguous()
        
        output = features.new( num_rois, self.time_dim, num_channels,  self.aligned_height, self.aligned_width).zero_()

        roi_align_3d.roi_align_forward_cuda(self.aligned_height,
                                            self.aligned_width,
                                            self.time_dim,
                                            self.spatial_scale, features.cuda(),
                                            rois.cuda(), output.cuda())
        exit(-1)
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



# TODO use save_for_backward instead
class RoIAlignFunction_3d(Function):
    def __init__(self, aligned_height, aligned_width, spatial_scale, time_dim):
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)
        self.time_dim = int(time_dim)
        self.rois = None
        self.feature_size = None

    def forward(self, features, rois):
        self.rois = rois
        self.feature_size = features.size()
        batch_size, num_channels, data_time, data_height, data_width = features.size()
        num_rois = rois.size(0)
        output = features.new(num_rois, num_channels, data_time, self.aligned_height, self.aligned_width).zero_()
        if features.is_cuda:
            roi_align.roi_align_forward_cuda(self.aligned_height,
                                             self.aligned_width,
                                             self.aligned_time,
                                             self.spatial_scale,
                                             self.time_scale, features,
                                             rois, output)
        else:
            roi_align.roi_align_forward(self.aligned_height,
                                        self.aligned_width,
                                        self.aligned_time,
                                        self.spatial_scale,
                                        self.time_scale, features,
                                        rois, output)
#            raise NotImplementedError

        return output

    def backward(self, grad_output):
        assert(self.feature_size is not None and grad_output.is_cuda)

        batch_size, num_channels, data_height, data_width = self.feature_size

        grad_input = self.rois.new(batch_size, num_channels, data_height,
                                  data_width).zero_()
        roi_align.roi_align_backward_cuda(self.aligned_height,
                                          self.aligned_width,
                                          self.spatial_scale, grad_output,
                                          self.rois, grad_input)

        # print grad_input

        return grad_input, None
