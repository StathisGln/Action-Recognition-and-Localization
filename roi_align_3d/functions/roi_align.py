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
        # print('self.rois.shape :',self.rois.shape)
        # print('self.time_dim :',self.time_dim)
        self.feature_size = features.size()
        # print('features_size :',features.size())
        batch_size, num_channels, data_time, data_height, data_width = features.size()
        num_rois = rois.size(0)
        # print('num_rois {}, num_channels {}, data_time {}, self.aligned_height {}, self.aligned_width {}'.format(
            # num_rois, num_channels, data_time, self.aligned_height, self.aligned_width))
        output = features.new( num_rois, num_channels, self.time_dim, self.aligned_height, self.aligned_width).zero_()
        # print('output.shape :', output.shape)
        # print('self.time_dim :',self.time_dim)
        # print('output.:', output)
        # with open('../feats.json', 'w') as fp:
        #     json.dump(dict({'aligned_height':self.aligned_height, 'aligned_width' : self.aligned_width, 'spatial_scale': self.spatial_scale, 'features' :features.cpu().tolist(), 'rois' : rois.cpu().tolist()}), fp)
        # print("Before C: self.spatial_scale :", self.spatial_scale, " self.temp_scale :", self.temp_scale)
        if features.is_cuda:
            # print('CUDAAA')
            # print('features[0][0] :',features[0][0])
            roi_align_3d.roi_align_forward_cuda(self.aligned_height,
                                             self.aligned_width,
                                             self.time_dim,
                                             self.spatial_scale, self.temp_scale, features,
                                             rois, output)
            # print('outttt')
        else:
            roi_align.roi_align_forward(self.aligned_height,
                                        self.aligned_width,
                                        self.time_dim,
                                        self.spatial_scale, self._temp_scale, features,
                                        rois, output)
        # print('output :',output[0][0][0].cpu().tolist())
        # print('Eksww output.shape :', output.shape)
        return output

    def backward(self, grad_output):
        assert(self.feature_size is not None and grad_output.is_cuda)

        batch_size, num_channels, data_height, data_width = self.feature_size

        grad_input = self.rois.new(batch_size, num_channels, data_height,
                                  data_width).zero_()
        if features.is_cuda:
            roi_align.roi_align_backward_cuda(self.aligned_height,
                                              self.aligned_width,
                                              self.time_dim,
                                              self.spatial_scale, self.temp_scale, grad_output,
                                              self.rois, grad_input)
        else:
            roi_align.roi_align_backward(self.aligned_height,
                                         self.aligned_width,
                                         self.time_dim,
                                         self.spatial_scale, self.temp_scale, grad_output,
                                         self.rois, grad_input)

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
