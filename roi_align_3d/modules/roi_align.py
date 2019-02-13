from torch.nn.modules.module import Module
from torch.nn.functional import avg_pool2d, max_pool2d, avg_pool3d
from ..functions.roi_align import RoIAlignFunction, RoIAlignFunction_3d


class RoIAlign(Module):
    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoIAlign, self).__init__()

        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return RoIAlignFunction(self.aligned_height, self.aligned_width,
                                self.spatial_scale)(features, rois)

class RoIAlignAvg(Module):
    def __init__(self, aligned_height, aligned_width, time_dim, spatial_scale):
        super(RoIAlignAvg, self).__init__()

        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.time_dim = int(time_dim)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        print('features.shape :',features.shape )
        print('rois.shape :',rois.shape)
        print('self.time_dim :',self.time_dim)
        x =  RoIAlignFunction(self.aligned_height+1, self.aligned_width+1, self.time_dim+1,
                                self.spatial_scale)(features, rois )
        print('ret x.shape :',x.shape)
        ret = avg_pool3d(x, kernel_size=2, stride=1)
        print('ret.shape :',ret.shape)
        return ret


class RoIAlignAvg_3d(Module):
    def __init__(self, aligned_height, aligned_width, aligned_time, spatial_scale, time_scale):
        super(RoIAlignAvg, self).__init__()

        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.aligned_time = int(aligned_time)
        self.spatial_scale = float(spatial_scale)
        self.time_scale = float(time_scale)

    def forward(self, features, rois):
        x =  RoIAlignFunction_3d(self.aligned_height+1, self.aligned_width+1, self.aligned_time+1,
                                 self.spatial_scale,self.time_scale)(features, rois )
        return avg_pool3d(x, kernel_size=2, stride=1)


class RoIAlignMax(Module):
    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoIAlignMax, self).__init__()

        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        x =  RoIAlignFunction(self.aligned_height+1, self.aligned_width+1,
                                self.spatial_scale)(features, rois)
        return max_pool2d(x, kernel_size=2, stride=1)
