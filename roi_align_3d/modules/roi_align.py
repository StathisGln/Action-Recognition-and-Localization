from torch.nn.modules.module import Module
<<<<<<< HEAD
from torch.nn.functional import avg_pool2d, max_pool2d, avg_pool3d, adaptive_max_pool3d,adaptive_avg_pool3d
=======
from torch.nn.functional import adaptive_max_pool3d, avg_pool2d, max_pool2d, avg_pool3d
>>>>>>> origin/anchors_3d
from ..functions.roi_align import RoIAlignFunction


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
<<<<<<< HEAD
    def __init__(self, aligned_height, aligned_width, time_dim,spatial_scale):
=======
    def __init__(self, aligned_height, aligned_width, time_dim, spatial_scale, temp_scale=1.):
>>>>>>> origin/anchors_3d
        super(RoIAlignAvg, self).__init__()

        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.time_dim = int(time_dim)
        self.spatial_scale = float(spatial_scale)
        self.temp_scale = float(temp_scale)
    def forward(self, features, rois):
<<<<<<< HEAD
        print('rois.shape :', rois.shape)
        print('self.aligned_width :', self.aligned_width, ' self.aligned_height :',self.aligned_height, 'self.time_dim :',self.time_dim)
        x =  RoIAlignFunction(self.aligned_height+1, self.aligned_width+1,self.time_dim+1,
                                self.spatial_scale)(features, rois )
        return adaptive_avg_pool3d(x,(self.aligned_height,self.aligned_width,self.time_dim))
=======
        # print('RoiAlignAvg ==> rois.shape :', rois.shape)
        x =  RoIAlignFunction(self.aligned_height+1, self.aligned_width+1,self.time_dim+1,
                                self.spatial_scale)(features, rois )
        return adaptive_max_pool3d(x,(self.aligned_height,self.aligned_width,self.time_dim))
>>>>>>> origin/anchors_3d

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
