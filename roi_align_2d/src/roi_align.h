int roi_align_forward(int aligned_height, int aligned_width, int time_dim, float spatial_scale, float temp_scale,
		      THFloatTensor * features, THFloatTensor * rois, THFloatTensor * output);

int roi_align_backward(int aligned_height, int aligned_width, int time_dim, float spatial_scale, float temp_scale,
                       THFloatTensor * top_grad, THFloatTensor * rois, THFloatTensor * bottom_grad);
