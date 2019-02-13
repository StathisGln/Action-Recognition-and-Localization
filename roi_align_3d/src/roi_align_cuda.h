int roi_align_forward_cuda(int aligned_height, int aligned_width, int time_dim, float spatial_scale, float temp_scale,
			   THCudaTensor * features, THCudaTensor * rois, THCudaTensor * output);

int roi_align_backward_cuda(int aligned_height, int aligned_width,  int time_dim, float spatial_scale, float temp_scale,
			    THCudaTensor * top_grad, THCudaTensor * rois, THCudaTensor * bottom_grad);
