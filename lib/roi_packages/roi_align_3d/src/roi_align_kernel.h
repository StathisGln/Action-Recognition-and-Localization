#ifndef _ROI_ALIGN_KERNEL
#define _ROI_ALIGN_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

  __global__ void ROIAlignForward(const int nthreads, const float* bottom_data, const float spatial_scale, const float temp_scale,
				  const int height, const int width, const int time, const int channels, const int aligned_height,
				  const int aligned_width, const int time_dim, const float* bottom_rois, float* top_data);

  
  int ROIAlignForwardLaucher(const float* bottom_data, const float spatial_scale, const float temp_scale, const int num_rois,
			     const int height, const int width, const int time, const int channels, const int aligned_height,
			     const int aligned_width, const int time_dim, const float* bottom_rois, float* top_data,
			     cudaStream_t stream);

  __global__   void ROIAlignBackward(const int nthreads, const float* top_diff, const float spatial_scale, const float temp_scale,
				     const int height, const int width, const int time, const int channels, const int aligned_height,
				     const int aligned_width, const int time_dim, float* bottom_diff, const float* bottom_rois);

  int ROIAlignBackwardLaucher(const float* top_diff, const float spatial_scale, const float temp_scale, const int batch_size,
			      const int num_rois, const int height, const int width, const int time, const int channels,
			      const int aligned_height, const int aligned_width, const int time_dim, const float* bottom_rois,
			      float* bottom_diff, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
