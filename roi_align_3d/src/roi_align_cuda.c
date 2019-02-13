#include <THC/THC.h>
#include <math.h>
#include "roi_align_kernel.h"
#include <stdio.h>

extern THCState *state;

int roi_align_forward_cuda(int aligned_height, int aligned_width, int time_dim, float spatial_scale,
                        THCudaTensor * features, THCudaTensor * rois, THCudaTensor * output)
{
    // Grab the input tensor
  /* printf(' aligned_height %d aligned_widht %d spatial_scale %d',aligned_height, aligned_width, spatial_scale); */

    float * data_flat = THCudaTensor_data(state, features);
    float * rois_flat = THCudaTensor_data(state, rois);

    float * output_flat = THCudaTensor_data(state, output);

    // Number of ROIs
    int num_rois = THCudaTensor_size(state, rois, 0);
    int size_rois = THCudaTensor_size(state, rois, 1);
    if (size_rois != 7)
    {
      return 0;
    }

    // data time
    int data_time = THCudaTensor_size(state, features, 2);
    // data height
    int data_height = THCudaTensor_size(state, features, 3);
    // data width
    int data_width = THCudaTensor_size(state, features, 4);
    // Number of channels
    int num_channels = THCudaTensor_size(state, features, 1);

    cudaStream_t stream = THCState_getCurrentStream(state);

    ROIAlignForwardLaucher(
			   data_flat, spatial_scale, num_rois, data_time, data_height,
			   data_width, num_channels, aligned_height,
			   aligned_width, rois_flat,
			   output_flat, stream);

    return 1;
}

int roi_align_backward_cuda(int aligned_height, int aligned_width, int time_dim, float spatial_scale,
                        THCudaTensor * top_grad, THCudaTensor * rois, THCudaTensor * bottom_grad)
{
    // Grab the input tensor
    float * top_grad_flat = THCudaTensor_data(state, top_grad);
    float * rois_flat = THCudaTensor_data(state, rois);

    float * bottom_grad_flat = THCudaTensor_data(state, bottom_grad);

    // Number of ROIs
    int num_rois = THCudaTensor_size(state, rois, 0);
    int size_rois = THCudaTensor_size(state, rois, 1);
    if (size_rois != 7)
    {
        return 0;
    }

    // batch size
    int batch_size = THCudaTensor_size(state, bottom_grad, 0);
    // data time
    int data_time = THCudaTensor_size(state, bottom_grad, 2);
    // data height
    int data_height = THCudaTensor_size(state, bottom_grad, 3);
    // data width
    int data_width = THCudaTensor_size(state, bottom_grad, 4);
    // Number of channels
    int num_channels = THCudaTensor_size(state, bottom_grad, 1);

    cudaStream_t stream = THCState_getCurrentStream(state);
    ROIAlignBackwardLaucher(
			    top_grad_flat, spatial_scale, batch_size, num_rois, data_time, data_height,
			    data_width, num_channels, aligned_height,
			    aligned_width, rois_flat,
			    bottom_grad_flat, stream);

    return 1;
}
