#include <THC/THC.h>
#include <math.h>
#include "roi_align_kernel.h"
#include <stdio.h>
extern THCState *state;

int roi_align_forward_cuda(int aligned_height, int aligned_width, int time_dim, float spatial_scale, float temp_scale,
                        THCudaTensor * features, THCudaTensor * rois, THCudaTensor * output)
{
    // Grab the input tensor
    float * data_flat = THCudaTensor_data(state, features);
    float * rois_flat = THCudaTensor_data(state, rois);

    float * output_flat = THCudaTensor_data(state, output);
    /* printf("-----Inside roi_align_cuda_c-----\n"); */
    /* printf("spatial_scale : %f temp_scale %f\n", spatial_scale,temp_scale); */
    /* // Number of ROIs */
    int num_rois = THCudaTensor_size(state, rois, 0);
    int size_rois = THCudaTensor_size(state, rois, 1);
    /* printf("size_rois : %d\n",size_rois); */
    /* printf("num_rois : %d\n",num_rois); */
    if (size_rois != 7)
    {
        return 0;
    }

    // data height
    int data_time = THCudaTensor_size(state, features, 2);
    /* printf("data_time %d\n",data_time); */
    // data height
    int data_height = THCudaTensor_size(state, features, 3);
    /* printf("data_height %d\n",data_height); */
    // data width
    int data_width = THCudaTensor_size(state, features, 4);
    /* printf("data_width %d\n", data_width); */
    // Number of channels
    int num_channels = THCudaTensor_size(state, features, 1);
    /* printf("num_channels %d\n",num_channels); */
    /* printf("data_time %d data_height %d data width %d num_channels %d\n",data_time,data_height,data_width,num_channels); */
    cudaStream_t stream = THCState_getCurrentStream(state);
    /* printf("temp_scale %f, spatial_scale %f\n",temp_scale,spatial_scale); */
    ROIAlignForwardLaucher(
			   data_flat, spatial_scale, temp_scale, num_rois, data_height,
			   data_width, data_time, num_channels, aligned_height,
			   aligned_width, time_dim, rois_flat,
			   output_flat, stream);

    return 1;
}

int roi_align_backward_cuda(int aligned_height, int aligned_width,  int time_dim, float spatial_scale, float temp_scale,
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
    // data height
    int data_time = THCudaTensor_size(state, bottom_grad, 2);
    // data height
    int data_height = THCudaTensor_size(state, bottom_grad, 3);
    // data width
    int data_width = THCudaTensor_size(state, bottom_grad, 4);
    // Number of channels
    int num_channels = THCudaTensor_size(state, bottom_grad, 1);

    cudaStream_t stream = THCState_getCurrentStream(state);
    ROIAlignBackwardLaucher(
			    top_grad_flat, spatial_scale, temp_scale, batch_size, num_rois, data_height,
			    data_width, data_time, num_channels, aligned_height,
			    aligned_width, time_dim, rois_flat,
			    bottom_grad_flat, stream);

    return 1;
}
