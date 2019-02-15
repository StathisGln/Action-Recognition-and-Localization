#include <TH/TH.h>
#include <math.h>
#include <omp.h>


void ROIAlignForwardCpu(const float* bottom_data, const float spatial_scale, const float temp_scale, const int num_rois,
			const int height, const int width,  const int time, const int channels,	const int aligned_height,
			const int aligned_width, const int time_dim, const float * bottom_rois,
			float* top_data);

void ROIAlignBackwardCpu(const float* top_diff, const float spatial_scale, const float temp_scale, const int num_rois,
			 const int height, const int width, const int time, const int channels, const int aligned_height,
			 const int aligned_width, const int time_dim, const float * bottom_rois,
			 float* bottom_diff);

int roi_align_forward(int aligned_height, int aligned_width, int time_dim, float spatial_scale, float temp_scale,
                     THFloatTensor * features, THFloatTensor * rois, THFloatTensor * output)
{
    //Grab the input tensor
    float * data_flat = THFloatTensor_data(features);
    float * rois_flat = THFloatTensor_data(rois);

    float * output_flat = THFloatTensor_data(output);

    // Number of ROIs
    int num_rois = THFloatTensor_size(rois, 0);
    int size_rois = THFloatTensor_size(rois, 1);
    if (size_rois != 7)
    {
        return 0;
    }

    // data time
    int data_time = THFloatTensor_size(features, 2);
    // data height
    int data_height = THFloatTensor_size(features, 3);
    // data width
    int data_width = THFloatTensor_size(features, 4);
    // Number of channels
    int num_channels = THFloatTensor_size(features, 1);

    // do ROIAlignForward
    ROIAlignForwardCpu(data_flat, spatial_scale, temp_scale, num_rois, data_height, data_width, data_time, num_channels,
		       aligned_height, aligned_width, time_dim, rois_flat, output_flat);

    return 1;
}

int roi_align_backward(int aligned_height, int aligned_width, int time_dim, float spatial_scale, float temp_scale,
                       THFloatTensor * top_grad, THFloatTensor * rois, THFloatTensor * bottom_grad)
{
    //Grab the input tensor
    float * top_grad_flat = THFloatTensor_data(top_grad);
    float * rois_flat = THFloatTensor_data(rois);

    float * bottom_grad_flat = THFloatTensor_data(bottom_grad);

    // Number of ROIs
    int num_rois = THFloatTensor_size(rois, 0);
    int size_rois = THFloatTensor_size(rois, 1);
    if (size_rois != 7)
    {
        return 0;
    }

    // batch size
    // int batch_size = THFloatTensor_size(bottom_grad, 0);
    // data time
    int data_time = THFloatTensor_size(bottom_grad, 2);
    // data height
    int data_height = THFloatTensor_size(bottom_grad, 3);
    // data width
    int data_width = THFloatTensor_size(bottom_grad, 4);
    // Number of channels
    int num_channels = THFloatTensor_size(bottom_grad, 1);

    // do ROIAlignBackward
    ROIAlignBackwardCpu(top_grad_flat, spatial_scale, temp_scale, num_rois, data_height,
			data_width,  data_time, num_channels, aligned_height, aligned_width,
			time_dim, rois_flat, bottom_grad_flat);

    return 1;
}

void ROIAlignForwardCpu(const float* bottom_data, const float spatial_scale, const float temp_scale, const int num_rois,
			const int height, const int width,  const int time, const int channels,	const int aligned_height,
			const int aligned_width, const int time_dim, const float * bottom_rois,
			float* top_data)
{
    const int output_size = num_rois * aligned_height * aligned_width * time * channels;

    int idx = 0;
    for (idx = 0; idx < output_size; ++idx)
    {
        // (n, c, ph, pw) is an element in the aligned output
        int pw = idx % aligned_width;
        int ph = (idx / aligned_width) % aligned_height;
	int pt = (idx / aligned_width) % time_dim;
        int c = (idx / aligned_width / aligned_height / time_dim) % channels;
        int n =  idx / aligned_width / aligned_height / time_dim / channels;

        float roi_batch_ind = bottom_rois[n * 7 + 0];
        float roi_start_w = bottom_rois[n * 7 + 1] * spatial_scale;
        float roi_start_h = bottom_rois[n * 7 + 2] * spatial_scale;
	float roi_start_t = bottom_rois[n * 7 + 3] * temp_scale;
        float roi_end_w = bottom_rois[n * 7 + 4] * spatial_scale;
        float roi_end_h = bottom_rois[n * 7 + 5] * spatial_scale;
	float roi_end_t = bottom_rois[n * 7 + 6] * temp_scale;

        // Force malformed ROI to be 1x1
        float roi_width = fmaxf(roi_end_w - roi_start_w + 1., 0.);
        float roi_height = fmaxf(roi_end_h - roi_start_h + 1., 0.);
	float roi_time = fmaxf(roi_end_t - roi_start_t + 1., 0.);
        float bin_size_h = roi_height / (aligned_height - 1.);
        float bin_size_w = roi_width / (aligned_width - 1.);
	float bin_size_t = roi_time / (time_dim - 1.);

        float h = (float)(ph) * bin_size_h + roi_start_h;
        float w = (float)(pw) * bin_size_w + roi_start_w;
	float t = (float)(pt) * bin_size_t + roi_start_t;

        int hstart = fminf(floor(h), height - 2);
        int wstart = fminf(floor(w), width - 2);
	int tstart = fminf(floor(t), time - 2);

        int img_start = roi_batch_ind * channels * time * height * width;

        // trilinear interpolation
        if (h < 0 || h >= height || w < 0 || w >= width || t < 0 || t >= time)
        {
            top_data[idx] = 0.;
        }
        else
        {
            float h_ratio = h - (float)(hstart);
            float w_ratio = w - (float)(wstart);
	    float t_ratio = t - (float)(tstart);

	    // for the front bilinear interpolation
	    int upleftfront = img_start + ((c * time + tstart) *height + hstart) * width + wstart;
	    int uprightfront = upleftfront + 1;

	    int downleftfront = upleftfront + width;
	    int downrightfront = downleftfront + 1;

	    // for the back bilinear interpolation
	    int upleftback = upleftfront + width * height;
	    int uprightback = upleftback + 1;

	    int downleftback = upleftback + width;
	    int downrightback = downleftback + 1;

	    float front_data = bottom_data[upleftfront] * (1. - h_ratio) * (1. - w_ratio)
	      + bottom_data[uprightfront] * (1. - h_ratio) * w_ratio
	      + bottom_data[downleftfront] * h_ratio * (1. - w_ratio)
	      + bottom_data[downrightfront] * h_ratio * w_ratio;

	    float read_data = bottom_data[upleftback] * (1. - h_ratio) * (1. - w_ratio)
	      + bottom_data[uprightback] * (1. - h_ratio) * w_ratio
	      + bottom_data[downleftback] * h_ratio * (1. - w_ratio)
	      + bottom_data[downrightback] * h_ratio * w_ratio;

	    top_data[idx] = front_data * (1 - t_ratio) + read_data * t_ratio;
        }
    }
}

void ROIAlignBackwardCpu(const float* top_diff, const float spatial_scale, const float temp_scale, const int num_rois,
			 const int height, const int width, const int time, const int channels, const int aligned_height,
			 const int aligned_width, const int time_dim, const float * bottom_rois,
			 float* bottom_diff)
{
    const int output_size = num_rois * aligned_height * aligned_width * channels;

    int idx = 0;
    for (idx = 0; idx < output_size; ++idx)
    {
        // (n, c, ph, pw) is an element in the aligned output
        int pw = idx % aligned_width;
        int ph = (idx / aligned_width) % aligned_height;
	int pt = (idx / aligned_width / aligned_height)% time ;
        int c  = (idx / aligned_width / aligned_height / time ) % channels;
        int n  =  idx / aligned_width / aligned_height / time / channels;

        float roi_batch_ind = bottom_rois[n * 7 + 0];
        float roi_start_w = bottom_rois[n * 7 + 1] * spatial_scale;
        float roi_start_h = bottom_rois[n * 7 + 2] * spatial_scale;
	float roi_start_t = bottom_rois[n * 7 + 3] * temp_scale;
        float roi_end_w = bottom_rois[n * 7 + 4] * spatial_scale;
        float roi_end_h = bottom_rois[n * 7 + 5] * spatial_scale;
	float roi_end_t = bottom_rois[n * 7 + 5] * temp_scale;

        // Force malformed ROI to be 1x1
        float roi_width = fmaxf(roi_end_w - roi_start_w + 1., 0.);
        float roi_height = fmaxf(roi_end_h - roi_start_h + 1., 0.);
	float roi_time = fmaxf(roi_end_t - roi_start_t + 1., 0.);
        float bin_size_h = roi_height / (aligned_height - 1.);
        float bin_size_w = roi_width / (aligned_width - 1.);
	float bin_size_t = roi_time / (time_dim - 1.);

        float h = (float)(ph) * bin_size_h + roi_start_h;
        float w = (float)(pw) * bin_size_w + roi_start_w;
	float t = (float)(pt) * bin_size_t + roi_start_t;

        int hstart = fminf(floor(h), height - 2);
        int wstart = fminf(floor(w), width - 2);
	int tstart = fminf(floor(t), time - 2);

        int img_start = roi_batch_ind * channels * time * height * width;

        // bilinear interpolation
        if (h < 0 || h >= height || w < 0 || w >= width || t < 0 || t >= time)
        {
            float h_ratio = h - (float)(hstart);
            float w_ratio = w - (float)(wstart);
	    float t_ratio = t - (float)(wstart);

            /* int upleft = img_start + (c * height + hstart) * width + wstart; */
            /* int upright = upleft + 1; */
            /* int downleft = upleft + width; */
            /* int downright = downleft + 1; */

	    // for the front bilinear interpolation
	    int upleftfront = img_start + ((c * time + tstart) *height + hstart) * width + wstart;
	    int uprightfront = upleftfront + 1;

	    int downleftfront = upleftfront + width;
	    int downrightfront = downleftfront + 1;

	    // for the back bilinear interpolation
	    int upleftback = upleftfront + width * height;
	    int uprightback = upleftback + 1;

	    int downleftback = upleftback + width;
	    int downrightback = downleftback + 1;

            bottom_diff[upleftfront] += top_diff[idx] * (1. - h_ratio) * (1. - w_ratio) * (1 - t_ratio);
            bottom_diff[uprightfront] += top_diff[idx] * (1. - h_ratio) *  w_ratio * (1 - t_ratio);
            bottom_diff[downleftfront] += top_diff[idx] * h_ratio * (1. - w_ratio) * (1 - t_ratio);
            bottom_diff[downrightfront] += top_diff[idx] * h_ratio * w_ratio * (1 - t_ratio);

            bottom_diff[upleftback] += top_diff[idx] * (1. - h_ratio) * (1. - w_ratio) * t_ratio;
            bottom_diff[uprightback] += top_diff[idx] * (1. - h_ratio) *  w_ratio * t_ratio;
            bottom_diff[downleftback] += top_diff[idx] * h_ratio * (1. - w_ratio) * t_ratio;
            bottom_diff[downrightback] += top_diff[idx] * h_ratio * w_ratio * t_ratio;

        }
    }
}
