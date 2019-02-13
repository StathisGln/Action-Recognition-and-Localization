#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <math.h>
#include <float.h>
#include "roi_align_kernel.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
            i += blockDim.x * gridDim.x)


  __global__ void ROIAlignForward(const int nthreads, const float* bottom_data, const float spatial_scale, const float temp_scale,
				  const int height, const int width, const int time, const int channels, const int aligned_height,
				  const int aligned_width, const int time_dim, const float* bottom_rois, float* top_data) {
        CUDA_1D_KERNEL_LOOP(index, nthreads) {
            // (n, c, t, ph, pw) is an element in the aligned output
            // int n = index;
            // int pw = n % aligned_width;
            // n /= aligned_width;
            // int ph = n % aligned_height;
            // n /= aligned_height;
            // int c = n % channels;
            // n /= channels;

            int pw = index % aligned_width;
            int ph = (index / aligned_width) % aligned_height;
	    int pt  = (index / aligned_width / aligned_height) % time_dim;
            int c  = (index / aligned_width / aligned_height / time_dim) % channels;
            int n  =  index / aligned_width / aligned_height / time_dim  / channels;

            // bottom_rois += n * 5;
            float roi_batch_ind = bottom_rois[n * 7 + 0];
            float roi_start_w = bottom_rois[n * 7 + 1] * spatial_scale;
            float roi_start_h = bottom_rois[n * 7 + 2] * spatial_scale;
	    float roi_start_t = bottom_rois[n * 7 + 3] * temp_scale;
            float roi_end_w = bottom_rois[n * 7 + 4] * spatial_scale;
            float roi_end_h = bottom_rois[n * 7 + 5] * spatial_scale;
	    float roi_end_t = bottom_rois[n * 7 + 6] * temp_scale;

            // Force malformed ROIs to be 1x1
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

            // trilinear interpolation = 2 bilinear interpolation + 1 linear interpolation
            if (h < 0 || h >= height || w < 0 || w >= width || t < 0 || w >= time) {
                top_data[index] = 0.;
            } else {
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

		top_data[index] = front_data * (1 - t_ratio) + read_data * t_ratio;

		  
            }
        }
    }


  int ROIAlignForwardLaucher(const float* bottom_data, const float spatial_scale, const float temp_scale, const int num_rois,
			     const int height, const int width, const int time, const int channels, const int aligned_height,
			     const int aligned_width, const int time_dim, const float* bottom_rois, float* top_data,
			     cudaStream_t stream) {
        const int kThreadsPerBlock = 1024;
        const int output_size = num_rois * aligned_height * aligned_width * channels;
        cudaError_t err;


        ROIAlignForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
														  output_size, bottom_data, spatial_scale, temp_scale, height,
														  width, time, channels, aligned_height, aligned_width,
														  time_dim, bottom_rois, top_data);

        err = cudaGetLastError();
        if(cudaSuccess != err) {
            fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
            exit( -1 );
        }

        return 1;
    }


  __global__ void ROIAlignBackward(const int nthreads, const float* top_diff, const float spatial_scale, const float temp_scale,
				   const int height, const int width, const int time, const int channels, const int aligned_height,
				   const int aligned_width, const int time_dim, float* bottom_diff, const float* bottom_rois) {
        CUDA_1D_KERNEL_LOOP(index, nthreads) {

            // (n, c, ph, pw) is an element in the aligned output
            int pw = index % aligned_width;
            int ph = (index / aligned_width) % aligned_height;
	    int pt = (index / aligned_width / aligned_height) % time_dim;
            int c  = (index / aligned_width / aligned_height  / time_dim) % channels;
            int n  =  index / aligned_width / aligned_height  / time_dim  / channels;

            float roi_batch_ind = bottom_rois[n * 7 + 0];
            float roi_start_w = bottom_rois[n * 7 + 1] * spatial_scale;
            float roi_start_h = bottom_rois[n * 7 + 2] * spatial_scale;
	    float roi_start_t = bottom_rois[n * 7 + 3] * temp_scale;
            float roi_end_w = bottom_rois[n * 7 + 4] * spatial_scale;
            float roi_end_h = bottom_rois[n * 7 + 5] * spatial_scale;
	    float roi_end_t = bottom_rois[n * 7 + 6] * temp_scale;
            /* int roi_start_w = round(bottom_rois[1] * spatial_scale); */
            /* int roi_start_h = round(bottom_rois[2] * spatial_scale); */
            /* int roi_end_w = round(bottom_rois[3] * spatial_scale); */
            /* int roi_end_h = round(bottom_rois[4] * spatial_scale); */

            // Force malformed ROIs to be 1x1
            float roi_width = fmaxf(roi_end_w - roi_start_w + 1., 0.);
            float roi_height = fmaxf(roi_end_h - roi_start_h + 1., 0.);
	    float roi_time = fmaxf(roi_end_t - roi_start_t + 1., 0.);
            float bin_size_h = roi_height / (aligned_height - 1.);
            float bin_size_w = roi_width / (aligned_width - 1.);
	    float bin_size_t = roi_width / (time_dim - 1.);

            float h = (float)(ph) * bin_size_h + roi_start_h;
            float w = (float)(pw) * bin_size_w + roi_start_w;
	    float t = (float)(pt) * bin_size_t + roi_start_t;

            int hstart = fminf(floor(h), height - 2);
            int wstart = fminf(floor(w), width - 2);
	    int tstart = fminf(floor(t), time - 2);

            int img_start = roi_batch_ind * channels * time * height * width;

            // bilinear interpolation
            if (!(h < 0 || h >= height || w < 0 || w >= width || t < 0 || t >= time)) {
                float h_ratio = h - (float)(hstart);
                float w_ratio = w - (float)(wstart);
		float t_ratio = t - (float)(tstart);

                // int upleft = img_start + (c * height + hstart) * width + wstart;
                // int upright = upleft + 1;
                // int downleft = upleft + width;
                // int downright = downleft + 1;

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


		// TODO understand what it does
                atomicAdd(bottom_diff + upleftfront, top_diff[index] * (1. - h_ratio) * (1 - w_ratio) * (1- t_ratio));
                atomicAdd(bottom_diff + uprightfront, top_diff[index] * (1. - h_ratio) * w_ratio * (1- t_ratio));
                atomicAdd(bottom_diff + downleftfront, top_diff[index] * h_ratio * (1 - w_ratio) * (1- t_ratio));
                atomicAdd(bottom_diff + downrightfront, top_diff[index] * h_ratio * w_ratio *(1- t_ratio));

                atomicAdd(bottom_diff + upleftback, top_diff[index] * (1. - h_ratio) * (1 - w_ratio) * t_ratio);
                atomicAdd(bottom_diff + uprightback, top_diff[index] * (1. - h_ratio) * w_ratio * t_ratio);
                atomicAdd(bottom_diff + downleftback, top_diff[index] * h_ratio * (1 - w_ratio) * t_ratio);
                atomicAdd(bottom_diff + downrightback, top_diff[index] * h_ratio * w_ratio * t_ratio);
            }
        }
    }

  int ROIAlignBackwardLaucher(const float* top_diff, const float spatial_scale, const float temp_scale, const int batch_size,
			      const int num_rois, const int height, const int width, const int time, const int channels,
			      const int aligned_height, const int aligned_width, const int time_dim, const float* bottom_rois,
			      float* bottom_diff, cudaStream_t stream) {
        const int kThreadsPerBlock = 1024;
        const int output_size = num_rois * aligned_height * aligned_width * channels;
        cudaError_t err;

        ROIAlignBackward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
														   output_size, top_diff, spatial_scale, temp_scale,
														   height, width, time, channels, aligned_height,
														   aligned_width, time_dim, bottom_diff, bottom_rois);

        err = cudaGetLastError();
        if(cudaSuccess != err) {
            fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
            exit( -1 );
        }

        return 1;
    }


#ifdef __cplusplus
}
#endif
