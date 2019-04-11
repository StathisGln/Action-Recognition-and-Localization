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


  __global__ void ROIAlignForward(const int nthreads, const float* bottom_data, const float spatial_scale,
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

	    // if (index == 50 ){
	    //   printf("pw %d ph %d pt %d c %d n %d index %d \n",pw,ph,pt,c,n,index);
	    //   }

	    // get the rois

            float roi_batch_ind = bottom_rois[n * 7 + 0];
            float roi_start_w = bottom_rois[n * 7 + 1] * spatial_scale;
            float roi_start_h = bottom_rois[n * 7 + 2]  * spatial_scale;
	    float roi_start_t = bottom_rois[n * 7 + 3];
            float roi_end_w = bottom_rois[n * 7 + 4] * spatial_scale;
            float roi_end_h = bottom_rois[n * 7 + 5] * spatial_scale;
	    float roi_end_t = bottom_rois[n * 7 + 6];

	    // if(index == 50){
	    //   printf("0 n :%d roi_start_h : %f %f %f %f %f %f\n",n, float(roi_start_h),
	    // 	     float( roi_start_w), float( roi_start_t),
	    // 	     float( roi_end_w), roi_end_h,
	    // 	     float( roi_end_t));
	    // } 

            // // Force malformed ROIs to be 1x1
            float roi_width = fmaxf(roi_end_w - roi_start_w + 1., 0.);
            float roi_height = fmaxf(roi_end_h - roi_start_h + 1., 0.);
	    float roi_time = fmaxf(roi_end_t - roi_start_t + 1., 0.);
	    if (index == 50){
	      printf("roi_width = %f, roi_height %f, roi_time %f\n",roi_width,roi_height,roi_time);
	      printf("aligned_height %d aligned_width %d time_dim %d\n", aligned_height, aligned_width, time_dim);
	    }
            float bin_size_h = roi_height / (aligned_height - 1.);
            float bin_size_w = roi_width / (aligned_width - 1.);
	    float bin_size_t = roi_time / (time_dim - 1);

	    // if (index == 50){
	    //   printf("bin_size_h = %f, bin_size_w %f, bin_size_t %f\n",bin_size_h,bin_size_w,bin_size_t);
	    //   printf("aligned_height = %d, aligned_width %d, time_dim %d\n",aligned_height, aligned_width, time_dim);
	    // }

            float h = (float)(ph) * bin_size_h + roi_start_h;
            float w = (float)(pw) * bin_size_w + roi_start_w;
	    float t = (float)(pt) * bin_size_t + roi_start_t;

            int hstart = fminf(floor(h), height - 2);
            int wstart = fminf(floor(w), width - 2);
	    int tstart = fminf(floor(t), time - 2);

            int img_start = roi_batch_ind * channels * time * height * width;

	    // if (index == 50){
	    //   printf("roi_start_t %f\n",roi_start_t);
	    //   printf("pt :%d\n",pt);
	    //   printf("time %d t %f tstart %d\n",time, t, tstart);
	    // }
	    // if (index == 150){
	    //   printf("roi_start_t %f\n",roi_start_t);
	    //   printf("pt :%d\n",pt);
	    //   printf("time %d t %f tstart %d\n",time, t, tstart);
	    // }

            if (h < 0 || h >= height || w < 0 || w >= width || t < 0 || t >= time) {

                top_data[index] = 0.;

            } else {

                float h_ratio = h - (float)(hstart);
                float w_ratio = w - (float)(wstart);

                int upleft = img_start + ((c * time + tstart) *height + hstart) * width + wstart;
                int upright = upleft + 1;

                int downleft = upleft + width;
                int downright = downleft + 1;

        	top_data[index] = bottom_data[upleft] * (1. - h_ratio) * (1. - w_ratio)
		  + bottom_data[upright] * (1. - h_ratio) * w_ratio
		  + bottom_data[downleft] * h_ratio * (1. - w_ratio)
		  + bottom_data[downright] * h_ratio * w_ratio;
            }
        }
    }


  int ROIAlignForwardLaucher(const float* bottom_data, const float spatial_scale,  const int num_rois,
			     const int height, const int width, const int time, const int channels, const int aligned_height,
			     const int aligned_width, const int time_dim, const float* bottom_rois, float* top_data,
			     cudaStream_t stream) {
        const int kThreadsPerBlock = 1024;
        const int output_size = num_rois * time_dim * aligned_height * aligned_width * channels;
        cudaError_t err;


        ROIAlignForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
														  output_size, bottom_data, spatial_scale,  height,
														  width, time, channels, aligned_height, aligned_width,
														  time_dim, bottom_rois, top_data);

        err = cudaGetLastError();
        if(cudaSuccess != err) {
            fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
            exit( -1 );
        }
        return 1;
    }


  __global__ void ROIAlignBackward(const int nthreads, const float* top_diff, const float spatial_scale, 
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
	    float roi_start_t = bottom_rois[n * 7 + 3];
            float roi_end_w = bottom_rois[n * 7 + 4] * spatial_scale;
            float roi_end_h = bottom_rois[n * 7 + 5] * spatial_scale;
	    float roi_end_t = bottom_rois[n * 7 + 6];

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

            // bilinear interpolation
            if (!(h < 0 || h >= height || w < 0 || w >= width || t < 0 || t >= time)) {

                float h_ratio = h - (float)(hstart);
                float w_ratio = w - (float)(wstart);

		// for the front bilinear interpolation

                int upleft = img_start + ((c * time + tstart) *height + hstart) * width + wstart;
                int upright = upleft + 1;

                int downleft = upleft + width;
                int downright = downleft + 1;

		// TODO understand what it does
                atomicAdd(bottom_diff + upleft, top_diff[index] * (1. - h_ratio) * (1 - w_ratio));
                atomicAdd(bottom_diff + upright, top_diff[index] * (1. - h_ratio) * w_ratio);
                atomicAdd(bottom_diff + downleft, top_diff[index] * h_ratio * (1 - w_ratio));
                atomicAdd(bottom_diff + downright, top_diff[index] * h_ratio * w_ratio);

            }
        }
    }

  int ROIAlignBackwardLaucher(const float* top_diff, const float spatial_scale, const int batch_size,
			      const int num_rois, const int height, const int width, const int time, const int channels,
			      const int aligned_height, const int aligned_width, const int time_dim, const float* bottom_rois,
			      float* bottom_diff, cudaStream_t stream) {
        const int kThreadsPerBlock = 1024;
        const int output_size = num_rois * time_dim * aligned_height * aligned_width * channels;
        cudaError_t err;

        ROIAlignBackward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
														   output_size, top_diff, spatial_scale,
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
