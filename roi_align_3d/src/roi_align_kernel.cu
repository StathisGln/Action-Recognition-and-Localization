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

	    // if (index == 50 ){
	    //   printf("pw %d ph %d pt %d c %d n %d index %d \n",pw,ph,pt,c,n,index);
	    //   }

	    // get the rois
	    // if (index==278528) printf("278528 n : %d\n",n);
	    // if (index==278529) printf("278529 n : %d\n",n);
            float roi_batch_ind = bottom_rois[n * 7 + 0];
            float roi_start_w = bottom_rois[n * 7 + 1] * spatial_scale;
            float roi_start_h = bottom_rois[n * 7 + 2]  * spatial_scale;
	    float roi_start_t = bottom_rois[n * 7 + 3] * temp_scale;
            float roi_end_w = bottom_rois[n * 7 + 4] * spatial_scale;
            float roi_end_h = bottom_rois[n * 7 + 5] * spatial_scale;
	    float roi_end_t = bottom_rois[n * 7 + 6] * temp_scale;


            // // Force malformed ROIs to be 1x1
            float roi_width = fmaxf(roi_end_w - roi_start_w + 1., 0.);
            float roi_height = fmaxf(roi_end_h - roi_start_h + 1., 0.);
	    float roi_time = fmaxf(roi_end_t - roi_start_t + 1., 0.);
	    // if (index == 50){
	    //   printf("roi_width = %f, roi_height %f, roi_time %f\n",roi_width,roi_height,roi_time);
	    //   printf("aligned_height %d aligned_width %d time_dim %d\n", aligned_height, aligned_width, time_dim);
	    // }
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
	    // if (index == 50){
	    //   printf("h = %f, w %f, t %f\n",h,w,t);
	    // }

            int hstart = fminf(floor(h), height - 2);
            int wstart = fminf(floor(w), width - 2);
	    int tstart = fminf(floor(t), time - 2);

	    // if (index == 50){
	    //   printf("hstart = %d, wstart %d, tstart %d\n",hstart,wstart,tstart);
	    // }

            int img_start = roi_batch_ind * channels * time * height * width;
	    // if (index == 50){
	    //   printf("img_start %d\n",img_start);
	    // }

	    // if (index==50)
	    //   printf("h %f w %f t %f\n", h,w,t);
            // // trilinear interpolation = 2 bilinear interpolation + 1 linear interpolation
	    // if( index==50 ) printf("height : %d width %d time %d \n",height,width,time);
	    // if (t>= time)
	    //   printf("index : %d time : %d  t : %f",index, time, t );
	    
            if (h < 0 || h >= height || w < 0 || w >= width || t < 0 || t >= time) {
	      // if(index==50)
	      // 	printf("epaeeeee\n");
                top_data[index] = 0.;
            } else {
	      // if (index ==524225)
		// if (index ==50)
		//   printf("t %f tstart %d\n",t,tstart);
                float h_ratio = h - (float)(hstart);
                float w_ratio = w - (float)(wstart);
		float t_ratio = 0.0; //t - (float)(tstart);
		// if (index == 524225){
		//   printf("index %d h_ratio %f f_ratio %f t_ratio %f\n",index,h_ratio,w_ratio,t_ratio);
		// }
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
		
		// if (upleftfront < 0) printf("upleftfront :%d\n", upleftfront);
		// if (uprightfront < 0) printf("uprightfront :%d\n", uprightfront);
		// if (downleftfront < 0) printf("downleftfront :%d\n", downleftfront);
		// if (downrightfront < 0) printf("downrightfront :%d\n", downrightfront);

		// if (upleftback < 0) printf("upleftback :%d\n", upleftback);
		// if (uprightback < 0) printf("uprightback :%d\n", uprightback);
		// if (downleftback < 0) printf("downleftback :%d\n", downleftback);
		// if (downrightback < 0) printf("downrightback :%d\n", downrightback);
		// if (index ==50){
		//   printf("upleftfront :%d\n", upleftfront);
		//   printf("uprightfront :%d\n", uprightfront);
		//   printf("downleftfront :%d\n", downleftfront);
		//   printf("downrightfront :%d\n", downrightfront);

		//   printf("upleftback :%d\n", upleftback);
		//   printf("uprightback :%d\n", uprightback);
		//   printf("downleftback :%d\n", downleftback);
		//   printf("downrightback :%d\n", downrightback);
		// }
		float front_data = bottom_data[upleftfront] * (1. - h_ratio) * (1. - w_ratio)
                    + bottom_data[uprightfront] * (1. - h_ratio) * w_ratio
                    + bottom_data[downleftfront] * h_ratio * (1. - w_ratio)
                    + bottom_data[downrightfront] * h_ratio * w_ratio;

		float rear_data = bottom_data[upleftback] * (1. - h_ratio) * (1. - w_ratio)
                    + bottom_data[uprightback] * (1. - h_ratio) * w_ratio
                    + bottom_data[downleftback] * h_ratio * (1. - w_ratio)
                    + bottom_data[downrightback] * h_ratio * w_ratio;

		// if (front_data < 0) printf("front_data :%f\n", front_data);
		// if (rear_data < 0) printf("rear_data :%f\n", rear_data);
		// if( !(isfinite(front_data) || isfinite(rear_data)))
		//   printf("front_data :%f rear_data :%f\n", front_data,rear_data);
		// if (index== 50) {
		//   printf("front_data :%f\n", front_data);
		//   printf("rear_data :%f\n", rear_data);
		//   printf("t_ratio :%f\n", t_ratio);
		//   printf("front_data * (1 - t_ratio) + rear_data * t_ratio %f\n", front_data * (1 - t_ratio) + rear_data * t_ratio);
		// }
		top_data[index] = front_data * (1 - t_ratio) + rear_data * t_ratio;
		// if (index==50)
		//   printf("top_data[index] %f\n",top_data[index]);
		
		  
            }
        }
    }


  int ROIAlignForwardLaucher(const float* bottom_data, const float spatial_scale, const float temp_scale, const int num_rois,
			     const int height, const int width, const int time, const int channels, const int aligned_height,
			     const int aligned_width, const int time_dim, const float* bottom_rois, float* top_data,
			     cudaStream_t stream) {
        const int kThreadsPerBlock = 1024;
        const int output_size = num_rois * time_dim * aligned_height * aligned_width * channels;
        cudaError_t err;

	// printf("edw exw temp_scale %f \n", temp_scale);
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
	    float bin_size_t = roi_time / (time_dim - 1.);

            float h = (float)(ph) * bin_size_h + roi_start_h;
            float w = (float)(pw) * bin_size_w + roi_start_w;
	    float t = (float)(pt) * bin_size_t + roi_start_t;

            int hstart = fminf(floor(h), height - 2);
            int wstart = fminf(floor(w), width - 2);
	    int tstart = fminf(floor(t), time - 2);

            int img_start = roi_batch_ind * channels * time * height * width;

            if (!(h < 0 || h >= height || w < 0 || w >= width || t < 0 || t >= time)) {
                float h_ratio = h - (float)(hstart);
                float w_ratio = w - (float)(wstart);
		// float t_ratio = t - (float)(tstart);

		// for the front bilinear interpolation

                int upleftfront = img_start + ((c * time + tstart) *height + hstart) * width + wstart;
                int uprightfront = upleftfront + 1;

                int downleftfront = upleftfront + width;
                int downrightfront = downleftfront + 1;

		// // for the back bilinear interpolation
                // int upleftback = upleftfront + width * height;
                // int uprightback = upleftback + 1;

                // int downleftback = upleftback + width;
                // int downrightback = downleftback + 1;


		// TODO understand what it does
                atomicAdd(bottom_diff + upleftfront, top_diff[index] * (1. - h_ratio) * (1 - w_ratio) );
                atomicAdd(bottom_diff + uprightfront, top_diff[index] * (1. - h_ratio) * w_ratio );
                atomicAdd(bottom_diff + downleftfront, top_diff[index] * h_ratio * (1 - w_ratio) );
                atomicAdd(bottom_diff + downrightfront, top_diff[index] * h_ratio * w_ratio );

                // atomicAdd(bottom_diff + upleftback, top_diff[index] * (1. - h_ratio) * (1 - w_ratio) * t_ratio);
                // atomicAdd(bottom_diff + uprightback, top_diff[index] * (1. - h_ratio) * w_ratio * t_ratio);
                // atomicAdd(bottom_diff + downleftback, top_diff[index] * h_ratio * (1 - w_ratio) * t_ratio);
                // atomicAdd(bottom_diff + downrightback, top_diff[index] * h_ratio * w_ratio * t_ratio);
            }
        }
    }

  int ROIAlignBackwardLaucher(const float* top_diff, const float spatial_scale, const float temp_scale, const int batch_size,
			      const int num_rois, const int height, const int width, const int time, const int channels,
			      const int aligned_height, const int aligned_width, const int time_dim, const float* bottom_rois,
			      float* bottom_diff, cudaStream_t stream) {
        const int kThreadsPerBlock = 1024;
        const int output_size = num_rois * time_dim * aligned_height * aligned_width * channels;
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
