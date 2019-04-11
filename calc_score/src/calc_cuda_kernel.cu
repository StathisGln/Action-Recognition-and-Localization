#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <math.h>
#include <float.h>
#include "calc_cuda_kernel.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
            i += blockDim.x * gridDim.x)


    __global__ void Calculate_scores(const int nthreads,const int K, const int N, const float thresh, const int array_size,
				    const int* pos, const int* pos_indices, const float *actioness, const float *overlaps_scr,
				    const float *scores, const float *overlaps, const int indx, int *next_pos,
				    int *next_pos_indices,float *next_actioness, float *next_overlaps_scr, float *f_scores) {
        CUDA_1D_KERNEL_LOOP(index, nthreads) {

	  float tmp_sum;
	  float new_score, new_overlap;
	  int idx_i, idx_j, m, tmp_pos;
	  int j, z;

	  j = index / K;
	  z = index % K;

	  idx_i = pos[j * N * 2 + pos_indices[j] * 2];
	  idx_j = pos[j * N * 2 + pos_indices[j] * 2 + 1];

	  new_score = actioness[j] + scores[z];
	  new_overlap = overlaps_scr[j] + overlaps[idx_i * K * K + idx_j * K + z];

	  m = pos_indices[j] + 1;

	  tmp_sum = new_score / (m+1) +  new_overlap / m; // 0.8
	  tmp_pos = j * K + z;
	  
	  if (tmp_sum > thresh){
	    next_pos_indices[tmp_pos] = pos_indices[j] + 1;
	    next_actioness[tmp_pos] = new_score;
	    next_overlaps_scr[tmp_pos] = new_overlap;
	    f_scores[tmp_pos] = tmp_sum;

	  }else{
	    next_pos_indices[tmp_pos] = -1;
	    next_actioness[tmp_pos] = -1;
	    next_overlaps_scr[tmp_pos] = -1;
	    f_scores[tmp_pos] = -1;
	  }
	}
    }


    int CalculationLaucher(const int K, const int N, const float thresh, const int array_size, const int* pos, const int* pos_indices,
			   const float *actioness, const float *overlaps_scr, const float *scores, const float *overlaps, const int indx,
			   int *next_pos, int *next_pos_indices,float *next_actioness, float *next_overlaps_scr, float *f_scores,
			   cudaStream_t stream) {
        const int kThreadsPerBlock = 1024;
        // const int kThreadsPerBlock = 64;

        const int output_size = array_size * K;
        cudaError_t err;

        Calculate_scores <<< (output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream >>>(
          output_size, K, N, thresh, array_size, pos, pos_indices, actioness, overlaps_scr, scores, overlaps, indx,
	  next_pos, next_pos_indices, next_actioness, next_overlaps_scr, f_scores);

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
