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


  __global__ void Calculate_scores(const int nthreads, const int K, const int N, const int n_frames, const int n_combs, const int sample_duration, const int step,  const float *p_tubes,
				   const int* combinations, float *ret_tubes) {
        CUDA_1D_KERNEL_LOOP(index, nthreads) {

	  int i,j;
	  int idx_i, idx_j;
	  int start_fr;
	  int comb_index, p_tube_index_i,p_tube_index_j, r_tube_index;
	  
	  comb_index = index * N * 2;
	  p_tube_index_i = K * sample_duration * 4; // for easy idx_i
	  p_tube_index_j = sample_duration * 4; // for easy idx_i
	  r_tube_index = index * n_frames * 4;
	   
	  for ( i=0; i<N; i++){

	    idx_i = combinations[comb_index + i * 2];
	    idx_j = combinations[comb_index + i * 2 + 1];

	    if (idx_i != -1){

	      start_fr = idx_i * step * 4;

	      for (j = 0; j<sample_duration; j++){

		if (j + idx_i * step >= n_frames)
		  break;

	      	ret_tubes[r_tube_index+start_fr + j * 4    ] = p_tubes[idx_i * p_tube_index_i + idx_j * p_tube_index_j +j * 4];
	      	ret_tubes[r_tube_index+start_fr + j * 4 + 1] = p_tubes[idx_i * p_tube_index_i + idx_j * p_tube_index_j + j * 4 + 1];
		ret_tubes[r_tube_index+start_fr + j * 4 + 2] = p_tubes[idx_i * p_tube_index_i + idx_j * p_tube_index_j + j * 4 + 2];
		ret_tubes[r_tube_index+start_fr + j * 4 + 3] = p_tubes[idx_i * p_tube_index_i + idx_j * p_tube_index_j + j * 4 + 3];
	      }
	      // else
	      // 	break;
	    }
	  }
	}
    }


	int CalculationLaucher(const int K, const int N, const int n_frames, const int n_combs, const int sample_duration, const int step, const float *p_tubes, const int *combinations,
			 float *ret_tubes,  cudaStream_t stream) {
        const int kThreadsPerBlock = 1024;
        // const int kThreadsPerBlock = 64;

        const int output_size =  n_combs;
        cudaError_t err;
	// printf("output_size %d\n",output_size);
        Calculate_scores <<< (output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream >>>(
        output_size, K, N, n_frames, n_combs, sample_duration, step, p_tubes, combinations,
	ret_tubes);

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
