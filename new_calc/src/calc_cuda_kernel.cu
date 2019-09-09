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
				     const int* pos, const int* pos_indices, const float *N_up, const float * N_down,
				     const float *f_actioness,  const float *temporal_scr,const float *temporal_rt,
				     const float *overlaps, const float *actioness, const float *tube_rate, const int indx, int *next_pos,
				     int *next_pos_indices, float *next_actioness, float *next_temporal_scr, float *next_temporal_rt,
				     float *next_N_up, float *next_N_down) {
        CUDA_1D_KERNEL_LOOP(index, nthreads) {

	  float tmp_sum;
	  float new_score, overlap_scr;
	  int idx_i, idx_j, m, tmp_pos;
	  int j, z;

	  int orio = 23;
	  j = index / K;  // big tube index
	  z = index % K;  // new tube index
	  tmp_pos = j * K + z; // pos in arrays

	  // if (index == 0){
	  //   printf("indx %d, K %d, j %d z %d\n", indx, K,j,z);
	  // }

	  // if (index == 1){
	  //   printf("K %d, j %d z %d\n", K,j,z);
	  // }

	  // if (index == 16){
	  //   printf("K %d, j %d z %d, tmp_pos %d\n", K,j,z, tmp_pos);
	  //   printf("tube_rate[z] %f\n",tube_rate[z]);
	  // }

	  // e.g [[0,1],[1,3]]
	  // idx_i = 1
	  // idx_j = 3
	  idx_i = pos[j * N * 2 + pos_indices[j] * 2];     // get number of clip of tube 
	  idx_j = pos[j * N * 2 + pos_indices[j] * 2 + 1]; // get number of tube in this clip

	  // if (index == 0){
	  //   printf("idx_i %d idx_j %d\n", idx_i, idx_j);
	  // }
	  // if (index == 1){
	  //   printf("idx_i %d idx_j %d\n", idx_i, idx_j);
	  // }
	  // if (index == 16){
	  //   printf("idx_i %d idx_j %d\n", idx_i, idx_j);
	  // }

	  // if (index == 0){
	  //   printf("overlaps %f \n", overlaps[idx_j * K + z]);
	  // }


	  if (indx - idx_i > 1){
	    // printf("indx %d idx_i %d idx_j %d z %d\n", indx, idx_i, idx_j, z);
	    overlap_scr = 0.0;}
	  else
	    {
	    overlap_scr = overlaps[idx_j * K + z];
	    // printf("Kalo indx %d z %d ,idx_i %d idx_j %d  overlap_scr %f\n", indx, z, idx_i, idx_j,  overlap_scr);
	    }	    
	  // printf("tmp_pos :%d ", tmp_pos);
	  // printf("overlap_scr %f\n", overlap_scr);

	  if (overlap_scr > 0.5){
	    // printf("MPIKE....\n");
	    if (temporal_scr[j] < 1 ){  // not complete
	      
	      // Update temporal info
	      if (temporal_scr[j] < tube_rate[z]){

	  	if (N_up[j] < orio) next_N_up[tmp_pos] = N_up[tmp_pos] +  1; else next_N_up[tmp_pos] = orio;
	      	if (N_down[j] > 0) next_N_down[tmp_pos] = N_down[j] - 1; else  next_N_down[tmp_pos] = 0;
	      }
	      else{
	      	if (N_down[j] < orio)  next_N_down[tmp_pos] = N_down[j] + 1; else next_N_down[tmp_pos] += orio;
	      	if (N_up[j] > 0)    next_N_up[tmp_pos] = N_up[j] - 1;     else next_N_up[tmp_pos] = 0;
	      }
	      if ( next_N_up[tmp_pos] == orio)
	  	next_temporal_scr[tmp_pos] == 1;
	      else if ( N_down[tmp_pos] == orio)
	  	next_temporal_scr[tmp_pos] == 0;

	      // Update score sum
	      next_actioness[tmp_pos] = f_actioness[j] + actioness[z];

	      // Update position 
	      next_pos_indices[tmp_pos] = pos_indices[j] + 1;
	    }
	    else{
	      // printf("pos_indices :%d\n",pos_indices[j]);
	      next_pos_indices[tmp_pos] = -1;
	    }
	  } else{
	    // printf("pos_indices :%d\n",pos_indices[j]);
	    next_pos_indices[tmp_pos] = -1;
	  }
    }
}


    int CalculationLaucher(const int K, const int N, const float thresh, const int array_size, const int* pos, const int* pos_indices,
			   const float *N_up, const float * N_down, const float *f_actioness_scr, const float *temporal_scr,
			   const float *temporal_rt, const float *overlaps, const float *actioness, float *tube_rate, const int indx,
			   int *next_pos, int *next_pos_indices,float *next_actioness, float *next_temporal_scr, float *next_temporal_rt,
			   float *next_N_up, float *next_N_down, cudaStream_t stream) {

        const int kThreadsPerBlock = 1024;
        // const int kThreadsPerBlock = 64;

        const int output_size = array_size * K;
        cudaError_t err;
	// printf("output_size %d\n",output_size);
        Calculate_scores <<< (output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream >>>(
		  output_size, K, N, thresh, array_size, pos, pos_indices, N_up, N_down, f_actioness_scr, temporal_scr, temporal_rt,
        	  overlaps, actioness, tube_rate, indx,
		  next_pos, next_pos_indices, next_actioness, next_temporal_scr, next_temporal_rt, next_N_up, next_N_down);

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
