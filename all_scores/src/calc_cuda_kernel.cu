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


    __global__ void Calculate_scores(const int nthreads,const int K, const int N, const int array_size,
				    const float *actioness_scr, const float *overlaps_scr,
				    float *tube_scores) {
        CUDA_1D_KERNEL_LOOP(index, nthreads) {



	  int idx_i, idx_j, pre_idx_i, pre_idx_j;
	  int tmp_K;
	  float tmp_actioness_scr, tmp_overlaps_scr;

	  const int max_n_clips=5;
	  int pos[max_n_clips];
	  

	  tmp_actioness_scr = 0;
	  tmp_overlaps_scr  = 0;
	  
	  // if (index == 0){
	  //   printf("K %d, N %d, j %d z %d array_size %d\n", K,N,j,z, array_size);
	  // }

	  if ( index < array_size ) {

	    tmp_K = 1;

	    for(int i=N-1; i>=0; i--){
	      pos[i] = (index / tmp_K) % K;
	      tmp_K = tmp_K * K;
	    }

	    // if(index==134)
	    //   for (int i=0; i<N; i++)
	    // 	printf("i %d pos[i] %d\n",i, pos[i]);
		
	    for (int i=0; i<max_n_clips; i++){

	      if ( i >= max_n_clips )
		break;
	      
	      if ( i == 0 ){

		idx_i = 0;
		idx_j = pos[i];

		// if (index == 1){
		//   printf("mpike...\n");
		//   printf("idx_i :%d, idx_j :%d \n",idx_i, idx_j);
		//   printf("actioness_scr[idx_i*K+idx_j] %f \n", actioness_scr[idx_i*K+idx_j]);
		// }
		tmp_actioness_scr = actioness_scr[idx_i*K+idx_j];
	      }
	      else{

		pre_idx_i = idx_i;
		pre_idx_j = idx_j;

		idx_i = i;
		idx_j = pos[i];
		  
		// if (index == 0){
		//   printf("pre_idx_i :%d, pre_idx_j :%d \n",pre_idx_i, pre_idx_j);
		//   printf("idx_i :%d, idx_j :%d \n",idx_i, idx_j);
		//   printf("pred_idx_i*K*N*K :%d \n",pre_idx_i*K*K);
		//   printf("pre_idx_i*K*N*K + pre_idx_j*K :%d\n",pre_idx_i*K*K + pre_idx_j*K);
		//   printf("pre_idx_i*K*N*K + pre_idx_j*K + idx_j :%d\n",pre_idx_i*K*K + pre_idx_j*K+idx_j);
		// }

		tmp_actioness_scr += actioness_scr[idx_i*K+idx_j];
		tmp_overlaps_scr  += overlaps_scr[pre_idx_i*K*K + pre_idx_j*K + idx_j];

		// if (index == 1){
		//   printf("overlaps_scr[pre_idx_i*K*N*K + pre_idx_j*K + idx_j]; :%f \n",overlaps_scr[pre_idx_i*K*K + pre_idx_j*K + idx_j]);
		// }
	      }
	    }
	    if (N > 1){
	      // if (index == 0){
	      //   printf("tmp_actioness_scr :%f tmp_overlaps_scr %f\n", tmp_actioness_scr, tmp_overlaps_scr);
	      // }
		
	      tube_scores[index] = tmp_actioness_scr/N + tmp_overlaps_scr/(N-1);}
	    else
	      tube_scores[index] = tmp_actioness_scr;
	  }
	}
    }


    int CalculationLaucher(const int K, const int N, const int array_size, const float *actioness_scr,
			   const float *overlaps_scr, float *tube_scores, cudaStream_t stream) {
        const int kThreadsPerBlock = 1024;
        // const int kThreadsPerBlock = 64;

        const int output_size = array_size * K;

        cudaError_t err;
	// printf("output_size %d\n",output_size);
        Calculate_scores <<< (output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream >>>(
          output_size, K, N, array_size, actioness_scr, overlaps_scr, tube_scores);

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
