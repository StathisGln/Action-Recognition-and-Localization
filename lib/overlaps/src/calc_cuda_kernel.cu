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


  __global__ void Calculate_means(const int nthreads, const int array_size, const int T, 
				  const float *overlaps,  float *means) {
        CUDA_1D_KERNEL_LOOP(index, nthreads) {

	  float sum = 0;
	  float k = 0;

	  for (int i = 0; i<T; i++){
	    if (overlaps[index*T+i] == -1.0)
	      break;
	    sum += overlaps[index*T+i];
	    k += 1;
	  }
	  if ( k > 0)
	    means[index] = sum / k;
	  else
	    means[index] = 0.0;
		
	}
    }


  int MeansLauncher(const int array_size, const int T, const float *overlaps, float *means,  cudaStream_t stream){
        // const int kThreadsPerBlock = 1024;
        const int kThreadsPerBlock = 64;

        const int output_size = array_size;
        cudaError_t err;

        Calculate_means <<< (output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0,
	  stream >>>(
		     output_size, array_size, T,
		     overlaps, means);
	
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
