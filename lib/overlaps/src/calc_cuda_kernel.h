#ifndef _CALC_KERNEL
#define _CALC_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

  __global__ void Calculate_means(const int nthreads, const int array_size, const int T,
				  const float *overlaps,  float *means);
  int MeansLauncher(const int array_size, const int T, const float *overlaps, float *means,  cudaStream_t stream);


#ifdef __cplusplus
}
#endif

#endif

