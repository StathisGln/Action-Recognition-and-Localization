#ifndef _CALC_KERNEL
#define _CALC_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

  __global__ void Calculate_scores(const int nthreads,const int K, const int N, const int n_frames, const int n_combs, const int sample_duration, const int step, const float *p_tubes,
				   const int* combinations, float *ret_tubes);

  int CalculationLaucher(const int K, const int N, const int n_frames, const int n_combs, const int sample_duration, const int step,  const float *p_tubes, const int *combinations,
			 float *ret_tubes,  cudaStream_t stream);


#ifdef __cplusplus
}
#endif

#endif

