#ifndef _CALC_KERNEL
#define _CALC_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

  /* __global__ void Calculate_scores(const int nthreads,const int K, const int N, const float thresh, const int array_size, */
  /* 				   const int* pos, const int* pos_indices, const float *actioness, const float *overlaps_scr, */
  /* 				   const float *scores, const float *overlaps, const int indx, int *next_pos, */
  /* 				   int *next_pos_indices,float *next_actioness, float *next_overlaps_scr, float *f_scores); */
  __global__ void Calculate_scores(const int nthreads,const int K, const int N, const int array_size,
				   const float *actioness_scr, const float *overlaps_scr,
				   float *tube_scores);
  /* int CalculationLaucher(const int K, const int N, const float thresh, const int array_size, const int* pos, const int* pos_indices, */
  /* 			 const float *actioness, const float *overlaps_scr, const float *scores, const float *overlaps, const int indx, */
  /* 			 int *next_pos, int *next_pos_indices,float *next_actioness, float *next_overlaps_scr, float *f_scores, */
  /* 			 cudaStream_t stream); */
  int CalculationLaucher(const int K, const int N, const int array_size,  const float *actioness_scr,
			 const float *overlaps_scr, float *tube_scores, cudaStream_t stream);


#ifdef __cplusplus
}
#endif

#endif

