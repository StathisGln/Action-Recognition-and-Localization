#ifndef _CALC_KERNEL
#define _CALC_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

  /* __global__ void Calculate_scores(const int nthreads,const int K, const int N, const float thresh, const int array_size, */
  /* 				   const int* pos, const int* pos_indices, const float *actioness,  const float *temporal_scr, */
  /* 				   const float *tube_rate, const float *overlaps, const int indx, int *next_pos, */
  /* 				   int *next_pos_indices,float *next_actioness, float *next_overlaps_scr, float *f_scores, */
  /* 				   float *prog_rate, float *next_prog_rate); */

  __global__ void Calculate_scores(const int nthreads,const int K, const int N, const float thresh, const int array_size,
				     const int* pos, const int* pos_indices, const float *N_up, const float * N_down,
				     const float *f_actioness,  const float *temporal_scr,const float *temporal_rt,
				     const float *overlaps, const float *actioness, const float *tube_rate, const int indx, int *next_pos,
				     int *next_pos_indices, float *next_actioness, float *next_temporal_scr, float *next_temporal_rt,
				     float *next_N_up, float *next_N_down);

  int CalculationLaucher(const int K, const int N, const float thresh, const int array_size, const int* pos, const int* pos_indices,
			 const float *N_up, const float * N_down, const float *f_actioness_scr, const float *temporal_scr,
			 const float *temporal_rt, const float *overlaps, const float *actioness, float *tube_rate, const int indx,
			 int *next_pos, int *next_pos_indices,float *next_actioness, float *next_temporal_scr, float *next_temporal_rt,
			 float *next_N_up, float *next_N_down, cudaStream_t stream);


#ifdef __cplusplus
}
#endif

#endif

