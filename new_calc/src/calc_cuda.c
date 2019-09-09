#include <THC/THC.h>
#include <math.h>
#include "calc_cuda_kernel.h"
#include <omp.h>
extern THCState *state;

/* int calc_test_cuda(int K, int N, float thresh, int array_size, THCudaTensor *pos, THCudaTensor *pos_indices, THCudaTensor * actioness, */
/* 		   THCudaTensor * overlaps_scr, THCudaTensor * scores, THCudaTensor * overlaps, int idx,  /\* return arrays *\/ */
/* 		   THCudaTensor * next_pos, THCudaTensor * next_pos_indices, THCudaTensor * next_actioness, */
/* 		   THCudaTensor * next_overlaps_scr, THCudaTensor * f_scores) */

/* int calc_test_cuda(int K, int N, float thresh, int array_size, THCudaIntTensor *pos, THCudaIntTensor *pos_indices, THCudaFloatTensor * actioness, */
/* 		   THCudaFloatTensor * overlaps_scr, THCudaFloatTensor * scores, THCudaFloatTensor * overlaps, int idx,  /\* return arrays *\/ */
/* 		   THCudaIntTensor * next_pos, THCudaIntTensor * next_pos_indices, THCudaFloatTensor * next_actioness, */
/* 		   THCudaFloatTensor * next_overlaps_scr, THCudaFloatTensor * f_scores) */

int calc_test_cuda(int K, int N, float thresh, int array_size, THCudaIntTensor *pos, THCudaIntTensor *pos_indices,
		   THCudaTensor *N_up, THCudaTensor *N_down,  THCudaTensor * f_actioness_scr,
		   THCudaTensor * f_temporal_scr, THCudaTensor * f_temporal_rt,
		   THCudaTensor * overlaps, THCudaTensor *actioness,  THCudaTensor * tube_rate,   int idx,  /* return arrays */
		   THCudaIntTensor * next_pos, THCudaIntTensor * next_pos_indices, THCudaTensor * next_actioness,
		   THCudaTensor * next_temporal_scr, THCudaTensor * next_temporal_rt,
		   THCudaTensor * next_N_up, THCudaTensor * next_N_down)

{
  int   * pos_data               = THCudaIntTensor_data(state, pos);
  int   * pos_indices_data       = THCudaIntTensor_data(state, pos_indices);

  float * N_up_data              = THCudaTensor_data(state, N_up);
  float * N_down_data            = THCudaTensor_data(state, N_down);

  float * f_actioness_scr_data   = THCudaTensor_data(state, f_actioness_scr);
  float * f_temporal_scr_data    = THCudaTensor_data(state, f_temporal_scr);
  float * f_temporal_rt_data    = THCudaTensor_data(state, f_temporal_rt);

  float * overlaps_data          = THCudaTensor_data(state, overlaps);
  float * actioness_data         = THCudaTensor_data(state, actioness);
  float * tube_rate_data         = THCudaTensor_data(state, tube_rate);

  int * next_pos_data            = THCudaIntTensor_data(state, next_pos);
  int * next_pos_indices_data    = THCudaIntTensor_data(state, next_pos_indices);

  float * next_actioness_data    = THCudaTensor_data(state, next_actioness);
  float * next_temporal_scr_data = THCudaTensor_data(state, next_temporal_scr);
  float * next_temporal_rt_data = THCudaTensor_data(state, next_temporal_rt);

  float * next_N_up_data              = THCudaTensor_data(state, next_N_up);
  float * next_N_down_data            = THCudaTensor_data(state, next_N_down);

  cudaStream_t stream = THCState_getCurrentStream(state);

  
  CalculationLaucher(K, N, thresh, array_size, pos_data, pos_indices_data, N_up_data, N_down_data,
		     f_actioness_scr_data, f_temporal_scr_data, f_temporal_rt_data, overlaps_data, actioness_data,
		     tube_rate_data, idx,
		     next_pos_data, next_pos_indices_data, next_actioness_data, next_temporal_scr_data,
		     next_temporal_rt_data, next_N_up_data, next_N_down_data, stream);

  return 1;

}
