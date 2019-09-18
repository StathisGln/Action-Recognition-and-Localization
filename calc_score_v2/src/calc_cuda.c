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

int calc_test_cuda(int K, int N, float thresh, int array_size, THCudaIntTensor *pos, THCudaIntTensor *pos_indices, THCudaTensor * actioness,
		   THCudaTensor * overlaps_scr, THCudaTensor * scores, THCudaTensor * overlaps, int idx,  /* return arrays */
		   THCudaIntTensor * next_pos_indices, THCudaTensor * next_actioness,
		   THCudaTensor * next_overlaps_scr, THCudaTensor * f_scores)

{
  int   * pos_data               = THCudaIntTensor_data(state, pos);
  int   * pos_indices_data       = THCudaIntTensor_data(state, pos_indices);

  float * actioness_data         = THCudaTensor_data(state, actioness);
  float * overlaps_scr_data      = THCudaTensor_data(state, overlaps_scr);

  float * scores_data            = THCudaTensor_data(state, scores);
  float * overlaps_data          = THCudaTensor_data(state, overlaps);

  int * next_pos_indices_data  = THCudaIntTensor_data(state, next_pos_indices);

  float * next_actioness_data    = THCudaTensor_data(state, next_actioness);
  float * next_overlaps_scr_data = THCudaTensor_data(state, next_overlaps_scr);
  float * f_scores_data          = THCudaTensor_data(state, f_scores);

  cudaStream_t stream = THCState_getCurrentStream(state);

  CalculationLaucher(K, N, thresh, array_size, pos_data, pos_indices_data,
		     actioness_data, overlaps_scr_data, scores_data, overlaps_data, idx,
		     next_pos_indices_data, next_actioness_data, next_overlaps_scr_data, f_scores_data,
		     stream);

  return 1;

}
