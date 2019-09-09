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

int calc_test_cuda(int K, int N, int array_size, THCudaTensor * actioness_scr,
		   THCudaTensor * overlaps_scr, THCudaTensor * tube_scores)

{

  float * actioness_scr_data     = THCudaTensor_data(state, actioness_scr);
  float * overlaps_scr_data      = THCudaTensor_data(state, overlaps_scr);
  float * tube_scores_data       = THCudaTensor_data(state, tube_scores);
  
  cudaStream_t stream = THCState_getCurrentStream(state);

  CalculationLaucher(K, N, array_size, actioness_scr_data,
		     overlaps_scr_data, tube_scores_data, stream);

  return 1;

}
