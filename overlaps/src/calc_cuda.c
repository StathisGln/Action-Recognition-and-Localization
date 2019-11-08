#include <THC/THC.h>
#include <math.h>
#include "calc_cuda_kernel.h"
#include <omp.h>
extern THCState *state;


int mean_overlaps(int array_size, int T, THCudaTensor * overlaps, THCudaTensor * means)
{

  float * overlaps_data  = THCudaTensor_data(state, overlaps);
  float * means_data     = THCudaTensor_data(state, means);

  cudaStream_t stream = THCState_getCurrentStream(state);

  MeansLauncher(array_size, T, overlaps_data,means_data, stream);

  return 1;

}
