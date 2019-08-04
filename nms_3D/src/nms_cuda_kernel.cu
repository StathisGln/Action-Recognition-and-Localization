// ------------------------------------------------------------------
// Faster R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Shaoqing Ren
// ------------------------------------------------------------------

#include <stdbool.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include "nms_cuda_kernel.h"

#define CUDA_WARN(XXX) \
    do { if (XXX != cudaSuccess) std::cout << "CUDA Error: " << \
        cudaGetErrorString(XXX) << ", at line " << __LINE__ \
<< std::endl; cudaDeviceSynchronize(); } while (0)

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      std::cout << cudaGetErrorString(error) << std::endl; \
    } \
  } while (0)

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
int const threadsPerBlock = sizeof(unsigned long long) * 8;

__device__ inline float devIoU(float const * const a, float const * const b) {
  float left = max(a[0], b[0]), right = min(a[3], b[3]);
  float top = max(a[1], b[1]), bottom = min(a[4], b[4]);
  float front = max(a[2], b[2]), back = min(a[5], b[5]);
  float width = max(right - left + 1, 0.f), height = max(bottom - top + 1, 0.f);
  float time = max(back - front + 1, 0.f);
  float interS = width * height;
  float Sa_xy = (a[3] - a[0] + 1) * (a[4] - a[1] + 1);
  float Sa_t = (a[5] - a[2] + 1);
  float Sb_xy = (b[3] - b[0] + 1) * (b[4] - b[1] + 1);
  float Sb_t = (b[5] - b[2] + 1);
  float xy = interS / (Sa_xy + Sb_xy - interS);
  float t  = time   / (Sa_t  + Sb_t  - time) * 1.25;
  return xy * t ;
  // float interS = width * height * time;
  // float Sa = (a[3] - a[0] + 1) * (a[4] - a[1] + 1) * (a[5] - a[2] + 1);
  // float Sb = (b[3] - b[0] + 1) * (b[4] - b[1] + 1) * (b[5] - b[2] + 1);
  // return interS / (Sa + Sb - interS);

}

__global__ void nms_kernel(int n_boxes, float nms_overlap_thresh,
                           float *dev_boxes, unsigned long long *dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  // if (row_start > col_start) return;

  const int row_size =
        min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ float block_boxes[threadsPerBlock * 7];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 7 + 0] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 7 + 0];
    block_boxes[threadIdx.x * 7 + 1] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 7 + 1];
    block_boxes[threadIdx.x * 7 + 2] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 7 + 2];
    block_boxes[threadIdx.x * 7 + 3] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 7 + 3];
    block_boxes[threadIdx.x * 7 + 4] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 7 + 4];
    block_boxes[threadIdx.x * 7 + 5] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 7 + 5];
    block_boxes[threadIdx.x * 7 + 6] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 7 + 6];
	
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const float *cur_box = dev_boxes + cur_box_idx * 7;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devIoU(cur_box, block_boxes + i * 7) > nms_overlap_thresh) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

void nms_cuda_compute(int* keep_out, int *num_out, float* boxes_host, int boxes_num,
          int boxes_dim, float nms_overlap_thresh) {

  float* boxes_dev = NULL;
  unsigned long long* mask_dev = NULL;

  const int col_blocks = DIVUP(boxes_num, threadsPerBlock);

  CUDA_CHECK(cudaMalloc(&boxes_dev,
                        boxes_num * boxes_dim * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(boxes_dev,
                        boxes_host,
                        boxes_num * boxes_dim * sizeof(float),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&mask_dev,
                        boxes_num * col_blocks * sizeof(unsigned long long)));

  dim3 blocks(DIVUP(boxes_num, threadsPerBlock),
              DIVUP(boxes_num, threadsPerBlock));
  dim3 threads(threadsPerBlock);

  // printf("i am at line %d\n", boxes_num);
  // printf("i am at line %d\n", boxes_dim);  

  nms_kernel<<<blocks, threads>>>(boxes_num,
                                  nms_overlap_thresh,
                                  boxes_dev,
                                  mask_dev);

  std::vector<unsigned long long> mask_host(boxes_num * col_blocks);
  CUDA_CHECK(cudaMemcpy(&mask_host[0],
                        mask_dev,
                        sizeof(unsigned long long) * boxes_num * col_blocks,
                        cudaMemcpyDeviceToHost));

  std::vector<unsigned long long> remv(col_blocks);
  memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

  // we need to create a memory for keep_out on cpu
  // otherwise, the following code cannot run

  int* keep_out_cpu = new int[boxes_num];

  int num_to_keep = 0;
  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;

    if (!(remv[nblock] & (1ULL << inblock))) {
      // orignal: keep_out[num_to_keep++] = i;
      keep_out_cpu[num_to_keep++] = i;
      unsigned long long *p = &mask_host[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }

  // copy keep_out_cpu to keep_out on gpu
  CUDA_WARN(cudaMemcpy(keep_out, keep_out_cpu, boxes_num * sizeof(int),cudaMemcpyHostToDevice));  

  // *num_out = num_to_keep;

  // original: *num_out = num_to_keep;
  // copy num_to_keep to num_out on gpu

  CUDA_WARN(cudaMemcpy(num_out, &num_to_keep, 1 * sizeof(int),cudaMemcpyHostToDevice));  

  // release cuda memory
  CUDA_CHECK(cudaFree(boxes_dev));
  CUDA_CHECK(cudaFree(mask_dev));
  // release cpu memory
  delete []keep_out_cpu;
}