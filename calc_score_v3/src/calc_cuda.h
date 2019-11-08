int calc_test_cuda(int K, int N, THCudaTensor * thresh, int array_size, THCudaIntTensor *pos, THCudaIntTensor *pos_indices, THCudaTensor * actioness,
		   THCudaTensor * overlaps_scr, THCudaTensor * scores, THCudaTensor * overlaps, int idx,  /* return arrays */
		   THCudaIntTensor * next_pos_indices, THCudaTensor * next_actioness,
		   THCudaTensor * next_overlaps_scr, THCudaTensor * f_scores);


/* int calc_test_cuda(int K, int N, float thresh, int array_size, THCudaTensor *pos, THCudaTensor *pos_indices, THCudaTensor * actioness, */
/* 		   THCudaTensor * overlaps_scr, THCudaTensor * scores, THCudaTensor * overlaps, int idx,  /\* return arrays *\/ */
/* 		   THCudaTensor * next_pos, THCudaTensor * next_pos_indices, THCudaTensor * next_actioness, */
/* 		   THCudaTensor * next_overlaps_scr, THCudaTensor * f_scores); */
