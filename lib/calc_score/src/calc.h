int calc_test(int K, int N, float thresh, int array_size, THIntTensor *pos, THIntTensor *pos_indices, THFloatTensor * actioness,
	      THFloatTensor * overlaps_scr, THFloatTensor * scores, THFloatTensor * overlaps, int idx,  /* return arrays */
	      THIntTensor * next_pos, THIntTensor * next_pos_indices, THFloatTensor * next_actioness,
	      THFloatTensor * next_overlaps_scr, THFloatTensor * f_scores);
