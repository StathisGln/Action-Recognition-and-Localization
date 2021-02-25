#include <TH/TH.h>
#include <stdio.h>

void find_scores(int K, int N, float thresh, int array_size, int *pos,int *pos_indices, float *actioness, float *overlaps_scr,
		 float *scores, float *overlaps, int indx, /* now the return arrays */
		 int *next_pos, int *next_pos_indices, float* next_actioness, float* next_overlaps_scr, float* f_scores){
/* void find_scores(int K, float thresh, int array_size, int *pos,int *pos_indices, float *actioness, float *overlaps_scr, */
/* 		 float *scores, float *overlaps, int indx, /\* now the return arrays *\/ */
/* 		 int *next_pos, int *next_pos_indices, float* next_actioness, float* next_overlaps_scr, float* f_scores){ */

  float tmp_sum;
  int idx_i, idx_j, m, tmp_pos;
  float **new_scores, **new_ovelaps;     // new_scores and new_overlaps are updated in every loop
  int next_pos_max_size;
  next_pos_max_size = array_size * K;

  // init arrays
  new_scores  = (float **) malloc (array_size * sizeof(float *));
  new_ovelaps = (float **) malloc (array_size * sizeof(float *));
  for (int j=0; j<array_size; j++){
    new_scores[j]  = (float *) malloc (K * sizeof (float));
    new_ovelaps[j] = (float *) malloc (K * sizeof (float));
  }

  // count sums for actioness and overlaps
  for (int j=0; j<array_size; j++){ // for each good tube

    idx_i = pos[j * N * 2 + pos_indices[j] * 2];
    idx_j = pos[j * N * 2 + pos_indices[j] * 2+ 1];

    for(int z=0; z<K; z++){
      new_scores[j][z] = actioness[j] + scores[z];
      new_ovelaps[j][z] = overlaps_scr[j] + overlaps[idx_i* K * K + idx_j * K + z];
    }

  }

  for (int j=0; j<array_size; j++){
    for (int z=0; z<K; z++){
      m = pos_indices[j]+1;
      tmp_sum = new_scores[j][z] / (m+1) + new_ovelaps[j][z] / m;
      tmp_pos = j * K + z;
      /* printf("tmp_pos : %d tmp_sum %f ", tmp_pos,tmp_sum); */
	
      if (tmp_sum > thresh){
	/* printf(" --> mpike\n"); */
	for (int q=0; q<pos_indices[j]+1; q++){
	  next_pos[tmp_pos * N * 2 + q * 2 + 0]=pos[j * N * 2 + q * 2 + 0];
	  next_pos[tmp_pos * N * 2 + q * 2 + 1]=pos[j * N * 2 + q * 2 + 1];
	}

	// update pos indices and add new tube
	next_pos_indices[tmp_pos]=pos_indices[j]+1;

	next_pos[tmp_pos * N * 2 + next_pos_indices[tmp_pos] * 2 + 0]=indx;
	next_pos[tmp_pos * N * 2 + next_pos_indices[tmp_pos] * 2 + 1]=z;

	next_actioness[tmp_pos] = new_scores[j][z];
	next_overlaps_scr[tmp_pos] = new_ovelaps[j][z];
	f_scores[tmp_pos] = tmp_sum;

      }
      else{
	/* printf("\n"); */
	next_pos_indices[tmp_pos] = -1;
	next_actioness[tmp_pos] = -1;
	next_overlaps_scr[tmp_pos] = -1;
	f_scores[tmp_pos] = -1;
      }
    }
  }
  // add now time for adding
  tmp_pos++;

  for (int z=0; z<K; z++){
    next_pos_indices[tmp_pos]=0;
    next_pos[tmp_pos * N * 2    ]=indx;
    next_pos[tmp_pos * N * 2 + 1]=z;
    next_actioness[tmp_pos] = scores[z];
    next_overlaps_scr[tmp_pos] = 0;
    f_scores[tmp_pos] = scores[z];

    tmp_pos++;
  }
}

int calc_test(int K, int N, float thresh, int array_size, THIntTensor *pos, THIntTensor *pos_indices, THFloatTensor * actioness,
	      THFloatTensor * overlaps_scr, THFloatTensor * scores, THFloatTensor * overlaps, int idx,  /* return arrays */
	      THIntTensor * next_pos, THIntTensor * next_pos_indices, THFloatTensor * next_actioness,
	      THFloatTensor * next_overlaps_scr, THFloatTensor * f_scores)
{
  int   * pos_data               = THIntTensor_data(pos);
  int   * pos_indices_data       = THIntTensor_data(pos_indices);

  float * actioness_data         = THFloatTensor_data(actioness);
  float * overlaps_scr_data      = THFloatTensor_data(overlaps_scr);

  float * scores_data            = THFloatTensor_data(scores);
  float * overlaps_data          = THFloatTensor_data(overlaps);

  float * next_pos_data          = THIntTensor_data(next_pos);
  float * next_pos_indices_data  = THIntTensor_data(next_pos_indices);
  float * next_actioness_data    = THFloatTensor_data(next_actioness);
  float * next_overlaps_scr_data = THFloatTensor_data(next_overlaps_scr);
  float * f_scores_data          = THFloatTensor_data(f_scores);

  /* printf("ok until here...\n"); */
  /* printf("pos_data[0]: %d\n",pos_data[0]); */
  /* printf("pos_data[1]: %d\n",pos_data[1]); */
  /* printf("pos_data[2]: %d\n",pos_data[2]); */
  /* printf("pos_data[3]: %d\n",pos_data[3]); */
  /* printf("pos_data[4]: %d\n",pos_data[4]); */
  /* printf("pos_data[5]: %d\n",pos_data[5]); */
  find_scores(K, N, thresh, array_size, pos_data, pos_indices_data, actioness_data, overlaps_scr_data,
  	      scores_data, overlaps_data, idx,
  	      next_pos_data, next_pos_indices_data, next_actioness_data, next_overlaps_scr_data, f_scores_data);

  return 1;
}
