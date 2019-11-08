from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
conf = __C

# Use GPU implementation of non-maximum suppression
__C.USE_GPU_NMS = True



# Train 
__C.TRAIN = edict()


__C.TRAIN.BBOX_NORMALIZE_MEANS_3d = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
__C.TRAIN.BBOX_NORMALIZE_STDS_3d  = (0.1, 0.1, 0.1, 0.2, 0.2, 0.1)
__C.TRAIN.BBOX_INSIDE_WEIGHTS_3d  = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

__C.TRAIN.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
__C.TRAIN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)
__C.TRAIN.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

######
__C.TRAIN.BATCH_SIZE = 16
__C.TRAIN.FG_FRACTION = 0.25

# IOU >= thresh: positive example
__C.TRAIN.RPN_POSITIVE_OVERLAP = 0.8

# IOU < thresh: negative example
__C.TRAIN.RPN_NEGATIVE_OVERLAP = 0.5

# Deprecated (outside weights)
__C.TRAIN.RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
# Set to -1.0 to use uniform example weighting
__C.TRAIN.RPN_POSITIVE_WEIGHT = -1.0

# Max number of foreground examples
__C.TRAIN.RPN_FG_FRACTION = 0.5
# Total number of examples
__C.TRAIN.RPN_BATCHSIZE = 256

# If an anchor statisfied by positive and negative conditions set to negative
__C.TRAIN.RPN_CLOBBER_POSITIVES = False
# Max number of foreground examples


# Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
__C.TRAIN.FG_THRESH = 0.9

# Overlap threshold for a ROI to be considered background (class = 0 if
# overlap in [LO, HI))
__C.TRAIN.BG_THRESH_HI = 0.5
__C.TRAIN.BG_THRESH_LO = 0.1

__C.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = True


###############
#     RPN     #
###############
# Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TRAIN.RPN_PRE_NMS_TOP_N = 12000
# Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TRAIN.RPN_POST_NMS_TOP_N = 2000

__C.TRAIN.RPN_NMS_THRESH = 0.7
__C.TRAIN.RPN_MIN_SIZE = 8

## TEST MODE

__C.TEST = edict()

__C.TEST.RPN_NMS_THRESH = 0.7
__C.TEST.RPN_MIN_SIZE = 16
# __C.TEST.RPN_MIN_SIZE = 8

## Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TEST.RPN_PRE_NMS_TOP_N = 6000
# __C.TEST.RPN_PRE_NMS_TOP_N = 2000
## Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TEST.RPN_POST_NMS_TOP_N = 150
# __C.TEST.RPN_POST_NMS_TOP_N = 100
# __C.TEST.RPN_POST_NMS_TOP_N = 30
# __C.TEST.RPN_POST_NMS_TOP_N = 30

################################
#    Connection algo params    #
################################


##################
#   ALL SCORES   #
##################

__C.ALL_SCORES_THRESH=2000
# __C.ALL_SCORES_THRESH=1000
# __C.ALL_SCORES_THRESH=500

# number of tubes after connection
__C.MAX_NUMBER_TUBES=2000


#################
#   CALC ALGO   #
#################

# __C.CONNECTION_THRESH=0.5
# __C.CONNECTION_THRESH=0.6
__C.CONNECTION_THRESH=0.75

# __C.UPDATE_THRESH=20000
__C.UPDATE_THRESH=15000

# __C.FINAL_SCORES_UPDATE = 50000
__C.FINAL_SCORES_UPDATE = 50000
__C.FINAL_SCORES_KEEP = 20000

# __C.CALC_THRESH =500
__C.CALC_THRESH =2000
# __C.CALC_THRESH =4000

# __C.FINAL_SCORES_MAX_NUM=50000
__C.FINAL_SCORES_MAX_NUM=50000
# __C.MODEL_PRE_NMS_TUBES=20000
__C.MODEL_PRE_NMS_TUBES=20000

# __C.MODEL_POST_NMS_TUBES=500
# __C.MODEL_POST_NMS_TUBES=2000
__C.MODEL_POST_NMS_TUBES=4000

__C.POOLING_TIME = 20
__C.POOLING_TIME_JHMDB = 2
