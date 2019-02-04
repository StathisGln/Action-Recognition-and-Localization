import os
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

gt = torch.Tensor([[[ 62.,  43.,   0.,  140., 189.,  15.,  21.],
                    [ 25.,  54.,   0.,   93., 183.,  15.,  21.],
                    [168.,  68.,   0.,  239., 208.,  15.,  21.]]]).cuda()
