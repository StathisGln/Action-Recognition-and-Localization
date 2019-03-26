import random
import math
import numpy as np

# class LoopPadding(object):
#     def __init__(self, size):
#         self.size = size

#     def __call__(self, frame_indices):
#         out = frame_indices
#         # print('len(out) : ', len(out))
#         # print('self.size : ', self.size)
#         for index in out:
#             if len(out) >= self.size:
#                 break
#             # print(index)
#             out.append(index)
#         # print('out :', out)
#         return out

class LoopPadding(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = np.zeros((self.size,), dtype=int)
        out[:frame_indices.shape[0]] = frame_indices
        for i in range(1,self.size-frame_indices.shape[0]+1):
            out[-i] = frame_indices[-1]

        return out


class TemporalCenterCrop(object):
    """Temporally crop the given frame indices at a center.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """

        center_index = len(frame_indices) // 2
        begin_index = max(0, center_index - (self.size // 2))
        end_index = min(begin_index + self.size, len(frame_indices))

        out = frame_indices[begin_index:end_index]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out
