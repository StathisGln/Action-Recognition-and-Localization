import os
import cv2
import numpy as np
import glob
from PIL import Image


def get_dataset_mean(datasetname):

    if dataset_name == 'jhmdb':

        return [103.75581543, 104.79421473,  91.16894564]


if __name__ == '__main__':

    values = np.zeros((3,))
    print (values.shape)
    # dataset_folder = '/gpu-data/sgal/JHMDB-frames'
    dataset_folder = '/gpu-data/sgal/UCF-101-frames/'
    files = glob.glob(dataset_folder+'/*/*/*.jpg')
    print(len(files))
    for fl in files:
        img = np.asarray(Image.open(fl))
        # print(fl)
        # other_mean = img.sum(0).sum(0)/(240*320)
        mean =img.mean(0).mean(0)
        values += mean

    print(values)
    print(values/len(files))



