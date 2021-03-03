import os
import cv2
import numpy as np
import glob
from PIL import Image
from tqdm import tqdm

def get_dataset_mean_and_std(dataset_name,scale=1):

    if dataset_name == 'jhmdb':
        return [103.75581543/255*scale, 104.79421473/255*scale,  91.16894564/255*scale],[1,1,1,]
    elif dataset_name == 'ucf-24':
        return [112.07945832/255*scale, 112.87372333/255*scale, 106.90993363/255*scale],[1,1,1,]
    elif dataset_name == 'kth':

        # [9922052.52541666 9923437.66989584 9921359.56510417]
        #
        ## array([0.0350671 , 0.03510509, 0.03504856]) ## variance
        # return [150.80253097, 150.8235834,  150.79199886]
        return [0.59138247*scale, 0.59146503*scale, 0.59134117*scale], \
               [(0.0350671)**(1/2)*scale , 0.03510509**(1/2)*scale, 0.03504856**(1/2)*scale]
    else:
        return None, None


def calculate_mean_and_std(files=None, h=120, w=160):

    sums = np.zeros((3))
    sums_sq = np.zeros((3))

    for fl in tqdm(files):
        img = np.asarray(Image.open(fl))
        img = img.copy()
        img = img / 255.0

        img = img.reshape(-1, 3)
        sums += img.sum(0)
        img_sq = img ** 2
        sums_sq += img_sq.sum(0)

    mean_image = sums/(len(files)*120*160)
    square_image = sums_sq / (len(files) * 120 * 160)
    variance_image = square_image - mean_image ** 2

    return mean_image, variance_image**(1/2)

if __name__ == '__main__':

    values = np.zeros((3,))
    print (values.shape)
    # dataset_folder = '/gpu-data/sgal/JHMDB-frames'
    dataset_folder = '../../dataset_frames'
    files = glob.glob(dataset_folder+'/*/*/*.jpg')
    # print(len(files))
    print(calculate_mean_and_std(files))
    # values = mean_image.mean(0).mean(0)

    # print(mean_image)
    #
    # sums = np.zeros((3))
    #
    # for fl in tqdm(files):
    #
    #     img = np.asarray(Image.open(fl)).reshape(-1,3)
    #     img = img.copy().reshape(-1,3)
    #     img = img / 255.0
    #     diff = (img-mean_image)**2
    #     sums += diff.sum(0)
    # print('ok')
    #
    #
    #
    #



