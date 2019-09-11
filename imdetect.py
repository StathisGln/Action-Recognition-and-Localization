import os
import numpy as np
import torch
import glob
from easydict import EasyDict

# def im_detect(net, folder_path, im, boxes, feature_scale=None, bboxIndices=None, boReturnClassifierScore=True, classifier = 'svm'): # trainers=None,
def im_detect(net, folder_path,  boxes=None, feature_scale=None, bboxIndices=None, boReturnClassifierScore=True, classifier = 'svm'): # trainers=None,
    # Return:
    #     scores (ndarray): R x K array of object class scores (K includes
    #         background as object category 0)
    #     (optional) boxes (ndarray): R x (4*K) array of predicted bounding boxes
    # load cntk output for the given image
    featsOutputPath = os.path.join(folder_path, 'feats.pt')
    
    feats = torch.load(featsOutputPath)
    ###################################################
    ## edit feats max/mean

    f_feats = []

    for i in range(feats.size(0)):
        
        f_feats.append(feats[i].max(0)[0])
        
    feats = torch.stack(f_feats)
    feats = feats.view(feats.size(0), -1).contiguous()

    # ###################################################

    # ## edit feats

    # # f_feats = torch.zeros(feats.size(0), 5, 64, 8, 7, 7).type_as(feats)
    # f_feats = torch.zeros(feats.size(0), 5, 64,  7, 7).type_as(feats)

    # for i in range(feats.size(0)):

    #     f_feats[i,:feats.size(1)] = feats[i].mean(2).squeeze()
        
    # feats = f_feats

    # feats = feats.view(feats.size(0), -1).contiguous()

    # # ###################################################

    if bboxIndices is not None:
        feats = feats[bboxIndices, :] # only keep output for certain rois
    elif boxes is not None:
        feats = feats[:len(boxes), :] # remove zero-padded rois
    
    # compute scores for each box and each class
    scores = None
    if boReturnClassifierScore:
        svmBias    = torch.from_numpy(net.params['cls_score'][1].data.transpose()).type_as(feats)
        svmWeights = torch.from_numpy(net.params['cls_score'][0].data.transpose()).type_as(feats)
        # print('svmBias.shape :',svmBias.shape)
        # print('svmWeights.shape:',svmWeights.shape)
        # print('feats.shape :',feats.shape)
        # print('torch.mm(feats * 1.0 / 10, svmWeights).shape:',torch.mm(feats * 1.0 / 10, svmWeights).shape)
        scores     = torch.mm(feats * 1.0 / 10, svmWeights) + svmBias
        # scores     = torch.dot(feats * 1.0 / feature_scale, svmWeights) + svmBias
        assert (np.unique(scores[:, 0]) == 0)  # svm always returns 0 for label 0
    return scores, None, feats

def scoreTubes(feats, svmWeights, svmBias, svmFeatScale, roiDim=None, decisionThreshold = None):

    n_tubes = feats.size(0)

    # scores = torch.mm(svmWeights, feats.t() * 1.0 / svmFeatScale) + svmBias.view(-1,1).expand(svmWeights.size(0),feats.
    scores = torch.mm( feats * 1.0 / svmFeatScale,svmWeights.t()) + svmBias

    return scores



class DummyNet(object):
    def __init__(self, dim, num_classes, cntkParsedOutputDir=None):
        self.name = 'dummyNet'
        self.cntkParsedOutputDir = cntkParsedOutputDir
        self.params = {
            "cls_score": [  EasyDict({'data': np.zeros((num_classes, dim), np.float32) }),
                            EasyDict({'data': np.zeros((num_classes, 1), np.float32) })],
            "trainers" : None,
        }

def create_tubedb(files_path):

    tube_db = []

    for i in files_path:

        tube_dict = {}
        labels_path = os.path.join(i,'labels.pt')
        tube_len   = os.path.join(i,'tube_len.pt')
        tubes       = os.path.join(i,'tubes.pt')
        tube_dict['path']     = i
        tube_dict['labels']   = torch.load(labels_path)
        # tube_dict['tubes']    = torch.load(tubes)
        tube_dict['tube_len'] = torch.load(tube_len)

        tube_db.append(tube_dict)

    print('len(tub_db) :',len(tube_db))
    return tube_db


if __name__ == '__main__':

    files = glob.glob('./JHMDB-features-linear/*/*')
    create_tubedb(files)
