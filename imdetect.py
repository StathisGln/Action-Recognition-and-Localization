import os
import numpy as np
import torch
import torch.nn.functional as F
import glob
from easydict import EasyDict

# def im_detect(net, folder_path, im, boxes, feature_scale=None, bboxIndices=None, boReturnClassifierScore=True, classifier = 'svm'): # trainers=None,

# def temporal_pool(features, time=2):

#     print('    features.shape :',    features.shape)
#     indexes = torch.linspace(0, features.size(0), time+1).int()
#     ret_feats = torch.zeros(time,features.size(1),  features.size(2),\
#                             features.size(3))

#     if features.size(0) < time:

#         ret_feats[:features.size(0)] = features
#         return ret_feats
    
#     for i in range(time):

#         t = features[indexes[i]:indexes[i+1]].permute(1,0,2,3)
#         t = t.view(t.size(0),t.size(1),-1)
#         t = F.max_pool2d(t,kernel_size=(indexes[i+1]-indexes[i],1)).squeeze()

#         ret_feats[i] = t.view(t.size(0),features.size(2),features.size(3))
#     # print('ret_feats.shape :',ret_feats.shape)
#     return ret_feats

def temporal_pool(features, time=2):

    indexes = torch.linspace(0, features.size(0), time+1).int()
    ret_feats = torch.zeros(time,features.size(1),  features.size(2),\
                            features.size(3), features.size(4))

    if features.size(0) < time:

        ret_feats[:features.size(0)] = features
        return ret_feats
    
    for i in range(time):

        t = features[indexes[i]:indexes[i+1]].permute(1,0,2,3,4)
        t = t.view(t.size(0),t.size(1),t.size(2),-1)
        t = F.max_pool3d(t,kernel_size=(indexes[i+1]-indexes[i],1,1)).squeeze()
        ret_feats[i] = t.view(t.size(0),t.size(1), features.size(3),features.size(4))

    return ret_feats


def im_detect(net, folder_path,  boxes=None, feature_scale=None, bboxIndices=None, boReturnClassifierScore=True, classifier = 'svm'): # trainers=None,
    # Return:
    #     scores (ndarray): R x K array of object class scores (K includes
    #         background as object category 0)
    #     (optional) boxes (ndarray): R x (4*K) array of predicted bounding boxes
    # load cntk output for the given image
    featsOutputPath = os.path.join(folder_path, 'feats.pt')
    feats = torch.load(featsOutputPath).cpu()

    ###################################################
    ## edit feats max/mean
    f_feats = []

    # for i in range(feats.size(0)):
    # # for i in range(32):

    # for i in range(24):
    # lim = 24
    # lim = 12
    # lim = 6
    lim = 3
    # print('im_detect lim :',lim)
    for i in range(lim):
        # f_feats.append(temporal_pool(feats[i],time=20))
        # f_feats.append(temporal_pool(feats[i]).max(2)[0])
        # f_feats.append(temporal_pool(feats[0,i]).mean(3).mean(2))
        # f_feats.append(feats[0,i].mean(2).mean(1))
        # f_feats.append(feats[0,i])
        # print('feats.shape :',feats[i].shape)
        # print('feats.shape :',feats[i].mean(0).shape)
        # print('temporal_pool(feats[i]).shape :',temporal_pool(feats[i]).shape)
        # print('temporal_pool(feats[i]).shape :',temporal_pool(feats[i]).mean(2).shape)
        f_feats.append(temporal_pool(feats[i]))

        # f_feats.append(temporal_pool(feats[i]).mean(2))
        # f_feats.append(temporal_pool(feats[i]).max(2)[0])
        # f_feats.append(feats[i].mean(0))
        # f_feats.append(feats[i].max(0)[0])

        # f_feats.append(feats[i].mean(0).mean(-1).mean(-1))
        # f_feats.append(feats[i].max(2)[0].mean(0))


    feats = torch.stack(f_feats).cpu()
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
        scores     = torch.mm(feats * 1.0 / feature_scale, svmWeights) + svmBias

        # assert (np.unique(scores[:, 0]) == 0)  # svm always returns 0 for label 0
    return scores, None, feats.detach()

# def scoreTubes(feats, svmWeights, svmBias, svmFeatScale, roiDim=None, decisionThreshold = None):

#     n_tubes = feats.size(0)
#     ###################################################
#     ## edit feats max/mean
#     print('feats.shape :',feats.shape)
#     # feats = feats.mean(3).mean(2)
#     print('feats.shape :',feats.shape)
#     feats = feats.view(feats.size(0), -1).contiguous()
#     print('feats.shape :',feats.shape)
#     # ###################################################

#     # scores = torch.mm(svmWeights, feats.t() * 1.0 / svmFeatScale) + svmBias.view(-1,1).expand(svmWeights.size(0),feats.size(0))
    
#     # print('scores.shape :',scores.shape)
#     # print('scores.shape :',scores.t().shape)

#     scores = torch.mm( feats * 1.0 / svmFeatScale,svmWeights.t()) + svmBias

#     return scores.cuda()
#     # return scores.t()

def scoreTubes(feats, svmWeights, svmBias, svmFeatScale, roiDim=None, decisionThreshold = None):

    n_tubes = feats.size(0)
    ###################################################
    ## edit feats max/mean
    # print('feats.shape :',feats.shape)
    f_feats = []

    for i in range(feats.size(0)):
        
        # f_feats.append(feats[i].mean(2).mean(1))
        # f_feats.append(temporal_pool(feats[i]).max(2)[0])
        # f_feats.append(temporal_pool(feats[i],time=20))
        # f_feats.append(temporal_pool(feats[i]).mean(2))
        # f_feats.append(temporal_pool(feats[i]))
        f_feats.append(feats[i].mean(0))
        # f_feats.append(feats[i].max(0)[0])
        # f_feats.append(feats[i].max(2)[0].mean(0))

    feats = torch.stack(f_feats).cuda()
    feats = feats.view(feats.size(0), -1).contiguous()

    # ###################################################

    # scores = torch.mm(svmWeights, feats.t() * 1.0 / svmFeatScale) + svmBias.view(-1,1).expand(svmWeights.size(0),feats.size(0))
    
    # print('scores.shape :',scores.shape)
    # print('scores.shape :',scores.t().shape)

    scores = torch.mm( feats * 1.0 / svmFeatScale,svmWeights.t()) + svmBias

    return scores.cuda()
    # return scores.t()



class DummyNet(object):
    def __init__(self, dim, num_classes, cntkParsedOutputDir=None):
        self.name = 'dummyNet'
        self.cntkParsedOutputDir = cntkParsedOutputDir
        self.params = {
            "cls_score": [  EasyDict({'data': np.zeros((num_classes, dim), np.float32) }),
                            EasyDict({'data': np.zeros((num_classes, 1), np.float32) })],
            "trainers" : None,
        }

# def create_tubedb(files_path):

#     tube_db = []
#     bg_tubes = 0
#     bg_tubes_lim = 40
#     for i in files_path:

#         tube_dict = {}
#         labels_path = os.path.join(i,'labels.pt')
#         tube_len   = os.path.join(i,'tube_len.pt')
#         tubes       = os.path.join(i,'tubes.pt')
#         tube_dict['path']     = i
#         lbls  =  torch.load(labels_path)
#         fg_idx = lbls.gt(0).nonzero().view(-1)
#         bg_idx = lbls.eq(0).nonzero().view(-1)
#         print('fg_idx :',fg_idx)
#         print('bg_idx :',bg_idx)
#         print('lbls[fg_idx] :',lbls[fg_idx])
#         print('lbls[bg_idx] :',lbls[bg_idx])
#         exit(-1)
#         if bg_tubes < bg_tubes_lim:
#             tube_dict['labels']   = fg_idx
#         # tube_dict['tubes']    = torch.load(tubes)
#         tube_dict['tube_len'] = torch.load(tube_len)

#         tube_db.append(tube_dict)

#     print('len(tub_db) :',len(tube_db))
#     return tube_db

## BACKUP implementation
def create_tubedb(files_path):

    # limit_tub = 16
    # limit_tub = 24
    # limit_tub = 12
    # limit_tub = 6
    limit_tub = 3
    print('limit_tub :',limit_tub)
    tube_db = []
    bg_tubes = 0
    bg_tubes_lim = 40
    for i in files_path:

        tube_dict = {}
        labels_path = os.path.join(i,'labels.pt')
        tube_len   = os.path.join(i,'tube_len.pt')
        tubes       = os.path.join(i,'tubes.pt')
        tube_dict['path']     = i
        lbls = torch.load(labels_path).cpu()
        # tube_dict['labels']   = lbls[0,:24]
        tube_dict['labels']   = lbls[:limit_tub]
        # tube_dict['tubes']    = torch.load(tubes)
        tub_ln = torch.load(tube_len)
        tube_dict['tube_len'] = tub_ln[:limit_tub]
        # tube_dict['tube_len'] = tub_ln[0,:24]
        # print('llbs :',tube_dict['labels'])
        tube_db.append(tube_dict)

    print('len(tub_db) :',len(tube_db))
    return tube_db


if __name__ == '__main__':

    files = glob.glob('./JHMDB-features-linear/*/*')
    create_tubedb(files)
