#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

## TODO create db for all tubes containing boxes
"""
Train post-hoc SVMs using the algorithm and hyper-parameters from
traditional R-CNN.
"""

import os
from sklearn import svm
import torch
import numpy as np
import glob
from imdetect import im_detect, DummyNet, create_tubedb

class SVMTrainer(object):
    """
    Trains post-hoc detection SVMs for all classes using the algorithm
    and hyper-parameters of traditional R-CNN.
    """

    def __init__(self, net, tube_db, num_classes, files_path,  im_detect=im_detect, #svmWeightsPath, svmBiassPath, svmFeatScalePath,
                 svm_C=0.001, svm_B=10.0, svm_nrEpochs=2, svm_retrainLimit=1000, svm_evictThreshold=-1.1, svm_posWeight="balanced",
                 svm_targetNorm=20.0, svm_penality='l2', svm_loss='l1', svm_rngSeed=3):

        self.net = net
        self.tube_db = tube_db
        self.num_classes = num_classes
        self.files_path = files_path
        self.im_detect = im_detect

        self.svm_nrEpochs = svm_nrEpochs # number of training iterations
        self.svm_targetNorm = svm_targetNorm # magic value from traditional R-CNN (helps with convergence)

        self.svmWeightsPath = 'svmWeights.txt'
        self.svmBiasPath = 'svmBias.txt'
        self.svmFeatScalePath = 'svmFeatScale.txt'

        self.hard_thresh = -1.0001
        self.neg_iou_thresh = 0.3
        dim = net.params['cls_score'][0].data.shape[1]
        # dim = 64 * 16
        self.feature_scale = self._get_feature_scale()
        print('Feature dim: {}'.format(dim))
        print('Feature scale: {:.3f}'.format(self.feature_scale))
        self.trainers = [SVMClassTrainer(cls, dim, self.feature_scale, svm_C, svm_B, svm_posWeight, svm_penality, svm_loss,
                                         svm_rngSeed, svm_retrainLimit, svm_evictThreshold) for cls in range(num_classes)]

    def _get_feature_scale(self, num_images=100):

        total_norm = 0.0
        total_sum = 0.0
        count = 0.0

        # num_images = len(self.files_path)
        # num_images = 1
        total_n_images = len(self.files_path)
        num_images = min(num_images, total_n_images)
        inds = np.random.choice(range(total_n_images), size=num_images, replace=False)

        for i_, i in enumerate(inds):

            path = self.files_path[i]
            scores, boxes, feat = self.im_detect(net, path, None, boReturnClassifierScore = False) #,t_mode= "train")

            total_norm += torch.sqrt((feat ** 2).sum(dim=1)).sum()
            total_sum += 1.0 * sum(sum(feat)) / len(feat)
            count += feat.shape[0]
            print('{}/{}: avg feature norm: {:.3f}, average value: {:.3f}'.format(i_ + 1, num_images,
                                                           total_norm / count, total_sum / count))

        return self.svm_targetNorm * 1.0 / (total_norm / count)

    def initialize_net(self):

        self.net.params['cls_score'][0].data[...] = 0
        self.net.params['cls_score'][1].data[...] = 0


    def _get_pos_counts(self):

        counts = torch.zeros(self.num_classes).int()
        tubedb = self.tube_db

        for i in range(len(tubedb)):
            for j in range(1, self.num_classes):

                lbl = tubedb[i]['labels'].eq(j).nonzero()
                # print('tubedb[i][\'labels\']:',tubedb[i]['labels'])
                # print('lbl :',lbl, ' j :',j)
                # lbl = lbl.eq(j).nonzero()
                # print('lbl :',lbl, ' j :',j, lbl.nelement())
                # I = np.where(roidb[i]['gt_classes'] == j)[0]
                counts[j] += lbl.nelement()
                
        for j in range(1, self.num_classes):
            print('class {} has {} positives'.
                  format(j, counts[j]))
        return counts

    def get_pos_examples(self):

        counts = self._get_pos_counts()

        for i in range(len(counts)):
            self.trainers[i].alloc_pos(counts[i])

        tube_db = self.tube_db
        num_videos = len(tube_db)

        for i in range(num_videos):

            gt_inds = tube_db[i]['labels'].gt(0).nonzero().view(-1)

            scores, boxes, feat = self.im_detect(self.net, tube_db[i]['path'], None, self.feature_scale, gt_inds, boReturnClassifierScore = False)
            # scores, boxes, feat = self.im_detect(self.net, i, gt_boxes, self.feature_scale, gt_inds, boReturnClassifierScore = False)

            for j in range(1, self.num_classes):
                cls_inds = tube_db[i]['labels'].eq(j).nonzero().view(-1)
                if cls_inds.nelement() > 0:
                    cls_feat = feat[cls_inds, :]
                    self.trainers[j].append_pos(cls_feat)

    def update_net(self, cls_ind, w, b):

        self.net.params['cls_score'][0].data[cls_ind, :] = w
        self.net.params['cls_score'][1].data[cls_ind] = b

    def train_with_hard_negatives(self):

        tubes_db = self.tube_db
        num_videos = len(tubes_db)

        for epoch in range(0,self.svm_nrEpochs):

            # num_images = 100
            for i in range(num_videos):
                print("*** EPOCH = %d, VIDEO = %d *** " % (epoch, i))

                scores, boxes, feat = self.im_detect(self.net, tube_db[i]['path'], tube_db[i]['labels'], self.feature_scale)

                for j in range(1, self.num_classes):

                    # print('tube_db[i][\'labels\'][0] :',tube_db[i]['labels'][0])
                    # # print('j :', j)
                    # # # # # # ## for 4 and 5
                    # if j != tube_db[i]['labels'][0]:
                    #     ## 4 method
                    #     hard_inds = (scores[:,j].gt(self.hard_thresh) & tube_db[i]['labels'].ne(j) \
                    #                  & tube_db[i]['labels'].ne(0)).nonzero().view(-1)
                    #     # # ## 5 method
                    #     # continue
                    # else:
                    #     hard_inds = (scores[:,j].gt(self.hard_thresh) & tube_db[i]['labels'].eq(0)) \
                    #     .nonzero().view(-1)

                    ## TODO change overlaps because now use only background rois
                    # print('tube_db[i][\'labels\'] :',tube_db[i]['labels'])
                    # print('tube_db[i][\'labels\'] :',tube_db[i]['labels'].ne(j))
                    # print('scores.device :',scores.device)
                    # print('tube_db[i][\'labels\'].device :',tube_db[i]['labels'].device)
                    # print('exw :',(scores[:,j].gt(self.hard_thresh) & tube_db[i]['labels'].ne(j)).nonzero().view(-1))
                    # hard_inds = (scores[:,j].gt(self.hard_thresh) & tube_db[i]['labels'].ne(j) \
                    #              & tube_db[i]['labels'].ne(0)).nonzero().view(-1)

                    # ## 3n only other pos  
                    # hard_inds = (scores[:,j].gt(self.hard_thresh) & tube_db[i]['labels'].ne(j) \
                    #              & tube_db[i]['labels'].ne(0)).nonzero().view(-1)
                    

                    ## 1 general background + other pos
                    hard_inds = (scores[:,j].gt(self.hard_thresh) & tube_db[i]['labels'].ne(j)).nonzero().view(-1)

                    # ## 2 only general background
                    # hard_inds = (scores[:,j].gt(self.hard_thresh) & tube_db[i]['labels'].eq(0)).nonzero().view(-1)

                    # print('hard_inds :',hard_inds)
                    # exit(-1)
                    # hard_inds = \
                    #     np.where((scores[:, j] > self.hard_thresh) &
                    #              (roidb[i]['gt_overlaps'][:, j].toarray().ravel() <
                    #               self.neg_iou_thresh))[0]

                    if len(hard_inds) > 0:
                        
                        hard_feat = feat[hard_inds, :].contiguous()

                        new_w_b = \
                            self.trainers[j].append_neg_and_retrain(feat=hard_feat)

                        if new_w_b is not None:
                            self.update_net(j, new_w_b[0], new_w_b[1])
                            np.savetxt(self.svmWeightsPath[:-4]   + "_epoch" + str(epoch) + ".txt", self.net.params['cls_score'][0].data)
                            np.savetxt(self.svmBiasPath[:-4]      + "_epoch" + str(epoch) + ".txt", self.net.params['cls_score'][1].data)
                            np.savetxt(self.svmFeatScalePath[:-4] + "_epoch" + str(epoch) + ".txt", [self.feature_scale])

            print(('train_with_hard_negatives: '
                   '{:d}/{:d} ').format(i + 1, len(tube_db)),
                                               )

    def train(self):

        # Inintialize SVMs
        self.initialize_net()

        # Pass over roidb to count num positives for each class
        #   a. Pre-allocate arrays for positive feature vectors
        # Pass over roidb, computing features for positives only
        self.get_pos_examples()

        self.train_with_hard_negatives()

        for j in range(1, self.num_classes):
            new_w_b = self.trainers[j].append_neg_and_retrain(force=True)
            self.update_net(j, new_w_b[0], new_w_b[1])

        #save svm
        np.savetxt(self.svmWeightsPath,   self.net.params['cls_score'][0].data)
        np.savetxt(self.svmBiasPath,      self.net.params['cls_score'][1].data)
        np.savetxt(self.svmFeatScalePath, [self.feature_scale])


class SVMClassTrainer(object):
    """Manages post-hoc SVM training for a single object class."""

    def __init__(self, cls, dim, feature_scale,
                 C, B, pos_weight, svm_penality, svm_loss, svm_rngSeed, svm_retrainLimit, svm_evictThreshold):
        self.pos = torch.zeros((0, dim))
        self.neg = torch.zeros((0, dim))
        self.B = B
        self.C = C
        self.cls = cls  # class of this svm
        self.pos_weight = pos_weight  # balanced
        self.dim = dim  # 64*8*7*7
        self.feature_scale = feature_scale 
        if type(pos_weight) == str:  #e.g. pos_weight == 'auto'
            class_weight = pos_weight
        else:
            class_weight = {1: pos_weight, -1: 1}

        self.svm = svm.LinearSVC(C=C, class_weight=class_weight,
                                 intercept_scaling=B, verbose=1,
                                 penalty=svm_penality, loss=svm_loss,
                                 random_state=svm_rngSeed, dual=True)

        self.pos_cur = 0
        self.num_neg_added = 0
        self.retrain_limit = svm_retrainLimit
        self.evict_thresh = svm_evictThreshold
        self.loss_history = []

    def alloc_pos(self, count):
        self.pos_cur = 0
        self.pos = torch.zeros([count, self.dim])

    def append_pos(self, feat):
        num = feat.shape[0]

        self.pos[self.pos_cur:self.pos_cur + num, :] = feat
        self.pos_cur += num

    def train(self):
        print('>>> Updating {} detector <<<'.format(self.cls))
        num_pos = self.pos.shape[0]
        num_neg = self.neg.shape[0]
        print('Cache holds {} pos examples and {} neg examples'.
              format(num_pos, num_neg))
        X = np.vstack((self.pos, self.neg)) * self.feature_scale
        y = np.hstack((np.ones(num_pos),
                       -np.ones(num_neg)))
        self.svm.fit(X, y)
        w = self.svm.coef_
        b = self.svm.intercept_[0]

        scores = self.svm.decision_function(X)
        pos_scores = scores[:num_pos]
        neg_scores = scores[num_pos:]

        num_neg_wrong = sum(neg_scores > 0)
        num_pos_wrong = sum(pos_scores < 0)
        meanAcc = 0.5 * (num_pos - num_pos_wrong) / num_pos + 0.5*(num_neg - num_neg_wrong) / num_neg
        if type(self.pos_weight) == str:
            pos_loss = 0
        else:
            pos_loss = (self.C * self.pos_weight *
                        np.maximum(0, 1 - pos_scores).sum())
        neg_loss = self.C * np.maximum(0, 1 + neg_scores).sum()
        reg_loss = 0.5 * np.dot(w.ravel(), w.ravel()) + 0.5 * b ** 2
        tot_loss = pos_loss + neg_loss + reg_loss
        self.loss_history.append((meanAcc, num_pos_wrong, num_pos, num_neg_wrong, num_neg, tot_loss, pos_loss, neg_loss, reg_loss))
        for i, losses in enumerate(self.loss_history):
            print(('    {:4d}: meanAcc={:.3f} -- pos wrong: {:5}/{:5}; neg wrong: {:5}/{:5};  '
                   '     obj val: {:.3f} = {:.3f}  (posUnscaled) + {:.3f} (neg) + {:.3f} (reg)').format(i, *losses))

        # Sanity check
        ### TODO check
        # scores_ret = (
        #                  X * 1.0 / self.feature_scale).dot(w.T * self.feature_scale) + b
        # assert np.allclose(scores, scores_ret[:, 0], atol=1e-5), \
        #         "Scores from returned model don't match decision function"

        return ((w * self.feature_scale, b), pos_scores, neg_scores)

    def append_neg_and_retrain(self, feat=None, force=False):

        if feat is not None:

            num = feat.shape[0]
            self.neg = torch.cat([self.neg.type_as(feat), feat],dim=0)
            self.num_neg_added += num

        # print('self.cls :',self.cls)
        if self.num_neg_added > self.retrain_limit or force:
            
            self.num_neg_added = 0
            new_w_b, pos_scores, neg_scores = self.train()
            # scores = np.dot(self.neg, new_w_b[0].T) + new_w_b[1]
            # easy_inds = np.where(neg_scores < self.evict_thresh)[0]
            print('    Pruning easy negatives')
            print('         before pruning: #neg = ' + str(len(self.neg)))
            not_easy_inds = np.where(neg_scores >= self.evict_thresh)[0]
            if len(not_easy_inds) > 0:
                self.neg = self.neg[not_easy_inds, :]
                # self.neg = np.delete(self.neg, easy_inds)
            print('         after pruning: #neg = ' + str(len(self.neg)))
            print('    Cache holds {} pos examples and {} neg examples'.
                  format(self.pos.shape[0], self.neg.shape[0]))
            print('    {} pos support vectors'.format((pos_scores <= 1).sum()))
            print('    {} neg support vectors'.format((neg_scores >= -1).sum()))
            return new_w_b
        else:
            return None

if __name__ == '__main__':

    #  files = glob.glob('./JHMDB-features-linear/*/*')
    # files = glob.glob('./JHMDB-features-64-7ver6/*/*')
    # files = glob.glob('./JHMDB-features-256-7ver6/*/*')
    # files = glob.glob('./JHMDB-features-256-modRoialign/*/*')
    # files = glob.glob('./JHMDB-features-256-7ver7/*/*')
    # files = glob.glob('./JHMDB-features-256-7ver7-16/*/*')
    # files = glob.glob('./JHMDB-features-64-7-orig-roialign/*/*')
    # files = glob.glob('./JHMDB-features-256-7-orig-roialign/*/*')
    # files = glob.glob('./JHMDB-features-256-7ver5/*/*')
    # files = glob.glob('./JHMDB-features-256-7ver3/*/*')
    # files = glob.glob('./JHMDB-features-256-7ver2/*/*')
    files = glob.glob('./JHMDB-features-256-8-2/*/*')
    # files = glob.glob('./JHMDB-features-256-16-4/*/*')
    # files = glob.glob('./JHMDB-features-256-32-8/*/*')
    # files = glob.glob('./UCF-101-features-256-7/*/*')
    print('len(files) :',len(files))
    n_classes = 21+1


    # print('torch.get_num_threads() :',torch.get_num_threads())
    # torch.set_num_threads(4)
    # print('torch.get_num_threads() :',torch.get_num_threads())

    # net = DummyNet(256*8,n_classes)
    # net = DummyNet(256*8*7*7,n_classes)
    # net = DummyNet(256*8*7,n_classes)
    # net = DummyNet(2*256*8*7*7,n_classes)
    # net = DummyNet(2*256*7*7,n_classes)
    # net = DummyNet(2*256*7*7,n_classes)
    # net = DummyNet(2*256*4*4,n_classes)
    # net = DummyNet(2*256*8*2*2,n_classes)
    # net = DummyNet(256*7*7,n_classes)
    # net = DummyNet(2*256*8*7*7,n_classes)
    # net = DummyNet(2*256*7*7,n_classes)
    # net = DummyNet(512*4*4,n_classes)
    # net = DummyNet(20*256*7*7,n_classes)
    # net = DummyNet(2*64*7*7,n_classes)
    # net = DummyNet(64*8*7*7,n_classes)
    # net = DummyNet(128*8*7*7,n_classes)
    # net = DummyNet(256*8,n_classes)
    # net = DummyNet(256*8*4*4,n_classes)
    # net = DummyNet(2*128*8,n_classes)
    # net = DummyNet(2*256*7*7,n_classes)
    net = DummyNet(2*256*8*7*7,n_classes)
    # net = DummyNet(256*8*7*7,n_classes)

    # net = DummyNet(2*256*16*7*7,n_classes)
    # net = DummyNet(256*16*7*7,n_classes)
    # net = DummyNet(256*8*7*7,n_classes)

    # net = DummyNet(64*8*7*7,n_classes)
    # net = DummyNet(25088,n_classes)
    # net = DummyNet(15680,n_classes)
    tube_db = create_tubedb(files)
    svm = SVMTrainer(net, tube_db, n_classes, files_path=files)
    svm.train()
