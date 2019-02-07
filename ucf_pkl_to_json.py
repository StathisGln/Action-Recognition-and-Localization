import os
import pickle
import json
import numpy as np



if __name__ == '__main__':

    with open('./pyannot.pkl','rb') as fp:
        data = pickle.load(fp)

    actions = ['Basketball','BasketballDunk','Biking','CliffDiving','CricketBowling',
               'Diving','Fencing','FloorGymnastics','GolfSwing','HorseRiding','IceDancing',
               'LongJump','PoleVault','RopeClimbing','SalsaSpin','SkateBoarding','Skiing',
               'Skijet','SoccerJuggling','Surfing','TennisSwing','TrampolineJumping',
               'VolleyballSpiking','WalkingWithDog']

    # classes = ['basketballdunk', 'basketballshooting','cliffdiving', 'cricketbowling', 'fencing', 'floorgymnastics',
    #            'icedancing', 'longjump', 'polevault', 'ropeclimbing', 'salsaspin', 'skateboarding',
    #            'skiing', 'skijet', 'surfing', 'biking', 'diving', 'golfswing', 'horseriding',
    #            'soccerjuggling', 'tennisswing', 'trampolinejumping', 'volleyballspiking', 'walking']
    key = next(data.__iter__())

    print('Class : %s' % key.split('/')[0].lower().replace('_','').replace(' ',''))
    lbl = key.split('/')[0].lower().replace('_','').replace(' ','')

    values = data[key]
    n_frames = values['numf']
    print('n_frames %d' % n_frames)
    print(data[key].keys())
    annots = values['annotations']
    print('len(annots) :',len(annots))
    n_samples = len(annots)
    rois = np.zeros((n_samples,n_frames,5))

    for k  in range(n_samples):
        sample = annots[k]
        s_frame = sample['sf']
        e_frame = sample['ef']
        s_label = sample['label']
        boxes   = sample['boxes']
        rois[k,s_frame:e_frame,1:] = boxes
        print('s_frame %d e_frame %d' % (s_frame, e_frame))
        print('rois :', rois.shape)
        print('label %d  action %s ' % (s_label, actions[s_label]))
