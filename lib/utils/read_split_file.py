import os
import glob



if __name__ == '__main__':

    actions = ['__background__', 'Basketball','BasketballDunk','Biking','CliffDiving','CricketBowling',
               'Diving','Fencing','FloorGymnastics','GolfSwing','HorseRiding','IceDancing',
               'LongJump','PoleVault','RopeClimbing','SalsaSpin','SkateBoarding','Skiing',
               'Skijet','SoccerJuggling','Surfing','TennisSwing','TrampolineJumping',
               'VolleyballSpiking','WalkingWithDog']
    print('len(actions):',len(actions))

    split_number = 1
    mode = 'train'
    spt_path = '/gpu-data2/sgal/UCF101_Action_detection_splits/'


    file_name =  '{}list{:0>2}.txt'.format(mode,split_number)

    with open(os.path.join(spt_path, file_name)) as fp:

        lines = fp.readlines()
        files = [i.split()[0][:-4] for i in lines]

    data =  [ i.split('/') for i in files]
    file_names = [i[1] for i in data]
    classes = list(set([i[0] for i in data]))

    if 'v_VolleyballSpiking_g15_c01' in file_names:
        print('[mpfeaf')
    # print('file_names :',file_names)
    # print('classes :',classes)
