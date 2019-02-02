import os
import numpy as np
import json
import subprocess
import numpy as np
import glob



if __name__=="__main__":

    video_folder = '/gpu-data/sgal/UCF-101'
    output_folder = '/gpu-data/sgal/UCF-101-frames'

    classes = ['BasketballDunk', 'CliffDiving', 'CricketBowling', 'Fencing', 'FloorGymnastics',
               'IceDancing', 'LongJump', 'PoleVault', 'RopeClimbing', 'SalsaSpin', 'SkateBoarding',
               'Skiing', 'Skijet', 'Surfing', 'Basketball','Biking', 'Diving', 'GolfSwing', 'HorseRiding',
               'SoccerJuggling', 'TennisSwing', 'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog']

    for cls in classes:

        currDir = os.path.join(video_folder, cls)
        video_files = glob.glob(currDir+'/*.avi')
        print(len(video_files))
        subprocess.call('mkdir {}'.format(os.path.join(output_folder,cls)), shell=True)
        for input_file in video_files:

            name = input_file.split('/')[-1][:-4]
            print(name)
            
            output_path = os.path.join(output_folder, cls,name)
            print(output_path)

            subprocess.call('mkdir {}'.format(os.path.join(output_path)), shell=True)
            subprocess.call('ffmpeg -i {} {}/image_%05d.jpg'.format(input_file, output_path),
                                shell=True)




