from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

UCF = edict()
UCF.split_txt_path = "UCF101_Action_detection_splits"
UCF.boxes_file = "pyannot.pkl"
UCF.dataset_frames_folder = "UCF-101-frames"
UCF.classes = ['__background__', 'Basketball', 'BasketballDunk', 'Biking', 'CliffDiving', 'CricketBowling',
    'Diving', 'Fencing', 'FloorGymnastics', 'GolfSwing', 'HorseRiding', 'IceDancing',
    'LongJump', 'PoleVault', 'RopeClimbing', 'SalsaSpin', 'SkateBoarding', 'Skiing',
    'Skijet', 'SoccerJuggling', 'Surfing', 'TennisSwing', 'TrampolineJumping',
    'VolleyballSpiking', 'WalkingWithDog']
cfg.UCF = UCF

KTH = edict()

KTH.split_txt_path = "00sequences.txt"
KTH.boxes_file = "dataset_actions_annots.json"
KTH.dataset_frames_folder = "dataset_frames"
KTH.classes = ['__background__','Walking']
cfg.KTH = KTH


def set_dataset(dataset):

    if dataset.upper().startswith('UCF'):
        cfg.dataset = cfg.UCF
    elif dataset.upper().startswith('KTH'):
        cfg.dataset = cfg.KTH