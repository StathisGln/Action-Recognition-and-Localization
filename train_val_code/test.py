from lib.models.action_net import ACT_net

actions = ['__background__', 'Basketball','BasketballDunk','Biking','CliffDiving','CricketBowling',
           'Diving','Fencing','FloorGymnastics','GolfSwing','HorseRiding','IceDancing',
           'LongJump','PoleVault','RopeClimbing','SalsaSpin','SkateBoarding','Skiing',
           'Skijet','SoccerJuggling','Surfing','TennisSwing','TrampolineJumping',
           'VolleyballSpiking','WalkingWithDog']


cls2idx = {actions[i]: i for i in range(0, len(actions))}
act_model = ACT_net(actions, 16)
