from __future__ import print_function

import RL_Functions as Functions
import GameEnv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


parameters = defaultdict(lambda: [])
# x dimension in pixels
parameters['nx'] = 4#32#17#32#17
# y dimension
parameters['ny'] = 4#18#11*2
# number of previous frames to use for s
parameters['n_frames'] = 4
# number of items to collect per player
parameters['n_items'] = 5#*2
# number of players
parameters['n_players'] = 2#3
# number of allowed actions (up,down,left,right)
parameters['n_actions'] = 4

# a list of cords for the obstacle locations [[y1,x1],[y2,x2]...]
parameters['wall_loc'] = []
# we use a seperate channel for the game (3 in total) - can also map to a single frame
parameters['img_channels'] = parameters['n_frames']*3
# number of conv filters layer 1
parameters['n_conv_filt1'] = 32
# number of conv filters layer 2
parameters['n_conv_filt2'] = 64
# number of conv filters layer 3
parameters['n_conv_filt3'] = 64
# number units in dense layer
parameters['n_dense1'] = 512

parameters['opt_loss'] = 'mse'
# learning rate
parameters['learning_rate'] = .001

## MODEL

model = Functions.create_pg_model(parameters['img_channels'], parameters['ny'], parameters['nx'], n_actions=parameters['n_actions'] )

# store the network and copies for the target and best 
parameters['model'] = model
parameters['model_target'] = Functions.create_duplicate_model(model)
parameters['model_best'] = Functions.create_duplicate_model(model)

# store the replay memory and model loss
parameters['replay'] = []
parameters['loss'] = []

# term conditions
# max moves before terminating a game
parameters['max_moves'] = 300

# use DDQ
parameters['dqn'] = True

# number of frames to observe before training
parameters['observe'] = 10000

# initial epsilon - explore vs exploit
parameters['epsilon'] = 1.0

# lowest epsilon - also use this for playing
parameters['epsilon_min'] = .05

# when to stop annealing epsilon - after this many games
parameters['epsilon_stop'] = 2000

# batch size of previous frames
parameters['batch_size'] = 32

# how many past experiences to store
parameters['replay_buffer'] = 28000

# number of games to play
parameters['n_games'] = 250

# how many frames between updating target network
parameters['update_target'] = 250

# should the game terminate with a collision? will only terminate with max_moves when False
parameters['term_on_collision'] = False

# should be able to get good results after ~ 1000 games with default params and reasonable after a couple hundred
parameters = Functions.train(parameters)

