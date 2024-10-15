# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 14:14:01 2018

@author: juschu

parameters for setup of Updating of Attention experiment
"""

defParams = {}

## positions (in head-centered coordinates) ##
# fixation point
defParams['SacStart_h'] = 0
defParams['SacStart_v'] = 0
# saccade target
defParams['SacTarget_h'] = 8
defParams['SacTarget_v'] = 0
## attention position
defParams['AttPos_h'] = 0
defParams['AttPos_v'] = 6

## timesteps (in ms) ##
# start and end of simulation
defParams['t_begin'] = -200
defParams['t_end'] = 200
# saccde onset
defParams['t_sacon'] = 0
# stimulus on- and offset
defParams['t_stimon'] = -180
defParams['t_stimoff'] = -170
# top-down attention on- and offset
defParams['t_atton'] = -200
defParams['t_attoff'] = 201
