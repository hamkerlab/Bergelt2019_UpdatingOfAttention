# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 14:14:01 2018

@author: juschu

parameters for setup of predictive remapping experiment
"""

defParams = {}

## positions (in head-centered coordinates) ##
# fixation point
defParams['SacStart_h'] = 0
defParams['SacStart_v'] = 0
# saccade target
defParams['SacTarget_h'] = 10
defParams['SacTarget_v'] = 8
## stimulus position
defParams['StimPos_h'] = -5
defParams['StimPos_v'] = -2

## timesteps (in ms) ##
# start and end of simulation
defParams['t_begin'] = -200
defParams['t_end'] = 200
# saccde onset
defParams['t_sacon'] = 0
# stimulus on- and offset
defParams['t_stimon'] = -150
defParams['t_stimoff'] = -50
