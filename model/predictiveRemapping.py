"""
@author: juschu

predictive remapping of stimuli
idea: from juschu (according to Duhamel et al., 1992)

Setup:
  - 1 fixation point, 1 saccade target, 1 stimulus
  - fixation task:
      present stimulus in RF, no saccade planned
  - saccade task:
      present stimulus in FRF shortly before saccade onset
  - get activity of Xb_PC and Xb_CD
"""


##############################
#### imports and settings ####
##############################
import time
time0 = time.time()

import sys
import numpy as np

import ANNarchy as ANN

# Including the network
import network as net
# Including the world
from world import init_inputsignals, set_input

# print simulation progress and saving
from auxFunctions_model import ProgressOutput, save_dict_to_hdf5

# load parameters
sys.path.append('../parameters/')
from param_predRemapping import defParams


###############################
#### compiling the network ####
###############################
time1 = time.time()
ANN.compile()
time2 = time.time()


#############################
#### auxiliary functions ####
#############################
def precalcEvents():
    '''
    define events (spatial and temporal) representing setup
    events are: fixation, saccade, stimulus on- and offset

    return: dictionary with events and order according to event onset
    '''

    ## get positions in correct format
    FP = np.array([defParams['SacStart_h'], defParams['SacStart_v']])       # fixation point
    ST = np.array([defParams['SacTarget_h'], defParams['SacTarget_v']])     # saccade target
    RF = np.array([defParams['StimPos_h'], defParams['StimPos_v']])         # receptive field
    FRF = RF + (ST-FP)                                                      # future receptive field

    NO_STIM = sys.float_info.max


    ## define events
    events = {}
    num_events = 0

    # 1. fixation
    event = {'name': 'eyes', 'type': 'EVENT_EYEPOS', 'time': defParams['t_begin'], 'value': FP}
    events['EVT'+str(num_events)] = event
    num_events += 1
    # 2a. stimulus onset
    SP = RF if task == 'fixation' else FRF      # stimulus in RF or FRF
    event = {'name': 'stim', 'type': 'EVENT_STIMULUS', 'time': defParams['t_stimon'], 'value': SP}
    events['EVT'+str(num_events)] = event
    num_events += 1
    # 2b. stimulus offset
    event = {'name': 'stim', 'type': 'EVENT_STIMULUS', 'time': defParams['t_stimoff'],
             'value': np.array([NO_STIM, NO_STIM])}
    events['EVT'+str(num_events)] = event
    num_events += 1
    # 3. saccade
    if task == 'saccade':
        event = {'name': 'eyes', 'type': 'EVENT_SACCADE', 'time': defParams['t_sacon'], 'value': ST}
        events['EVT'+str(num_events)] = event
        num_events += 1

    # total number of events
    events['num_events'] = num_events

    # get order by sorting events according to time
    eventtimes = []
    eventnames = []
    for i in range(num_events):
        eventtimes.append(events['EVT' + str(i)]['time'])
        eventnames.append('EVT' + str(i))
    order = sorted(range(num_events), key=lambda x: eventtimes[x])
    eventsOrderedByTime = []
    for i in range(num_events):
        eventsOrderedByTime.append(eventnames[order[i]])
    events['order'] = eventsOrderedByTime


    ## return events
    return events


##############
#### main ####
##############
if __name__ == '__main__':

    ## Definitions ##
    # experiment type: fixation or saccade
    if len(sys.argv) > 1:
        # argument from command line
        task = sys.argv[1]
    else:
        #default
        task = 'fixation'
    # folder for saving results
    saveDir = '../data/predRemapping/' + task + '/'
    print("save at %s" % saveDir)

    precalcParam = {}
    duration = defParams['t_end'] - defParams['t_begin'] + 1

    # events
    precalcParam['EVENTS'] = precalcEvents()

    # start and end of simulation
    precalcParam.update({'tbegin': defParams['t_begin'], 'tend': defParams['t_end']})


    ## Initialization ##
    # input signals
    _, signals = init_inputsignals(precalcParam, saveDir)

    # monitors for recording
    monitors = {}
    for pop in ANN.populations():
        monitors[pop.name] = ANN.Monitor(pop, 'r')


    ## Run the simulation ##
    print('running simulation for %d ms' % duration)
    dout = ProgressOutput()
    time3 = time.time()

    for t in range(duration):
        if not t%100:
            dout.print_sim(t, duration)

        set_input(t, signals, ANN.populations()) # --> x*_baseline = x*_sig[t]
        ANN.step()

    time4 = time.time()
    print("finished simulation")


    ## Save results ##
    # get recorded firing rates
    print("save rates")
    recorded_rates = {}
    for layer in monitors:
        # firing rates of neurons over time
        recorded_rates[layer] = monitors[layer].get('r', reshape=True)
    # save everything
    save_dict_to_hdf5(recorded_rates, saveDir + 'rates/dict_rates.hdf5')

    time5 = time.time()


    ## Finish ##
    print("Create: %3d:%02d" % ((time1-time0) / 60, (time1-time0) % 60))
    print("Compile: %3d:%02d" % ((time2-time1) / 60, (time2-time1) % 60))
    print("Simulate: %3d:%02d" % ((time4-time3) / 60, (time4-time3) % 60))
    print("Save: %3d:%02d" % ((time5-time4) / 60, (time5-time4) % 60))

    print('finished')
