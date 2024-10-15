"""
@author: juschu

Updating of attention pointers created by juschu
idea: see experiment from Jonikaitis et al. (2013)

Setup:
  - 1 fixation point, 1 saccade target, 1 attention position
  - perform saccade from fixation point to saccade target
  - cued attention:
      cued attention with stimulus through Xr
  - top-down attention:
      external top-down attention through Xh
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
from param_updateAtt import defParams


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
    events are: fixation, saccade, stimulus or attention on- and offset

    return: dictionary with events and order according to event onset
    '''

    ## get positions in correct format
    FP = np.array([defParams['SacStart_h'], defParams['SacStart_v']])       # fixation point
    ST = np.array([defParams['SacTarget_h'], defParams['SacTarget_v']])     # saccade target
    AP = np.array([defParams['AttPos_h'], defParams['AttPos_v']])           # attention position

    NO_STIM = sys.float_info.max


    ## define events
    events = {}
    num_events = 0

    # 1. fixation
    event = {'name': 'eyes', 'type': 'EVENT_EYEPOS', 'time': defParams['t_begin'], 'value': FP}
    events['EVT'+str(num_events)] = event
    num_events += 1
    # 2. saccade
    event = {'name': 'eyes', 'type': 'EVENT_SACCADE', 'time': defParams['t_sacon'], 'value': ST}
    events['EVT'+str(num_events)] = event
    num_events += 1
    if attType == 'cued':
        # 3a. stimulus onset
        event = {'name': 'stim', 'type': 'EVENT_STIMULUS', 'time': defParams['t_stimon'],
                 'value': AP}
        events['EVT'+str(num_events)] = event
        num_events += 1
        # 3b. stimulus offset
        event = {'name': 'stim', 'type': 'EVENT_STIMULUS', 'time': defParams['t_stimoff'],
                 'value': np.array([NO_STIM, NO_STIM])}
        events['EVT'+str(num_events)] = event
        num_events += 1
    else:
        # 3a. attention onset
        event = {'name': 'att', 'type': 'EVENT_ATTENTION', 'time': defParams['t_atton'],
                 'value': AP}
        events['EVT'+str(num_events)] = event
        num_events += 1
        # 3b. attention offset
        event = {'name': 'att', 'type': 'EVENT_ATTENTION', 'time': defParams['t_attoff'],
                 'value': np.array([NO_STIM, NO_STIM])}
        events['EVT'+str(num_events)] = event
        num_events += 1

    # total number of events
    events['num_events'] = num_events

    #get order by sorting events according to time
    eventtimes = []
    eventnames = []
    for i in xrange(num_events):
        eventtimes.append(events['EVT' + str(i)]['time'])
        eventnames.append('EVT' + str(i))
    order = sorted(range(num_events), key=lambda x: eventtimes[x])
    eventsOrderedByTime = []
    for i in xrange(num_events):
        eventsOrderedByTime.append(eventnames[order[i]])
    events['order'] = eventsOrderedByTime


    ## return events
    return events


##############
#### main ####
##############
if __name__ == '__main__':

    ## Definitions ##
    # attention type: cued or top-down
    if len(sys.argv) > 1:
        # argument from command line
        attType = sys.argv[1]
    else:
        #default
        attType = 'cued'
    # folder for saving results
    saveDir = '../data/updateAtt/' + attType + '/'
    print "save at", saveDir

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
    print 'running simulation for', duration, 'ms'
    dout = ProgressOutput()
    time3 = time.time()

    for t in xrange(duration):
        if not t%100:
            dout.print_sim(t, duration)

        set_input(t, signals, ANN.populations()) # --> x*_baseline = x*_sig[t]
        ANN.step()

    time4 = time.time()
    print "finished simulation"


    ## Save results ##
    # get recorded firing rates
    print "save rates"
    recorded_rates = {}
    for layer in monitors:
        # firing rates of neurons over time
        recorded_rates[layer] = monitors[layer].get('r', reshape=True)
    # save everything
    save_dict_to_hdf5(recorded_rates, saveDir + 'rates/dict_rates.hdf5')

    time5 = time.time()


    ## Finish ##
    print "Create: %3d:%02d" % ((time1-time0) / 60, (time1-time0) % 60)
    print "Compile: %3d:%02d" % ((time2-time1) / 60, (time2-time1) % 60)
    print "Simulate: %3d:%02d" % ((time4-time3) / 60, (time4-time3) % 60)
    print "Save: %3d:%02d" % ((time5-time4) / 60, (time5-time4) % 60)

    print 'finished'
