#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 11:04:06 2018

@author: juschu

auxiliary functions for plotting scripts:
 - get setup and model structure
 - get eye and stimulus position over time

 - get summarized data for setup (eye and stimulus postion, input rates)
 - get summarized data for results (eye and stimulus postion, firing rates of LIP)

 - define own color maps

 - transform degree to index of neuron
"""


#################
#### imports ####
#################
import sys
import pylab as plt
import numpy as np

sys.path.append('../model/')
from auxFunctions_model import load_dict_from_hdf5


##################################
#### read out data from files ####
##################################
def getSetup(d):
    '''
    get experimental setup from xml

    params: d        -- dictionary with loaded parameters

    return: spatial  -- dictionary with spatial layout containing:
                        ST - saccade target, FP - fixation point, AP - attention position
            temporal -- dictionary with spatial layout containing:
                        duration - duration of simulation, sacOnset - time of saccade onset
    '''

    # set data from file
    # spatial layout
    # saccade target
    ST = np.array([d['SacTarget_h'], d['SacTarget_v']])
    # fixation point
    FP = np.array([d['SacStart_h'], d['SacStart_v']])
    # attention position
    if 'AttPos_h' in d:
        AP = np.array([d['AttPos_h'], d['AttPos_v']])
    else:
        AP = None
    # (future) receptive field
    if 'StimPos_h' in d:
        RF = np.array([d['StimPos_h'], d['StimPos_v']])
        FRF = RF + (ST-FP)
    else:
        RF = None
        FRF = None

    spatial = {'ST': ST, 'FP': FP, 'AP': AP, 'RF': RF, 'FRF': FRF}

    # temporal layout
    duration = d['t_end'] - d['t_begin'] + 1
    sacOnset = d['t_sacon'] - d['t_begin']
    temporal = {'duration': duration, 'sacOnset': sacOnset}

    # return
    return spatial, temporal

def getModelStructure(d):
    '''
    get model structure from xml

    params: d           -- dictionary with loaded parameters

    return: numNeurons* -- number of neurons for each dimension
            vf*         -- visual field
    '''

    # set data from file
    numNeurons_h = d['layer_size_h']
    numNeurons_v = d['layer_size_v']
    vf_h = [-d['vf_h']/2.0, d['vf_h']/2.0]
    vf_v = [-d['vf_v']/2.0, d['vf_v']/2.0]

    # return
    return numNeurons_h, numNeurons_v, vf_h, vf_v


def getEyepos(inputfile, t):
    '''
    get eye position over time from txt-file

    params: inputfile -- name of txt-file
            t         -- duration of simulation

    return: eyepos    -- numpy array of shape (t, 2) with eye positions over time
    '''

    # read file
    f = open(inputfile)
    content = f.readlines()
    f.close()

    # init
    eyepos = np.zeros((t, 2))

    # read out data from file
    numLines = len(content)
    for i in xrange(numLines):
        splittedLine = content[i].split()
        h = splittedLine[1].replace('[', '')
        v = splittedLine[2].replace(']', '')
        eyepos[i][0] = float(h)
        eyepos[i][1] = float(v)

    # return
    return eyepos

def getStimpos(inputfile, t):
    '''
    get stimulus position over time from txt-file

    params: inputfile -- name of txt-file
            t         -- duration of simulation

    return: stimpos   -- numpy array of shape (t, numEvents, 2) with stimulus positions
                         for each event over time
            events    -- list of event names
    '''

    # read file
    f = open(inputfile)
    content = f.readlines()
    f.close()

    # init
    numEvents = len(content) / (t+1)
    stimpos = np.zeros((t, numEvents, 2))
    events = []

    # read out data from file
    for k in xrange(numEvents):
        splittedLine = content[0].split()
        events.append(splittedLine[1])
        content.pop(0)
        for i in xrange(t):
            currentLine = content[0].replace('-', ' -')
            splittedLine = currentLine.split()
            content.pop(0)
            h = splittedLine[-2].replace('[', '')
            v = splittedLine[-1].replace(']', '')
            stimpos[i][k][0] = float(h)
            stimpos[i][k][1] = float(v)

    # return
    return stimpos, events


def getData_setup(path, duration, numNeurons, visualField, display):
    '''
    get eye and stimulus position as well as firing rates of inputs and prepare them for plotting
    for setup plotting

    params: path        -- folder of saved results
            duration    -- duration of simulation
            numNeurons  -- number of neurons for each dimension
            visualField -- visual field for each dimension
            display     -- part of space that should be plotted

    return: {...}       -- dictionary with input rates over time cropped according to display as
                           numpy arrays of shape (numNeurons_w_cropped, numNeurons_h_cropped, duration)
            ep          -- eye position over time as numpy array with shape (duration, 2)
            sp          -- stimulus position over time as numpy array with shape (duration, numEvents, 2)
    '''

    ## eye position ##
    ep = getEyepos(path + '_eyepos.txt', duration)


    ## stimulus position ##
    try:
        sp, _ = getStimpos(path + '_stimpos.txt', duration)
    except IOError:
        print "no stimulus"
        sp = np.full((duration, 1, 2), None)


    ## input rates ##
    # load saved rates from hdf5-file
    dict_rates_all = load_dict_from_hdf5(path + 'rates/dict_inputs.hdf5')

    # get range of neurons that corresponds to display
    startNeuron_h = deg2idx(display[0][0], numNeurons[0], visualField[0])
    startNeuron_v = deg2idx(display[1][0], numNeurons[1], visualField[1])
    endNeuron_h = deg2idx(display[0][1], numNeurons[0], visualField[0])
    endNeuron_v = deg2idx(display[1][1], numNeurons[1], visualField[1])

    ## PC
    rates = dict_rates_all['_xe_input']
    # crop according to display
    xe = rates[:, startNeuron_h:endNeuron_h+1, startNeuron_v:endNeuron_v+1]
    # normalize to 0-1
    if np.max(xe) > 0.0:
        xe = xe / float(np.max(xe))

    ## CD
    rates = dict_rates_all['_xe2_input']
    # crop according to display
    xe2 = rates[:, startNeuron_h:endNeuron_h+1, startNeuron_v:endNeuron_v+1]
    # normalize to 0-1
    if np.max(xe2) > 0.0:
        xe2 = xe2 / float(np.max(xe2))

    ## Xr
    rates = dict_rates_all['_xr_input']
    # crop according to display
    xr = rates[:, startNeuron_h:endNeuron_h+1, startNeuron_v:endNeuron_v+1]
    # normalize to 0-1
    if np.max(xr) > 0.0:
        xr = xr / float(np.max(xr))

    ## Xh
    rates = dict_rates_all['_xh_input']
    # crop according to display
    xh = rates[:, startNeuron_h:endNeuron_h+1, startNeuron_v:endNeuron_v+1]
    # normalize to 0-1
    if np.max(xh) > 0.0:
        xh = xh / float(np.max(xh))


    ## return ##
    return {'PC': xe, 'CD': xe2, 'ret': xr, 'att': xh}, ep, sp

def getData_results(path, duration, numNeurons, visualField, display):
    '''
    get eye and stimulus position as well as firing rates of LIP and prepare them for plotting
    for results plotting

    params: path        -- folder of saved results
            duration    -- duration of simulation
            numNeurons  -- number of neurons for each dimension
            visualField -- visual field for each dimension
            display     -- part of space that should be plotted

    return: dict_rates  -- dictionary with firing rates of LIP over time cropped according to display
                           and their projections to horizontal, vertical, and spatial dimensions
            ep          -- eye position over time as numpy array with shape (duration, 2)
            sp          -- stimulus position over time as numpy array with shape (duration, numEvents, 2)
    '''

    ## eye position ##
    ep = getEyepos(path + '_eyepos.txt', duration)


    ## stimulus position ##
    try:
        sp, _ = getStimpos(path + '_stimpos.txt', duration)
    except IOError:
        print "no stimulus"
        sp = np.full((duration, 1, 2), None)


    ## firing rates of LIP ##
    layers_LIP = {'Xb_PC', 'Xb_CD'}
    # load saved rates from hdf5-file
    dict_rates_all = load_dict_from_hdf5(path + 'rates/dict_rates.hdf5')

    # get range of neurons that corresponds to display
    startNeuron_h = deg2idx(display[0][0], numNeurons[0], visualField[0])
    startNeuron_v = deg2idx(display[1][0], numNeurons[1], visualField[1])
    endNeuron_h = deg2idx(display[0][1], numNeurons[0], visualField[0])
    endNeuron_v = deg2idx(display[1][1], numNeurons[1], visualField[1])

    # get firing rates
    dict_rates = {'r': {}}
    maxRate = 0
    for l in layers_LIP:
        # crop according to display
        rates = dict_rates_all[l]
        dict_rates['r'][l] = rates[:, startNeuron_h:endNeuron_h+1, startNeuron_v:endNeuron_v+1,
                                   startNeuron_h:endNeuron_h+1, startNeuron_v:endNeuron_v+1]
        maxRate = max(maxRate, np.amax(dict_rates['r'][l]))
    # normalize and project firing rates
    dict_rates.update({'horizontal': {}, 'vertical': {}, 'spatial': {}})
    for l in layers_LIP:
        # normalize rates of both LIP layers equally to 0-100
        dict_rates['r'][l] = dict_rates['r'][l] * 100/float(np.amax(dict_rates['r'][l])) #float(maxRate)
        # project to first and third dimensions (=horizontal information)
        dict_rates['horizontal'][l] = np.max(np.max(dict_rates['r'][l], axis=4), axis=2)
        # project to second and forth dimensions (=vertical information)
        dict_rates['vertical'][l] = np.max(np.max(dict_rates['r'][l], axis=3), axis=1)
        # project to first two (=spatial) dimensions
        dict_rates['spatial'][l] = np.max(np.max(dict_rates['r'][l], axis=4), axis=3)


    ## return ##
    return dict_rates, ep, sp

def getData_rates(path, duration, numNeurons, visualField, display):
    '''
    get eye and stimulus position as well as firing rates of all maps and prepare them for plotting
    for rates plotting

    params: path        -- folder of saved results
            duration    -- duration of simulation
            numNeurons  -- number of neurons for each dimension
            visualField -- visual field for each dimension
            display     -- part of space that should be plotted

    return: dict_rates  -- dictionary with firing rates of all maps over time cropped according to
                           display and their projections to horizontal and vertical dimensions
            ep          -- eye position over time as numpy array with shape (duration, 2)
            sp          -- stimulus position over time as numpy array with shape (duration, numEvents, 2)
    '''

    ## eye position ##
    ep = getEyepos(path + '_eyepos.txt', duration)


    ## stimulus position ##
    try:
        sp, _ = getStimpos(path + '_stimpos.txt', duration)
    except IOError:
        print "no stimulus"
        sp = np.full((duration, 1, 2), None)


    ## firing rates ##
    # 2d maps
    layers_2d = {'Xr', 'Xe_PC', 'Xe_CD', 'Xh'}
    # 4d maps
    layers_4d = {'Xe_FEF', 'Xb_PC', 'Xb_CD'}
    # load saved rates from hdf5-file
    dict_rates_all = load_dict_from_hdf5(path + 'rates/dict_rates.hdf5')

    # get range of neurons that corresponds to display
    startNeuron_h = deg2idx(display[0][0], numNeurons[0], visualField[0])
    startNeuron_v = deg2idx(display[1][0], numNeurons[1], visualField[1])
    endNeuron_h = deg2idx(display[0][1], numNeurons[0], visualField[0])
    endNeuron_v = deg2idx(display[1][1], numNeurons[1], visualField[1])

    # get firing rates
    dict_rates = {'r': {}}
    maxRate = 0
    for l in layers_2d:
        # crop according to display
        rates = dict_rates_all[l]
        dict_rates['r'][l] = rates[:, startNeuron_h:endNeuron_h+1, startNeuron_v:endNeuron_v+1]
        maxRate = max(maxRate, np.amax(dict_rates['r'][l]))
    for l in layers_4d:
        # crop according to display
        rates = dict_rates_all[l]
        dict_rates['r'][l] = rates[:, startNeuron_h:endNeuron_h+1, startNeuron_v:endNeuron_v+1,
                                   startNeuron_h:endNeuron_h+1, startNeuron_v:endNeuron_v+1]
        maxRate = max(maxRate, np.amax(dict_rates['r'][l]))
    # normalize and project firing rates
    dict_rates.update({'horizontal': {}, 'vertical': {}})
    for l in layers_2d:
        # normalize rates of both LIP layers equally to 0-100
        dict_rates['r'][l] = dict_rates['r'][l] * 100/float(maxRate)
    for l in layers_4d:
        # normalize rates of both LIP layers equally to 0-100
        dict_rates['r'][l] = dict_rates['r'][l] * 100/float(maxRate)
        # project to first and third dimensions (=horizontal information)
        dict_rates['horizontal'][l] = np.max(np.max(dict_rates['r'][l], axis=4), axis=2)
        # project to second and forth dimensions (=vertical information)
        dict_rates['vertical'][l] = np.max(np.max(dict_rates['r'][l], axis=3), axis=1)


    ## return ##
    return dict_rates, ep, sp


################
### plotting ###
################
def defineColormaps(expTpye=''):
    '''
    define own color map: red / blue / green / orange with transparency
    dependent on experiment type

    define colors for legend

    params: expTpye -- type of experiment ('UoA' or 'PR')
    '''

    ## new color maps ##
    # define color maps
    cdict_red = {'red':   ((0.0, 1.0, 1.0),
                           (1.0, 1.0, 1.0)),
                 'green': ((0.0, 1.0, 1.0),
                           (1.0, 0.0, 0.0)),
                 'blue':  ((0.0, 1.0, 1.0),
                           (1.0, 0.0, 0.0)),
                 'alpha': ((0.0, 0.0, 0.0),
                           (0.4, 0.4, 0.4),
                           (0.7, 0.6, 0.6),
                           (1.0, 1.0, 1.0))
                }

    if expTpye == 'PR':
        cdict_blue = {'red':   ((0.0, 1.0, 1.0),
                                (0.4, 0.0, 0.0),
                                (1.0, 0.0, 0.0)),
                      'green': ((0.0, 1.0, 1.0),
                                (0.4, 0.0, 0.0),
                                (1.0, 0.0, 0.0)),
                      'blue':  ((0.0, 1.0, 1.0),
                                (0.4, 0.8, 0.8),
                                (1.0, 1.0, 1.0)),
                      'alpha': ((0.0, 0.0, 0.0),
                                (1.0, 1.0, 1.0))
                     }
    elif expTpye == 'UoA':
        cdict_blue = {'red':   ((0.0, 1.0, 1.0),
                                (1.0, 0.0, 0.0)),
                      'green': ((0.0, 1.0, 1.0),
                                (1.0, 0.0, 0.0)),
                      'blue':  ((0.0, 1.0, 1.0),
                                (1.0, 1.0, 1.0)),
                      'alpha': ((0.0, 0.0, 0.0),
                                (0.4, 0.4, 0.4),
                                (0.7, 0.6, 0.6),
                                (1.0, 1.0, 1.0))
                     }
    else:
        print "No valid experiment type given. Use standard color map."
        cdict_blue = {'red':   ((0.0, 1.0, 1.0),
                                (1.0, 0.0, 0.0)),
                      'green': ((0.0, 1.0, 1.0),
                                (1.0, 0.0, 0.0)),
                      'blue':  ((0.0, 1.0, 1.0),
                                (1.0, 1.0, 1.0)),
                      'alpha': ((0.0, 0.0, 0.0),
                                (1.0, 1.0, 1.0))
                     }


    cdict_green = {'red':   ((0.0, 1.0, 1.0),
                             (1.0, 0.0, 0.0)),
                   'green': ((0.0, 1.0, 1.0),
                             (1.0, 1.0, 1.0)),
                   'blue':  ((0.0, 1.0, 1.0),
                             (1.0, 0.0, 0.0)),
                   'alpha': ((0.0, 0.0, 0.0),
                             (0.4, 0.4, 0.4),
                             (0.7, 0.6, 0.6),
                             (1.0, 1.0, 1.0))
                  }

    cdict_orange = {'red':   ((0.0, 1.0, 1.0),
                              (1.0, 1.0, 1.0)),
                    'green': ((0.0, 1.0, 1.0),
                              (1.0, 0.65, 0.65)),
                    'blue':  ((0.0, 1.0, 1.0),
                              (1.0, 0.0, 0.0)),
                    'alpha': ((0.0, 0.0, 0.0),
                              (0.4, 0.4, 0.4),
                              (0.7, 0.6, 0.6),
                              (1.0, 1.0, 1.0))
                   }

    # register color maps
    plt.register_cmap(name='RedAlpha', data=cdict_red)
    plt.register_cmap(name='BlueAlpha', data=cdict_blue)
    plt.register_cmap(name='GreenAlpha', data=cdict_green)
    plt.register_cmap(name='OrangeAlpha', data=cdict_orange)


    ## legend ##
    # define colors
    cdict_label = {'red': [1.0, 0.0, 0.0, 1], 'blue': [0.0, 0.0, 1.0, 1],
                   'green': [0.0, 1.0, 0.0, 1], 'orange': [1.0, 0.65, 0.0, 1],
                   'purple': [0.65, 0.0, 1.0, 1]}

    return cdict_label


#############################
#### auxiliary functions ####
#############################
def deg2idx(deg, max_idx, vf):
    '''
    transform degree in visual space to corresponding index of neuron

    params: deg       -- degree in visual space
            max_idx   -- total number of neurons
            vf        -- visual field

    return: idx       -- index of neuron
    '''

    idx = deg - vf[0]
    idx = idx / float(vf[1]-vf[0])
    idx = idx * (max_idx-1)
    # round to integer
    idx = int(round(idx))

    # return
    return idx
