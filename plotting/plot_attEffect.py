# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 09:30:55 2018

@author: juschu

plot attentional effect on different positions over time
"""


##############################
#### imports and settings ####
##############################
import sys
import math
import numpy as np
import pylab as plt

from auxFunctions_plotting import getSetup, getModelStructure, getEyepos, defineColormaps, deg2idx

sys.path.append('../model/')
from auxFunctions_model import load_dict_from_hdf5

# load parameters
sys.path.append('../parameters/')
from param_updateAtt import defParams as params_setup
from param_network import defParams as params_model


##################
#### plotting ####
##################
def plotAttEffect():
    '''
    plot attentional effect of both LIP maps and sum of it, saparated for 3 points (R/L)AP
    '''

    # define colors
    cdict_label = defineColormaps()

    plt.figure(figsize=(20, 5))
    plt.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.13)

    subPlotNr = 1
    # plot attention at 3 positions (R/L)AP
    for p in ['RAP', 'AP', 'LAP']:

        # plot attention from LIP PC, LIP CD and sum
        for l in sorted(layers_4d.keys()):

            plt.subplot(1, 3, subPlotNr)
            plt.title(p, fontsize=fs_title)

            # attention from l at p
            if subPlotNr == 1:
                if l == 'sum':
                    labeltext = 'sum of attention'
                else:
                    labeltext = 'attention from ' + l.replace('Xb_', 'LIP ')
                plt.plot(time_scaled, attention[p][l], c=cdict_label[layers_4d[l]['color']],
                         linewidth=2, linestyle=layers_4d[l]['ls'], label=labeltext)
            else:
                plt.plot(time_scaled, attention[p][l], c=cdict_label[layers_4d[l]['color']],
                         linewidth=2, linestyle=layers_4d[l]['ls'])

            # rectangle for saccade
            sacRect = plt.Rectangle((0, 0), sacDur, 100, color=col_sac,
                                    transform=plt.gca().transData)
            line, = plt.plot(time_scaled, attention[p][l], c=col_sac, alpha=1,
                             linewidth=3)
            line.set_clip_path(sacRect)
            ax = plt.gca()
            ax.add_patch(sacRect)
            plt.text(sacDur/2.0, max_value-0.02, 'saccade', color='white', fontsize=fs_text,
                     horizontalalignment='center', verticalalignment='center')

            # arrange plot
            plt.xlim(time_scaled[0], time_scaled[-1])
            plt.xticks(time_scaled[::100])
            plt.xlabel('Time relative to saccade onset (ms)', fontsize=fs_title)
            plt.ylim(0, max_value)
            plt.yticks(np.linspace(0, max_value, int(max_value*10+1)))
            if subPlotNr == 1:
                plt.ylabel('Firing rate', fontsize=fs_title)
                # legend
                plt.legend(bbox_to_anchor=(0, 1), loc=2, fontsize=fs_text)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(fs_axes)

        subPlotNr += 1

    #plt.show()
    plt.savefig(resultspath+'attEffect.tiff', dpi=600)


##############
#### main ####
##############
if __name__ == '__main__':

    ## Definitions ##
    # folder of saved results
    resultspath = '../data/updateAtt/cued/'

    # plotting parameters
    # name of layers to plot with corresponding color and line style
    layers_4d = {'Xb_PC': {'color': 'red', 'ls': '-'}, 'Xb_CD':  {'color': 'blue', 'ls': '-'}}
    col_sac = 'lightgrey'   # color for rectangle for saccade
    # font sizes
    fs_text = 13    # text for saccade
    fs_axes = 15    # axes
    fs_title = 20   # title


    ## Initialization ##
    print("get data from %s" % resultspath)
    # get experimental setup: saccade target, fixation point, attention position,
    # duration of simulation, time of saccade onset
    spatial, temporal = getSetup(params_setup)
    # remapped attention position
    spatial['RAP'] = (spatial['AP'][0]-(spatial['ST'][0]-spatial['FP'][0]),
                      spatial['AP'][1]-(spatial['ST'][1]-spatial['FP'][1]))
    # lingering attention position
    spatial['LAP'] = (spatial['AP'][0]+(spatial['ST'][0]-spatial['FP'][0]),
                      spatial['AP'][1]+(spatial['ST'][1]-spatial['FP'][1]))
    # simulation time relative to saccade onset
    time_scaled = np.linspace(0, temporal['duration']-1, temporal['duration'])-temporal['sacOnset']

    # get model structure: number of neurons, size of visual field
    numNeurons_h, numNeurons_v, visualField_h, visualField_v = getModelStructure(params_model)

    # get eye position
    eyepos = getEyepos(resultspath+'_eyepos.txt', temporal['duration'])
    # calculate saccade offset and duration
    # timestep, when eye reaches saccade target
    sacOffset = np.argmax((eyepos == spatial['ST']).all(axis=1))
    sacDur = sacOffset - temporal['sacOnset']

    # get firing rates of LIP
    dict_rates = load_dict_from_hdf5(resultspath + 'rates/dict_rates.hdf5')
    # project to first two (=spatial) dimensions
    proj_space = {}
    for l in layers_4d:
        proj_space[l] = (dict_rates[l].max(axis=4)).max(axis=3)
    # get attention at 3 (head-centered) positions (R/L)AP
    # readout acitivity of (eye-centered) projected firing rates for 3 corresponding neurons
    # (with respect to eye position!)
    # calculate sum over attention at 3 positions
    attention = {}
    for p in ['AP', 'RAP', 'LAP']:
        attention[p] = {}
        # init sum with zero
        attention[p]['sum'] = np.zeros(temporal['duration'])
        for l in layers_4d:
            att = np.zeros(temporal['duration'])
            for t in range(temporal['duration']):
                neuron = [deg2idx(spatial[p][0]-eyepos[t][0], numNeurons_h, visualField_h),
                          deg2idx(spatial[p][1]-eyepos[t][1], numNeurons_v, visualField_v)]
                att[t] = proj_space[l][t, neuron[0], neuron[1]]
            attention[p][l] = att
            attention[p]['sum'] += att
    # get maximum value of attention for plotting
    # round up to 1 digit after the decimal point
    max_value = math.ceil(max([max(attention[p]['sum']) for p in attention])*10)/10.
    # add color and line style for sum of attention
    layers_4d['sum'] = {'color': 'purple', 'ls': '--'}


    ## Plotting ##
    plotAttEffect()
