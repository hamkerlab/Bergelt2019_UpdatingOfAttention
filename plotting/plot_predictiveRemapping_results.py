# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 13:57:38 2017

@author: juschu

plot/movie of simulation results of LIP maps
for predicitve reampping, separated in fixation and saccade task
"""


##############################
#### imports and settings ####
##############################
import sys
import os
import numpy as np
import pylab as plt
from matplotlib import gridspec

from auxFunctions_plotting import getSetup, getModelStructure, getData_results, defineColormaps

# load parameters
sys.path.append('../parameters/')
from param_predRemapping import defParams as params_setup
from param_network import defParams as params_model


################
### plotting ###
################
def plotSetup(ep, sp, expType):
    '''
    plot spatial layout of experiment:
    fixation point, (future) receptive field and saccade target and saccade vector if available
    as well as current eye position and stimulus position

    params: ep      -- current eye position
            sp      -- current stimulus position
            expType -- name of experiment type (fixation or saccade)
    '''

    # horizontal and vertical alignments
    ha = 'center'
    ha2 = 'left'
    va = 'bottom'
    va2 = 'center'

    # (F)RF
    plt.scatter([spatial['RF'][0], spatial['FRF'][0]], [spatial['RF'][1], spatial['FRF'][1]],
                marker='o', s=marker_size_RF, color='black', facecolors='none', linewidth=2,
                linestyle='--')
    plt.text(spatial['RF'][0], spatial['RF'][1], 'RF\n', fontsize=fs_text, horizontalalignment=ha,
             verticalalignment=va)
    plt.text(spatial['FRF'][0], spatial['FRF'][1], 'FRF\n', fontsize=fs_text,
             horizontalalignment=ha, verticalalignment=va)

    # FP
    plt.scatter(spatial['FP'][0], spatial['FP'][1], marker='o', s=marker_size_setup, color='black',
                linewidth=2)
    plt.text(spatial['FP'][0], spatial['FP'][1], '  FP', fontsize=fs_text, horizontalalignment=ha2,
             verticalalignment=va2)

    if expType == 'saccade':
        # ST
        plt.scatter(spatial['ST'][0], spatial['ST'][1], marker='o', s=marker_size_setup,
                    edgecolor='black', facecolors='white', linewidth=2)
        plt.scatter(spatial['ST'][0], spatial['ST'][1], marker='o', s=marker_size_setup/10,
                    facecolors='black', linewidth=2)
        plt.text(spatial['ST'][0], spatial['ST'][1], '  ST', fontsize=fs_text,
                 horizontalalignment=ha2, verticalalignment=va2)
        # saccade
        plt.arrow(spatial['FP'][0], spatial['FP'][1], 9.2, 7.3, head_width=0.5, head_length=1,
                  fc='black', ec='black', linewidth=2)
        plt.text(5.3, 3.2, 'saccade', fontsize=fs_text, horizontalalignment=ha2,
                 verticalalignment=va2)

    # eye position
    plt.scatter(ep[0], ep[1], marker='x', s=500, color='red', linewidth=4)

    # stimulus position
    for i in range(np.shape(sp)[0]):
        plt.scatter(sp[i][0], sp[i][1], marker=(10, 1, 0), s=500, color='green')

    # arrange plot
    ax = plt.gca()
    ax.set_xlim(display_setup[0][0], display_setup[0][1])
    ax.set_ylim(display_setup[1][0], display_setup[1][1])
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%d$^\circ$'))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%d$^\circ$'))
    for label in ax.get_xticklabels()+ax.get_yticklabels():
        label.set_fontsize(fs_axes)

def plotActivityBlobs(pc, cd, ep):
    '''
    plot projected activities of both LIP maps in setup

    params: pc -- projected activity from LIP PC
            cd -- projected activity from LIP CD
            ep -- current eye position
    '''

    extent_rates_eye = [extent_rates['spatial'][0]+ep[0], extent_rates['spatial'][1]+ep[0],
                        extent_rates['spatial'][2]+ep[1], extent_rates['spatial'][3]+ep[1]]

    plt.imshow(pc.T, cmap='RedAlpha', interpolation='gaussian', origin='lower',
               extent=extent_rates_eye)
    plt.clim(0, 100)
    plt.imshow(cd.T, cmap='BlueAlpha', interpolation='gaussian', origin='lower',
               extent=extent_rates_eye)
    plt.clim(0, 100)

def plotRates(rates, t, l, dim):
    '''
    plot firing rate of LIP map projected to horizontal and vertical dimension, respectively

    params: rates -- firing rates
            t     -- current time step
            l     -- name of LIP map
            dim   -- dimension (horizontal or vertical)
    '''

    dim_nr = {'horizontal': 0, 'vertical': 1}

    # plot diagonal
    m = -1 # slope of diagonal
    n = SP[dim_nr[dim]] - m*spatial['FP'][dim_nr[dim]] # offset of diagonal
    x = np.linspace(extent_rates[dim][0], extent_rates[dim][1], 100)
    y = m*x + n
    plt.plot(x, y, color='yellow', linewidth=2, linestyle='--')

    # activity
    cr = rates[dim]['Xb_'+l][t, :, :]
    plt.imshow(cr, cmap='hot', aspect='equal', extent=extent_rates[dim], origin='lower',
               interpolation='gaussian')
    plt.xticks([-10, 0, 10])
    plt.yticks([-10, 0, 10])
    plt.clim(0, 100)
    plt.title('LIP ' + l + ' ' + dim, fontsize=fs_title)
    if l == 'CD':
        plt.xlabel('"eye" position ($^\circ$)', fontsize=fs_axes, labelpad=lp)
    if dim == 'horizontal':
        plt.ylabel('"visual" position ($^\circ$)', fontsize=fs_axes, labelpad=lp)
    ax = plt.gca()
    for label in ax.get_xticklabels()+ax.get_yticklabels():
        label.set_fontsize(fs_axes)


##############
#### main ####
##############
if __name__ == '__main__':

    ## Definitions ##
    # experiment types
    expTypes = ['fixation', 'saccade']
    # folder of saved results
    resultspath = '../data/predRemapping/'

    # plotting parameters
    # visual space for plotting in [[horizontal_min, horizontal_max], [vertical_min, vertical_max]]
    display = [[-14, 14], [-13, 13]]        # for firing rates
    display_setup = [[-32, 32], [-8, 20]]   # for setup
    # imshow extension for plotting
    extent_rates = {'horizontal': [display[0][0]-1, display[0][1]+1,
                                   display[0][0]-1, display[0][1]+1],
                    'vertical': [display[1][0]-1, display[1][1]+1,
                                 display[1][0]-1, display[1][1]+1],
                    'spatial': [display[0][0]-1, display[0][1]+1, display[1][0]-1, display[1][1]+1]}
    # marker size for receptive fields and points in setup
    marker_size_RF = 1500
    marker_size_setup = 300
    # font size of labels for points in setup, axes and titles
    fs_text = 20
    fs_axes = 15
    fs_title = 20
    # labelpad for axes labels
    lp = 0


    ## Initialization ##
    print("get data from %s" % resultspath)
    # get experimental setup: saccade target, fixation point, stimulus position,
    # duration of simulation, time of saccade onset
    spatial, temporal = getSetup(params_setup)

    # get model structure: number of neurons, size of visual field
    numNeurons_h, numNeurons_v, visualField_h, visualField_v = getModelStructure(params_model)

    # get firing rates of LIP, eye and stimulus position over time cropped according to display
    dict_rates = {}
    eyepos = {}
    stimpos = {}
    for exp in expTypes:
        dict_rates[exp], eyepos[exp], stimpos[exp] = getData_results(resultspath+exp+'/',
                                                                     temporal['duration'],
                                                                     [numNeurons_h, numNeurons_v],
                                                                     [visualField_h, visualField_v],
                                                                     display)


    ## Plotting ##
    defineColormaps('PR')

    # folder where plots should be saved
    dirMovie = resultspath + "movie"
    if not os.path.exists(dirMovie):
        os.makedirs(dirMovie)

    # iterate over time
    for timestep in range(temporal['duration']):

        if not timestep%50:
            sys.stdout.write('.')
            sys.stdout.flush()

        fig = plt.figure(figsize=(20, 13))
        plt.subplots_adjust(left=0.05, right=0.98, bottom=0.05, top=0.90, wspace=0.0)

        gs = gridspec.GridSpec(1, 2)
        gs.update(wspace=0.2)

        # ugly!
        titletext = '        fixation task                                           '
        titletext += '%0.3d' %(timestep-temporal['sacOnset']) + 'ms'
        titletext += '                                         saccade task'
        plt.suptitle(titletext, fontsize=20)

        # left part fixation task, right part saccade task
        counter = 0
        for exp in expTypes:
            gs_sub = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=gs[counter], wspace=0.0)

            # middle panel: setup with attention blobs
            plt.subplot(gs_sub[1, :])
            plotSetup(eyepos[exp][timestep], stimpos[exp][timestep], exp)
            plotActivityBlobs(dict_rates[exp]['spatial']['Xb_PC'][timestep, :, :],
                              dict_rates[exp]['spatial']['Xb_CD'][timestep, :, :],
                              eyepos[exp][timestep])
            plt.gca().set_aspect(1)


            if exp == 'fixation':
                # stimulus in RF
                SP = spatial['RF']
            else:
                # stimulus in FRF
                SP = spatial['FRF']

            # upper panels LIP PC, lower panels LIP CD
            counter_layer = 0
            for layer in ['PC', 'CD']:
                # left panel horizontal, right panel vertical information
                counter_dimension = 0
                for dimension in ['horizontal', 'vertical']:
                    plt.subplot(gs_sub[counter_layer, counter_dimension])
                    plotRates(dict_rates[exp], timestep, layer, dimension)

                    counter_dimension += 1

                counter_layer += 2

            counter += 1

        # show plot or save it
        #plt.show()
        plt.savefig(dirMovie + "/" + '%0.3d' %(timestep) + ".tiff", dpi=100)
        plt.close(fig)

    print("finished")
