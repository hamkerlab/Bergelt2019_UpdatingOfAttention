# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 13:57:38 2017

@author: juschu

plot/movie of spatial and temporal setup in one panel
for predicitve reampping, separated in fixation and saccade task
"""


##############################
#### imports and settings ####
##############################
import sys
import os
import numpy as np
import pylab as plt
import matplotlib.patches as mpatches

from auxFunctions_plotting import getSetup, getModelStructure, getData_setup, defineColormaps

# load parameters
sys.path.append('../parameters/')
from param_predRemapping import defParams as params_setup
from param_network import defParams as params_model


################
### plotting ###
################
def plotSetup(expType):
    '''
    plot spatial layout of experiment:
    fixation point, (future) receptive field and saccade target and saccade vector if available

    params: expType -- name of experiment type (fixation or saccade)
    '''

    # horizontal and vertical alignments
    ha = 'center'
    va = 'bottom'

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
    plt.text(spatial['FP'][0], spatial['FP'][1], 'FP\n', fontsize=fs_text, horizontalalignment=ha,
             verticalalignment=va)

    if expType == 'saccade':
        # ST
        plt.scatter(spatial['ST'][0], spatial['ST'][1], marker='o', s=marker_size_setup,
                    edgecolor='black', facecolors='white', linewidth=2)
        plt.scatter(spatial['ST'][0], spatial['ST'][1], marker='o', s=marker_size_setup/10,
                    facecolors='black', linewidth=2)
        plt.text(spatial['ST'][0], spatial['ST'][1], 'ST\n', fontsize=fs_text,
                 horizontalalignment=ha, verticalalignment=va)
        # saccade
        plt.arrow(spatial['FP'][0], spatial['FP'][1], 9.2, 7.3, head_width=0.5, head_length=1,
                  fc='black', ec='black', linewidth=2)
        plt.text(5.3, 3.2, 'saccade', fontsize=fs_text, horizontalalignment='left',
                 verticalalignment='center')

    # arrange plot
    ax = plt.gca()
    ax.set_xlim(display[0][0], display[0][1])
    ax.set_ylim(display[1][0], display[1][1])
    #ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%d$^\circ$'))
    #ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%d$^\circ$'))
    plt.xlabel('horizontal position ($^\circ$)', fontsize=fs_axes)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fs_axes)
    if expType == 'saccade':
        ax.get_yaxis().set_visible(False)
    else:
        plt.ylabel('vertical position ($^\circ$)', fontsize=fs_axes)

def plotRates(r, ep, sp, leg):
    '''
    plot input rates (with respect to current eye position) as well as eye and stimulus position
    for one timestep

    params: r   -- rates of all active inputs
            ep  -- eye position
            sp  -- stimulus position
            leg -- flag, if legend should be plotted
    '''

    ## imshow extension for plotting ##
    # head-centered
    extent_rates = [max(visualField_h[0], display[0][0])-1, min(visualField_h[1], display[0][1])+1,
                    max(visualField_v[0], display[1][0])-1, min(visualField_v[1], display[1][1])+1]
    # eye-centered
    extent_rates_eye = [extent_rates[0]+ep[0], extent_rates[1]+ep[0],
                        extent_rates[2]+ep[1], extent_rates[3]+ep[1]]


    ## current eye and simulus position ##
    # eye position
    ep_scatter = plt.scatter(ep[0], ep[1], marker='x', s=500, color='red', linewidth=4,
                             label='eye position')
    # stimulus position
    for i in xrange(np.shape(sp)[0]):
        sp_scatter = plt.scatter(sp[i][0], sp[i][1], marker=(10, 1, 0), s=500, color='green',
                                 label='stimulus position')


    ## rates ##
    # labels according to colors
    labels = {}

    # eye-centered
    # CD signal
    if 'CD' in r:
        plt.imshow(r['CD'].T, cmap='BlueAlpha', origin='lower', extent=extent_rates_eye,
                   interpolation='gaussian')
        labels['blue'] = 'CD signal'
        plt.clim(0, 1)
    # retinal signal
    if 'ret' in r:
        plt.imshow(r['ret'].T, cmap='GreenAlpha', origin='lower', extent=extent_rates_eye,
                   interpolation='gaussian')
        labels['green'] = 'retinal signal'
        plt.clim(0, 1)

    # head-centered
    # PC signal
    if 'PC' in r:
        plt.imshow(r['PC'].T, cmap='RedAlpha', origin='lower', extent=extent_rates,
                   interpolation='gaussian')
        labels['red'] = 'PC signal'
        plt.clim(0, 1)


    ## legend ##
    if leg:
        # legend for eye and simulus position
        patches = [ep_scatter, sp_scatter]
        # create patches as legend for rates
        for i in labels:
            patches.append(mpatches.Patch([0, 0], color=cdict_label[i], label=labels[i]))
        plt.legend(handles=patches, loc='lower right', fontsize=fs_text)


##############
#### main ####
##############
if __name__ == '__main__':

    ## Definitions ##
    # experiment types
    expTypes = ['fixation', 'saccade']
    # folder of saved results
    resultspath = '../data/predRemapping/'

    # plot only keyframe or for each timestep
    plotKeyframe = False
    # timesteps for signals and positions for keyframe
    # all timesteps must be prior to saccade onset
    timesteps = {'ret': 100, 'PC': 10, 'CD': 190, 'ep': 0, 'sp': 100}

    # plotting parameters
    # visual space for plotting in [[horizontal_min, horizontal_max], [vertical_min, vertical_max]]
    display = [[-10, 26], [-7, 21]]
    # marker size for receptive fields and points in setup
    marker_size_RF = 1500
    marker_size_setup = 300
    # font size of labels for points in setup and axes
    fs_text = 20
    fs_axes = 20
    # plot legend for experiment
    plotLegend = {'fixation': False, 'saccade': True}


    ## Initialization ##
    print "get data from", resultspath
    # get experimental setup: saccade target, fixation point, stimulus position,
    # duration of simulation, time of saccade onset
    spatial, temporal = getSetup(params_setup)

    # get model structure: number of neurons, size of visual field
    numNeurons_h, numNeurons_v, visualField_h, visualField_v = getModelStructure(params_model)

    # get input rates, eye and stimulus position over time cropped according to display
    dict_rates = {}
    eyepos = {}
    stimpos = {}
    # set flag, if input rates are non-zero
    activeRates = {}
    for exp in expTypes:
        dict_rates[exp], eyepos[exp], stimpos[exp] = getData_setup(resultspath+exp+'/',
                                                                   temporal['duration'],
                                                                   [numNeurons_h, numNeurons_v],
                                                                   [visualField_h, visualField_v],
                                                                   display)
        # Which input rates are non-zero?
        activeRates[exp] = []
        for sig, rate in dict_rates[exp].iteritems():
            if np.max(rate) > 0:
                activeRates[exp].append(sig)

    ## Plotting ##
    cdict_label = defineColormaps()

    if plotKeyframe:
        ## plot only keyframe
        fig = plt.figure(figsize=(20, 8))
        plt.subplots_adjust(left=0.02, right=0.98, bottom=0.1, top=0.90, wspace=0.1)

        # left panel fixation task, right panel saccade task
        counter = 1
        for exp in expTypes:
            ax = plt.subplot(1, 2, counter)

            # setup (points, receptive fields)
            plotSetup(exp)

            # input rates (signals, positions)
            input_rates = {}
            for sig in activeRates[exp]:
                input_rates[sig] = dict_rates[exp][sig][timesteps[sig], :, :]
            plotRates(input_rates, eyepos[exp][timesteps['ep']], stimpos[exp][timesteps['sp']],
                      plotLegend[exp])

            # background
            #ax.set_facecolor((0.8, 0.8, 0.8))

            # title
            plt.title(exp + " task", fontsize=20)

            counter += 1

        # show plot or save it
        #plt.show()
        plt.savefig(resultspath + "movie_setup.tiff", dpi=600)
        plt.close(fig)
    else:
        ## plot for each timestep
        # folder where plots should be saved
        dirMovie = resultspath + "movie_setup"
        if not os.path.exists(dirMovie):
            os.makedirs(dirMovie)

        # iterate over time
        for timestep in range(temporal['duration']):

            if not timestep%50:
                sys.stdout.write('.')
                sys.stdout.flush()

            fig = plt.figure(figsize=(20, 8))
            plt.subplots_adjust(left=0.02, right=0.98, bottom=0.1, top=0.90, wspace=0.1)
            plt.suptitle('%0.3d' %(timestep-temporal['sacOnset']) + "ms", fontsize=20)

            # left panel fixation task, right panel saccade task
            counter = 1
            for exp in expTypes:
                ax = plt.subplot(1, 2, counter)

                # setup (points, recetive fields)
                plotSetup(exp)

                # input rates (signals, positions)
                input_rates = {}
                for sig in activeRates[exp]:
                    input_rates[sig] = dict_rates[exp][sig][timestep, :, :]
                plotRates(input_rates, eyepos[exp][timestep], stimpos[exp][timestep],
                          plotLegend[exp])

                # background
                #ax.set_facecolor((0.8, 0.8, 0.8))

                # title
                plt.title(exp + " task", fontsize=20)

                counter += 1

            # show plot or save it
            #plt.show()
            plt.savefig(dirMovie + "/" + '%0.3d' %(timestep) + ".tiff", dpi=100)
            plt.close(fig)

    print "finished"
