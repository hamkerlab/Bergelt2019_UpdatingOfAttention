# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 13:57:38 2017

@author: juschu

plot/movie of spatial and temporal setup in one panel
for updating of attention (Jonikaitis)
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
from param_updateAtt import defParams as params_setup
from param_network import defParams as params_model


################
### plotting ###
################
def plotSetup():
    '''
    plot spatial layout of experiment:
    fixation point, saccade target and attention position as well as saccade vector
    '''

    # horizontal and vertical alignments
    ha = 'center'
    va = 'top'
    va2 = 'bottom'

    # FP
    plt.scatter(spatial['FP'][0], spatial['FP'][1], marker='o', s=marker_size_setup, color='black',
                linewidth=2)
    plt.text(spatial['FP'][0], spatial['FP'][1], '\nFP', fontsize=fs_text, horizontalalignment=ha,
             verticalalignment=va)

    # ST
    plt.scatter(spatial['ST'][0], spatial['ST'][1], marker='o', s=marker_size_setup,
                edgecolor='black', facecolors='white', linewidth=2)
    plt.scatter(spatial['ST'][0], spatial['ST'][1], marker='o', s=marker_size_setup/10,
                facecolors='black', linewidth=2)
    plt.text(spatial['ST'][0], spatial['ST'][1], '\nST', fontsize=fs_text, horizontalalignment=ha,
             verticalalignment=va)
    # saccade
    plt.arrow(spatial['FP'][0], spatial['FP'][1], 7.0, spatial['ST'][1], head_width=0.7,
              head_length=1, fc='black', ec='black', linewidth=2)
    plt.text(0.5*(spatial['FP'][0]+spatial['ST'][0]), 0.5*(spatial['FP'][1]+spatial['ST'][1]),
             'saccade', fontsize=fs_text, horizontalalignment=ha, verticalalignment=va2)

    # (R/L)AP
    plt.scatter(spatial['AP'][0], spatial['AP'][1], marker='s', s=marker_size_setup, color='black',
                facecolors='none', linewidth=3)
    plt.scatter(spatial['RAP'][0], spatial['RAP'][1], marker='s', s=marker_size_setup,
                color='0.75', facecolors='none', linewidth=3)
    plt.scatter(spatial['LAP'][0], spatial['LAP'][1], marker='s', s=marker_size_setup,
                color='0.75', facecolors='none', linewidth=3)
    plt.text(spatial['AP'][0], spatial['AP'][1], 'AP\n', fontsize=fs_text, horizontalalignment=ha,
             verticalalignment=va2)
    plt.text(spatial['RAP'][0], spatial['RAP'][1], 'RAP\n', fontsize=fs_text,
             horizontalalignment=ha, verticalalignment=va2)
    plt.text(spatial['LAP'][0], spatial['LAP'][1], 'LAP\n', fontsize=fs_text,
             horizontalalignment=ha, verticalalignment=va2)

    # arrange plot
    ax = plt.gca()
    ax.set_xlim(display[0][0], display[0][1])
    ax.set_ylim(display[1][0], display[1][1])
    ax.set_xticks([-8, 0, 8])
    ax.set_yticks([0, 6])
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%d$^\circ$'))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%d$^\circ$'))
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fs_axes)

def plotRates(r, ep, sp):
    '''
    plot input rates (with respect to current eye position) as well as eye and stimulus position
    for one timestep

    params: r  -- rates of all active inputs
            ep -- eye position
            sp -- stimulus position
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
    if sp.any():
        for i in range(np.shape(sp)[0]):
            sp_scatter = plt.scatter(sp[i][0], sp[i][1], marker=(10, 1, 0), s=500, color='green',
                                     label='stimulus position')
    else:
        sp_scatter = None


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
    # PC signal
    if 'att' in r:
        plt.imshow(r['att'].T, cmap='OrangeAlpha', origin='lower', extent=extent_rates,
                   interpolation='gaussian')
        labels['orange'] = 'attention signal'
        plt.clim(0, 1)


    ## legend ##
    # legend for eye and simulus position
    patches = [ep_scatter]
    if sp_scatter:
        patches.append(sp_scatter)
    # create patches as legend for rates
    for i in labels:
        patches.append(mpatches.Patch(color=cdict_label[i], label=labels[i]))
    plt.legend(handles=patches, fontsize=fs_text)


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
    # folder of saved results
    resultspath = '../data/updateAtt/'+attType+'/'

    # plot only keyframe or for each timestep
    plotKeyframe = False
    # timesteps for signals and positions for keyframe
    # all timesteps must be prior to saccade onset
    timesteps = {'ret': 75, 'PC': 5, 'CD': 195, 'att': 5, 'ep': 5, 'sp': 25}

    # plotting parameters
    # visual space for plotting in [[horizontal_min, horizontal_max], [vertical_min, vertical_max]]
    display = [[-18, 18], [-5, 11]]
    # marker size for points in setup
    marker_size_setup = 300
    # font size of labels for points in setup
    fs_text = 20
    fs_axes = 20


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

    # get model structure: number of neurons, size of visual field
    numNeurons_h, numNeurons_v, visualField_h, visualField_v = getModelStructure(params_model)

    # get input rates, eye and stimulus position over time cropped according to display
    dict_rates, eyepos, stimpos = getData_setup(resultspath, temporal['duration'],
                                                [numNeurons_h, numNeurons_v],
                                                [visualField_h, visualField_v], display)
    # Which input rates are non-zero?
    activeRates = []
    for sig, rate in dict_rates.items():
        if np.max(rate) > 0:
            activeRates.append(sig)

    ## Plotting ##
    cdict_label = defineColormaps()

    if plotKeyframe:
        ## plot only keyframe
        fig = plt.figure(figsize=(20, 10))

        # setup (points, saccade)
        plotSetup()

        # input rates (signals, positions)
        input_rates = {}
        for sig in activeRates:
            input_rates[sig] = dict_rates[sig][timesteps[sig], :, :]
        plotRates(input_rates, eyepos[timesteps['ep']], stimpos[timesteps['sp']])

        # background
        #ax = plt.gca()
        #ax.set_facecolor((0.8, 0.8, 0.8))

        plt.tight_layout()

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

            fig = plt.figure(figsize=(20, 10))

            # setup (points, saccade)
            plotSetup()

            # input rates (signals, positions)
            input_rates = {}
            for sig in activeRates:
                input_rates[sig] = dict_rates[sig][timestep, :, :]
            plotRates(input_rates, eyepos[timestep], stimpos[timestep])

            # background
            #ax = plt.gca()
            #ax.set_facecolor((0.8, 0.8, 0.8))

            plt.title('%0.3d' %(timestep-temporal['sacOnset']) + "ms", fontsize=20)
            plt.tight_layout()

            # show plot or save it
            #plt.show()
            plt.savefig(dirMovie + "/" + '%0.3d' %(timestep) + ".tiff", dpi=100)
            plt.close(fig)


    print("finished")
