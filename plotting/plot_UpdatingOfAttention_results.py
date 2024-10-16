# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 13:57:38 2017

@author: juschu

plot/movie of simulation results of LIP maps
for updating of attention (Jonikaitis)
"""


##############################
#### imports and settings ####
##############################
import sys
import os
import numpy as np
import pylab as plt

from auxFunctions_plotting import getSetup, getModelStructure, getData_results, defineColormaps

# load parameters
sys.path.append('../parameters/')
from param_updateAtt import defParams as params_setup
from param_network import defParams as params_model


################
### plotting ###
################
def plotSetup(ep, sp):
    '''
    plot spatial layout of experiment:
    fixation point, saccade target, attention position and saccade vector
    as well as current eye position and stimulus position

    params: ep -- current eye position
            sp -- current stimulus position
    '''

    # horizontal and vertical alignments
    ha = 'center'
    va = 'top'
    va2 = 'bottom'

    # FP
    plt.scatter(spatial['FP'][0], spatial['FP'][1], marker='o', s=marker_size_setup,
                facecolors='black', linewidth=3)
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
    plt.arrow(spatial['FP'][0], spatial['FP'][1], 7.0, spatial['ST'][1], head_width=0.8,
              head_length=1, fc='black', ec='black', linewidth=2)
    plt.text(0.5*(spatial['FP'][0]+spatial['ST'][0]), 0.5*(spatial['FP'][1]+spatial['ST'][1]),
             'saccade', fontsize=fs_text, horizontalalignment=ha, verticalalignment=va2)

    # (R/L)AP
    plt.scatter(spatial['AP'][0], spatial['AP'][1], marker='s', s=marker_size_setup, color='black',
                facecolors='none', linewidth=3)
    plt.scatter(spatial['RAP'][0], spatial['RAP'][1], marker='s', s=marker_size_setup, color='0.75',
                facecolors='none', linewidth=3)
    plt.scatter(spatial['LAP'][0], spatial['LAP'][1], marker='s', s=marker_size_setup, color='0.75',
                facecolors='none', linewidth=3)
    plt.text(spatial['AP'][0], spatial['AP'][1], 'AP\n', fontsize=fs_text, horizontalalignment=ha,
             verticalalignment=va2)
    plt.text(spatial['RAP'][0], spatial['RAP'][1], 'RAP\n', fontsize=fs_text,
             horizontalalignment=ha, verticalalignment=va2)
    plt.text(spatial['LAP'][0], spatial['LAP'][1], 'LAP\n', fontsize=fs_text,
             horizontalalignment=ha, verticalalignment=va2)

    # eye position
    plt.scatter(ep[0], ep[1], marker='x', s=500, color='red', linewidth=4)

    # stimulus position
    if sp.any():
        for i in range(np.shape(stimpos)[1]):
            plt.scatter(sp[i][0], sp[i][1], marker=(10, 1, 0), s=500, color='green')

    # arrange plot
    ax = plt.gca()
    ax.set_xlim(display_setup[0][0], display_setup[0][1])
    ax.set_ylim(display_setup[1][0], display_setup[1][1])
    plt.xticks([-20, -10, 0, 10, 20])
    plt.yticks([-5, 0, 5, 10])
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%d$^\circ$'))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%d$^\circ$'))
    for label in ax.get_xticklabels() + ax.get_yticklabels():
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
    # offset of diagonal
    n = spatial['FP'][dim_nr[dim]] - m*(spatial['AP'][dim_nr[dim]]-spatial['FP'][dim_nr[dim]])
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
    # attention type: cued or top-down
    if len(sys.argv) > 1:
        # argument from command line
        attType = sys.argv[1]
    else:
        #default
        attType = 'cued'
    # folder of saved results
    resultspath = '../data/updateAtt/'+attType+'/'

    # plotting parameters
    # visual space for plotting in [[horizontal_min, horizontal_max], [vertical_min, vertical_max]]
    display = [[-14, 14], [-13, 13]]            # for firing rates
    display_setup = [[-20.2, 20.2], [-5, 12]]   # for setup
    # imshow extension for plotting
    extent_rates = {'horizontal': [display[0][0]-1, display[0][1]+1,
                                   display[0][0]-1, display[0][1]+1],
                    'vertical': [display[1][0]-1, display[1][1]+1,
                                 display[1][0]-1, display[1][1]+1],
                    'spatial': [display[0][0]-1, display[0][1]+1, display[1][0]-1, display[1][1]+1]}
    # font size of labels for points in setup, axes and titles
    fs_axes = 15
    fs_title = 20
    fs_text = 20
    # marker size points in setup
    marker_size_setup = 300
    # labelpad for axes labels
    lp = 0


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

    # get firing rates of LIP, eye and stimulus position over time cropped according to display
    dict_rates, eyepos, stimpos = getData_results(resultspath, temporal['duration'],
                                                  [numNeurons_h, numNeurons_v],
                                                  [visualField_h, visualField_v], display)


    ## Plotting ##
    defineColormaps('UoA')

    # folder where plots should be saved
    dirMovie = resultspath + "movie"
    if not os.path.exists(dirMovie):
        os.makedirs(dirMovie)

    # iterate over time
    for timestep in range(temporal['duration']):

        if not timestep%50:
            sys.stdout.write('.')
            sys.stdout.flush()

        fig = plt.figure(figsize=(10, 13))
        plt.subplots_adjust(left=0.05, right=1, bottom=0.06, top=0.95, wspace=0.0)

        # middle panel: setup with attention blobs
        plt.subplot(3, 1, 2)
        plotSetup(eyepos[timestep], stimpos[timestep])
        plotActivityBlobs(dict_rates['spatial']['Xb_PC'][timestep, :, :],
                          dict_rates['spatial']['Xb_CD'][timestep, :, :], eyepos[timestep])
        plt.title('{0}ms'.format(int(timestep-temporal['sacOnset'])), fontsize=fs_title)

        # upper panels LIP PC, lower panels LIP CD
        counter = 1
        for layer in ['PC', 'CD']:
            # left panel horizontal, right panel vertical information
            for dimension in ['horizontal', 'vertical']:
                plt.subplot(3, 2, counter)
                plotRates(dict_rates, timestep, layer, dimension)

                counter += 1

            counter += 2

        # show plot or save it
        #plt.show()
        plt.savefig(dirMovie + "/" + '%0.3d' %(timestep) + ".tiff", dpi=100)
        plt.close(fig)


    print("finished")
