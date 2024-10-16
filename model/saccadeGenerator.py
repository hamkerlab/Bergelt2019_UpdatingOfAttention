# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 11:05:32 2017

@author: juschu

saccade generator of Van Wetter & Van Opstal (2008)
"""


#################
#### imports ####
#################
import math
import numpy


###########################
#### saccade generator ####
###########################
def vanWetterVanOpstal(duration, t, FP, ST, epMap, loadedParam, *args):
    '''
    making saccade form FP to ST starting at timestep t

    params: duration    -- duration of simulation (in ms)
            t           -- timestep the saccade starts (in ms)
            FP, ST      -- start and end point of saccade (in deg)
            epMap       -- map of eye position (in deg) over time
            loadedParam -- predefined parameters

    args:   0           -- how end of saccade is detected:
                           'endBySpeed' --> end of saccade is calculated by speed
                           'endByPos' --> end of saccade is calculated by position

    return: epMap       -- eye position map updated to saccade
            dur         -- saccade duration
    '''

    if (FP == ST).all():
        # FP and ST are equal --> no saccade
        epMap[t:] = FP
        dur = 0

    else:
        # FP and ST differ --> saccade
        # we use the extended model from WetterOpstal
        m0 = loadedParam['sac_m0']
        vpk = loadedParam['sac_vpk']    # speed in deg / ms

        how_sac_ended = args[0]

        T = numpy.linalg.norm(ST-FP)                    # saccade amplitude in deg
        direction = 1/numpy.linalg.norm(ST-FP)*(ST-FP)  # direction of saccade

        A = 1.0 / (1.0 - math.exp(-T/m0))
        E = FP      # start is fixation point
        saccade_in_progress = True
        for j in range(duration-t):
            E_previous = E
            E = (FP + direction*(m0 * math.log((A * math.exp(vpk * j / m0)) /
                                               (1.0 + A * math.exp((vpk * j - T)/m0)))))

            # detect saccade end
            sac_has_ended = False
            if how_sac_ended == 'endBySpeed':
                # saccade end is calculated by eye speed
                # one time step is one ms, so now "current_eye_speed" is in deg / s
                current_eye_speed = numpy.linalg.norm(E - E_previous)*1000
                if (current_eye_speed < loadedParam['sac_endspeed']) and (j > 0):
                    sac_has_ended = True
            elif how_sac_ended == 'endByPos':
                # saccade end is calculated by position
                if numpy.linalg.norm(E - ST) < loadedParam['sac_offset_threshold']:
                    sac_has_ended = True
            else:
                print("NO SACCADE TERMINATION DEFINED")
                if E == ST:
                    sac_has_ended = True

            if sac_has_ended:
                # saccade has ended
                if loadedParam['sac_set_end_to_target']:
                    epMap[j+t] = ST
                else:
                    epMap[j+t] = E
                if saccade_in_progress:
                    dur = j
                    saccade_in_progress = False
            else:
                # saccade still ongoing
                epMap[j+t] = E

    return epMap, dur
