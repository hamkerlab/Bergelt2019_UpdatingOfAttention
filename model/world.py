"""
@author: juschu

defining the environment for the model
 - generate inputs from 'EVENTS'
 - set inputs for each timestep to neurons

 - inputs are:
         PC signal (pc_sig, pc_forFEF_sig)
         CD signal (cd_sig)
         retinal signal (ret_sig)
         top-down attention signal (att_sig)
"""


##############################
#### imports and settings ####
##############################
import math
import os
import sys
from hashlib import sha1
import numpy as np

# import saccade generator
from saccadeGenerator import vanWetterVanOpstal as sacGen

# saving
from auxFunctions_model import save_dict_to_hdf5

# load parameters
sys.path.append('../parameters/')
from param_network import defParams


####################################
#### defining the input signals ####
####################################
def init_inputsignals(precalcParam, saveDir, count='', subfolder=''):
    '''
    generate inputs from 'EVENTS'
    inputs are: pc_sig, (pc_forFEF_sig), cd_sig, ret_sig, att_sig

    params: precalcParam     -- predefined parameters, e.g. EVENTS
            saveDir          -- directory for saving
            count, subfolder -- parameters defining where to save

    return: results          -- dictionary containing saccade duration
            signals          -- dictionary of inputs
    '''

    ## init
    print 'init inputsignals'

    results = {}

    duration = precalcParam['tend'] - precalcParam['tbegin'] + 1
    size_h = defParams['layer_size_h']
    size_v = defParams['layer_size_v']


    NO_STIM = sys.float_info.max

    # do we need a copy of Xe_PC for the projection to Xe_FEF? (signal not suppressed)
    needCopylayer = defParams['split_XePC']

    # initialize maps
    suppressionMap = np.ones(duration)      # suppression factor during saccades, init with 1.0
    epMap = np.zeros((duration, 2))         # eye position over time, init with [0, 0] deg
    spMap = {}                              # stimuli positions over time
    apMap = {}                              # attention positions over time

    # initialize signals
    # eye position at [0, 0] deg too
    k_PC = defParams['PC_k']
    sig_PC = defParams['PC_sigma']
    xe = esig2d(size_h, size_v, [0, 0], sig_PC, sig_PC, k_PC)
    pc_sig = np.ones((duration, 1, 1)) * xe
    # copy of eye position signal without suppression (if needed)
    if needCopylayer:
        pc_forFEF_sig = np.ones((duration, 1, 1)) * xe

    cd_sig = np.zeros((duration, size_h, size_v))   # corollary discharge signal
    att_sig = np.zeros((duration, size_h, size_v))  # attention signal
    ret_sig = np.zeros((duration, size_h, size_v))  # retinal signal

    ## pass one: generate eye position signal (input for Xe_PC) and
    # corollary discharge signal (input for Xe_CD)
    # as well as eye position, stimulus position, attention position and suppression map
    for i in precalcParam['EVENTS']['order']:
        currentEvent = precalcParam['EVENTS'][i]
        t = currentEvent['time'] - precalcParam['tbegin']

        # event eyepos
        if currentEvent['type'] == 'EVENT_EYEPOS':
            eyepos = currentEvent['value']
            epMap[t:] = eyepos
            xe = esig2d(size_h, size_v, eyepos, sig_PC, sig_PC, k_PC)
            pc_sig[t:] = xe
            if needCopylayer:
                pc_forFEF_sig[t:] = xe

        # event saccade
        if currentEvent['type'] == 'EVENT_SACCADE':
            # 1. generate eye position signal (input for Xe_PC)
            eye0 = epMap[t]                 # source position
            eye1 = currentEvent['value']    # target position

            xe0 = esig2d(size_h, size_v, eye0, sig_PC, sig_PC, k_PC)
            xe1 = esig2d(size_h, size_v, eye1, sig_PC, sig_PC, k_PC)

            # we use the saccade model from vanWetter & vanOpstal
            epMap, dur = sacGen(duration, t, eye0, eye1, epMap, defParams, 'endBySpeed')
            # save saccade duration
            results['saccade_duration'] = dur

            # generate eye position signal (PC signal)
            # offset of old eye position signal
            t_PC_off = t + dur + defParams['PC_off']
            t_PC_off = min(max(0, t_PC_off), duration) # bounded between 0 and duration
            # onset of new eye position signal
            t_PC_on = t + dur + defParams['PC_on']
            t_PC_on = min(max(0, t_PC_on), duration) # bounded between 0 and duration
            # onset of suppression of eye position signal
            t_PC_supp_on = t + dur + defParams['PC_supp_on']
            t_PC_supp_on = min(max(0, t_PC_supp_on), duration) # bounded between 0 and duration
            # offset of suppression of eye position signal
            t_PC_supp_off = t + dur + defParams['PC_supp_off']
            t_PC_supp_off = min(max(0, t_PC_supp_off), duration) # bounded between 0 and duration

            if defParams['XePC_noSuppression']:
                supp_strength = 1.0
            else:
                supp_strength = defParams['PC_supp_strength']

            threshold = defParams['PC_threshold_activity']

            # remove previously existing eye position signal
            # also part which has to be suppressed, if any
            start_removal = min(t_PC_off, t_PC_supp_on)
            pc_sig[start_removal:] = np.zeros((size_h, size_v))
            if needCopylayer:
                pc_forFEF_sig[start_removal:] = np.zeros((size_h, size_v))
            # recreate previously existing eye position signal with proper suppression, if applicable
            for j in xrange(t_PC_supp_on, t_PC_off):
                if (j > t_PC_supp_on) and (j < t_PC_supp_off):
                    supp = supp_strength
                else:
                    supp = 1.0

                if supp * k_PC > threshold:
                    pc_sig[j] += supp * xe0
                if needCopylayer and (k_PC > threshold):
                    pc_forFEF_sig[j] += xe0
            # add new eye position signal
            for j in xrange(t_PC_on, duration):
                if (j > t_PC_supp_on) and (j < t_PC_supp_off):
                    supp = supp_strength
                else:
                    supp = 1.0

                if supp * k_PC > threshold:
                    pc_sig[j] += supp * xe1
                if needCopylayer and (k_PC > threshold):
                    pc_forFEF_sig[j] += xe1
            # add gaussian decay
            for j in xrange(t_PC_off+1, duration):
                if(j > t_PC_supp_on) and (j < t_PC_supp_off):
                    supp = supp_strength
                else:
                    supp = 1.0
                factor = (math.exp(-((j-t_PC_off)*(j-t_PC_off))/
                                   (2.0*defParams['PC_off_decay']*defParams['PC_off_decay'])))
                xe_sig_new = xe0*factor
                if supp * k_PC * factor > threshold:
                    pc_sig[j] = np.maximum(pc_sig[j], supp * xe_sig_new)
                if needCopylayer and (k_PC * factor > threshold):
                    pc_forFEF_sig[j] = np.maximum(pc_forFEF_sig[j], xe_sig_new)
            # add linear buildup
            for j in xrange(t_PC_on-1, max(0, t_PC_on), -1):
                if (j > t_PC_supp_on) and (j < t_PC_supp_off):
                    supp = supp_strength
                else:
                    supp = 1.0
                factor = 1.0-(t_PC_on - j)/float(defParams['PC_on_buildup'])
                xe_sig_new = xe1*factor
                if supp * k_PC * factor > threshold:
                    pc_sig[j] = np.maximum(pc_sig[j], supp * xe_sig_new)
                if needCopylayer and (k_PC * factor > threshold):
                    pc_forFEF_sig[j] = np.maximum(pc_forFEF_sig[j], xe_sig_new)


            # 2. generate corollary discharge signal (input for Xe_CD)
            eye0 = epMap[t]                 # source position
            eye1 = currentEvent['value']    # target position
            # generate CD-Signal
            t_CD = t + defParams['CD_peak'] # peak of CD signal
            k_CD = defParams['CD_k']
            sig_CD = defParams['CD_sigma']
            rise = defParams['CD_rise']
            decay = defParams['CD_decay']
            for j in xrange(t_CD-defParams['CD_range'], t_CD+defParams['CD_range']):
                if (j >= 0) and (j < duration):
                    # CD-Signal is retinotop
                    xo = osig2d(size_h, size_v, eye1-eye0, j-t_CD, sig_CD, sig_CD, rise, decay, k_CD)
                    # are we above the (optional) threshold?
                    # otherwise no CD-Signal will be generated...
                    under_threshold = np.all(xo <= defParams['CD_threshold_activity']*np.ones((size_h, size_v)))
                    if not under_threshold:
                        cd_sig[j] += xo


            # 3. generate suppression map for retinal signal (input for Xr)
            #=========================================================================#
            # "Suppression" of retinal signal during saccades, latency is being added #
            #    *******       **********  ---- 1.0                                   #
            #           *     *                                                       #
            #            *****             ---- supp_strength                         #
            #          | |   | |                                                      #
            #          1 2   3 4                                                      #
            # 1 - "supp_begin"   - measured from saccade onset                        #
            # 2 - "supp_max"     - measured from saccade onset                        #
            # 3 - "supp_release" - measured from saccade offset                       #
            # 4 - "supp_off"     - measured from saccade offset                       #
            #=========================================================================#
            if defParams['ret_suppression']:
                supp_begin = t + defParams['supp_begin'] + defParams['ret_latency']
                supp_max = t + defParams['supp_max'] + defParams['ret_latency']
                supp_release = t + dur + defParams['supp_release'] + defParams['ret_latency']
                supp_off = t + dur + defParams['supp_off'] + defParams['ret_latency']
                for j in xrange(supp_begin, supp_max):
                    supp = (1.0 +
                            (defParams['supp_strength']-1.0)/(defParams['supp_max']-defParams['supp_begin'])
                            *(j-t-(defParams['supp_begin']+defParams['ret_latency'])))
                    suppressionMap[j] = min(suppressionMap[j], supp)
                for j in xrange(supp_max, supp_release):
                    suppressionMap[j] = min(suppressionMap[j], defParams['supp_strength'])
                for j in xrange(supp_release, supp_off):
                    supp = (defParams['supp_strength'] +
                            (1.0-defParams['supp_strength'])/(defParams['supp_off']-defParams['supp_release'])
                            *(j-t-dur-(defParams['supp_release']+defParams['ret_latency'])))
                    suppressionMap[j] = min(suppressionMap[j], supp)

        #event stimulus
        if currentEvent['type'] == 'EVENT_STIMULUS':
            # 4. generate stimulus position map for retinal signal (input for Xr)
            nameOfEvent = currentEvent['name']
            # init spMap for stimulus with "stimulus off"
            if nameOfEvent not in spMap:
                spMap[nameOfEvent] = np.ones((duration, 2))*NO_STIM
            # stimulus on from now on
            spMap[nameOfEvent][t:] = currentEvent['value']

        #event attention
        if currentEvent['type'] == 'EVENT_ATTENTION':
            # 4. generate attention position map for attention signal (input for Xh)
            nameOfEvent = currentEvent['name']
            # init apMap for attention with "attention off"
            if nameOfEvent not in apMap:
                apMap[nameOfEvent] = np.ones((duration, 2))*NO_STIM
            # attention on from now on
            apMap[nameOfEvent][t:] = currentEvent['value']


    ## pass two: generate attention signal (input for Xh)
    k_att = defParams['att_k']
    sig_att = defParams['att_sigma']
    for it in apMap:
        att = np.argwhere(apMap[it] != [NO_STIM, NO_STIM])

        att_start = max(0, att[0][0])
        att_end = min(duration, att[-1][0]+1)
        att_pos = apMap[it][att_start]

        xh = esig2d(size_h, size_v, att_pos, sig_att, sig_att, k_att)
        att_sig[att_start:att_end] += xh


    ## pass three: generate retinal signal (input for Xr)
    k_ret = float(defParams['ret_k'])
    told = 0
    stimpos_old = set()
    eyepos_old = 0
    signal_old = np.zeros((size_h, size_v))
    for tnew in xrange(duration-1):
        if told > duration-1:
            break
        stimpos = set()
        strength = {}
        for it in spMap:
            a_stimpos = spMap[it][tnew]
            if a_stimpos[0] != NO_STIM and a_stimpos[1] != NO_STIM:
                stimpos.add(hashable(a_stimpos))
                strength[hashable(a_stimpos)] = k_ret
        eyepos = epMap[tnew]
        signal = rsig2d(size_h, size_v, eyepos, stimpos, strength)
        if (np.any(stimpos != stimpos_old)) or (np.any(eyepos != eyepos_old)) or (tnew == duration-2):
            # use latency
            ret_start = told + defParams['ret_latency']
            if ret_start >= duration-1:
                break
            ret_end = min(tnew+defParams['ret_latency'], duration-1)
            # 1. generate the actual stimulus (using suppression and depression)
            for t in xrange(ret_start, ret_end):
                # use depression?
                if defParams['ret_depression']:
                    depr = ret_depr(t-ret_start, defParams['ret_tau'], defParams['ret_d'])
                else:
                    depr = 1.0
                ret_sig[t] = np.maximum(ret_sig[t], signal_old * suppressionMap[t] * depr)
            # 2. generate the retinal signal after stimulus release
            # generate the decay after the stimulus is off
            # (again using suppression and depression)
            if ret_end < duration-1:
                decay = min(duration-1-ret_end, defParams['ret_decay'])
                suppression = 1.0
                for t in xrange(ret_end, ret_end+decay):
                    # use depression?
                    if defParams['ret_depression']:
                        depr = ret_depr(t-ret_start, defParams['ret_tau'], defParams['ret_d'])
                    else:
                        depr = 1.0
                    # if the end of a suppression phase is within the decay, prevent the
                    # activity from rising after the suppression phase by extending
                    # the suppression until decay is over
                    suppression = min(suppression, suppressionMap[t])
                    ret_sig[t] = np.maximum(ret_sig[t],
                                            signal_old*suppression*depr*(1-defParams['ret_decay_rate']*(t-ret_end)))
            told = tnew
        signal_old = signal
        stimpos_old = stimpos
        eyepos_old = eyepos


    ## finished ##


    # cut off everything under 1e-5
    limit = 0.00001
    print "set inputs under {0} to 0".format(limit)
    for d in xrange(duration):
        for i in xrange(size_h):
            pc_sig[d, i] = [0 if r < limit else r for r in pc_sig[d, i]]
            cd_sig[d, i] = [0 if r < limit else r for r in cd_sig[d, i]]
            if needCopylayer:
                pc_forFEF_sig[d, i] = [0 if r < limit else r for r in pc_forFEF_sig[d, i]]
            ret_sig[d, i] = [0 if r < limit else r for r in ret_sig[d, i]]
            att_sig[d, i] = [0 if r < limit else r for r in att_sig[d, i]]


    # save inputs and generated stuff if wanted
    if 'save_inputs' in defParams and defParams['save_inputs']:
        # save the new generated inputs as an txt-file
        #ret_sig, pc_sig, (pc_forFEF_sig,) cd_sig, att_sig
        dirname = saveDir+'rates/'+subfolder+'/'
        print 'save inputs at ' + dirname
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        dict_inputs = {str(count)+'_xr_input': ret_sig, str(count)+'_xe_input': pc_sig,
                       str(count)+'_xe2_input': cd_sig, str(count)+'_xh_input': att_sig}
        if needCopylayer:
            dict_inputs[str(count)+'_xe_forFEF_input'] = pc_forFEF_sig

        save_dict_to_hdf5(dict_inputs, dirname+'dict_inputs.hdf5')

    # save eye position over time as an txt-file
    if 'save_eyePosition' in defParams and defParams['save_eyePosition']:
        dirname = saveDir+subfolder+'/'
        print 'save eye position at ' + dirname

        filename = dirname + str(count) + '_eyepos.txt'
        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        strToWrite = ''
        for t in xrange(duration):
            strToWrite += '{0}: {1}\n'.format(t, epMap[t])
        f = open(filename, 'w')
        f.write(strToWrite)
        f.close()

    # save stimulus position over time as an txt-file
    if 'save_stimPosition' in defParams and defParams['save_stimPosition']:
        dirname = saveDir+subfolder+'/'
        print 'save stimulus position at ' + dirname

        filename = dirname + str(count) + '_stimpos.txt'
        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        strToWrite = ''
        for k in spMap:
            strToWrite += 'event {0}: \n'.format(k)
            vec = spMap[k]
            for t in xrange(duration):
                strToWrite += '{0}: {1}\n'.format(t, vec[t])
        f = open(filename, 'w')
        f.write(strToWrite)
        f.close()


    # add saccade duration to results if not existing
    if 'saccade_duration' not in results:
        results['saccade_duration'] = 0

    ## summarize inputs in dictionary
    signals = {'pc_signal': pc_sig, 'cd_signal': cd_sig, 'att_signal': att_sig,
               'ret_signal': ret_sig}
    if needCopylayer:
        signals['pc_forFEF_signal'] = pc_forFEF_sig

    return results, signals


###############################
#### set the input signals ####
###############################
def set_input(t, signals, populations):
    '''
    set the baseline for each population for the current timestep t
    t = act_time - 'tbegin'

    params: t           -- timestep (in ms)
            signals     -- dictionary of inputs
            populations -- all created populations of model
    '''

    #print 'set_input'
    for pop in populations:
        if pop.name == "Xr":
            pop.baseline = signals['ret_signal'][t]

        if pop.name == "Xe_PC":
            pop.baseline = signals['pc_signal'][t]

        if pop.name == "Xe_PC_forFEF":
            pop.baseline = signals['pc_forFEF_signal'][t]

        if pop.name == "Xe_CD":
            pop.baseline = signals['cd_signal'][t]

        if pop.name == "Xh":
            pop.baseline = signals['att_signal'][t]


#############################
#### auxiliary functions ####
#############################
class hashable(object):
    r'''Hashable wrapper for ndarray objects.

        Instances of ndarray are not hashable, meaning they cannot be added to
        sets, nor used as keys in dictionaries. This is by design - ndarray
        objects are mutable, and therefore cannot reliably implement the
        __hash__() method.

        The hashable class allows a way around this limitation. It implements
        the required methods for hashable objects in terms of an encapsulated
        ndarray object. This can be either a copied instance (which is safer)
        or the original object (which requires the user to be careful enough
        not to modify it).
    '''
    def __init__(self, wrapped, tight=False):
        r'''Creates a new hashable object encapsulating an ndarray.

            wrapped
                The wrapped ndarray.

            tight
                Optional. If True, a copy of the input ndaray is created.
                Defaults to False.
        '''
        self.__tight = tight
        self.__wrapped = np.array(wrapped) if tight else wrapped
        self.__hash = int(sha1(wrapped.view(np.uint8)).hexdigest(), 16)

    def __eq__(self, other):
        return all(self.__wrapped == other.__wrapped)

    def __hash__(self):
        return self.__hash

    def unwrap(self):
        r'''Returns the encapsulated ndarray.

            If the wrapper is "tight", a copy of the encapsulated ndarray is
            returned. Otherwise, the encapsulated ndarray itself is returned.
        '''
        if self.__tight:
            return np.array(self.__wrapped)

        return self.__wrapped


def esig2d(dim_h, dim_v, ep, sigma_h, sigma_v, strength):
    '''
    Returns an internal eye position signal given the head-centered eye position in degrees
    also used for attention-signal
    Eq 5: r^Xe_PC,in = strength_PC * exp(...)

    params: dim_*    -- geometry of input
            ep       -- head-centered eye position (in deg)
            sigma_*  -- width of signal (in deg)
            strength -- strength of signal

    return: xe       -- input signal
    '''

    idxs_h = np.arange(float(dim_h))
    deg_h = idx_to_deg(idxs_h, dim_h, defParams['vf_h'])
    idxs_v = np.arange(float(dim_v))
    deg_v = idx_to_deg(idxs_v, dim_v, defParams['vf_v'])
    summand_h = ((ep[0]-deg_h)*(ep[0]-deg_h) / (2.0*sigma_h*sigma_h)).reshape(dim_h, 1)
    summand_v = ((ep[1]-deg_v)*(ep[1]-deg_v) / (2.0*sigma_v*sigma_v)).reshape(1, dim_v)
    xe = strength * np.exp(-(summand_h + summand_v))

    return xe

def osig2d(dim_h, dim_v, st, t, sigma_h, sigma_v, alpha, beta, strength):
    '''
    Returns an internal corollary discharge signal given the eye-centered CD in degrees
    Eq 14: r^Xe_CD,in = strength_CD * exp(...) * S_CD(t)

    params: dim_*       -- geometry of input
            st          -- eye-centered CD (in deg)
            t           -- peak of CD (in ms)
            sigma_*     -- width of signal (in deg)
            alpha, beta -- sigma for rise and decay of CD
            strength    -- strength of signal

    return: xo          -- input signal
    '''

    # time course of CD (S_CD(t), Eq. 16+17)
    # gauss-Version
    if t <= 0:
        # S_CD rises
        sig_S = alpha
    else:
        # S_CD decays
        sig_S = beta
    S = math.exp(-(t*t)/(2.0*sig_S*sig_S))

    idxs_h = np.arange(float(dim_h))
    deg_h = idx_to_deg(idxs_h, dim_h, defParams['vf_h'])
    idxs_v = np.arange(float(dim_v))
    deg_v = idx_to_deg(idxs_v, dim_v, defParams['vf_v'])
    summand_h = (((st[0]-deg_h)*(st[0]-deg_h)) / (2.0*sigma_h*sigma_h)).reshape(dim_h, 1)
    summand_v = (((st[1]-deg_v)*(st[1]-deg_v)) / (2.0*sigma_v*sigma_v)).reshape(1, dim_v)
    xo = strength * np.exp(-(summand_h + summand_v)) * S

    return xo

def rsig2d(dim_h, dim_v, eyepos, hc_stimpos, strengthPerStimpos):
    '''
    Returns an eye-centered retina signal given the head-centered eye position in degrees
    and the head-centered stimulus positions in degrees
    Eq 1: r^Xr,in = (S^Xr) * strength_r * exp(...)

    params: dim_*              -- geometry of input
            eyepos             -- head-centered eye position (in deg)
            hc_stimpos         -- list of head-centered stimulus positions (in deg)
            strengthPerStimpos -- strength of signal for each position

    return: xr                 -- input signal
    '''

    idxs_h = np.arange(float(dim_h))
    deg_h = idx_to_deg(idxs_h, dim_h, defParams['vf_h'])
    idxs_v = np.arange(float(dim_v))
    deg_v = idx_to_deg(idxs_v, dim_v, defParams['vf_v'])

    xr = np.zeros((dim_h, dim_v))
    for it in hc_stimpos:
        strength = strengthPerStimpos[it]
        ec_stimpos = -eyepos + it.unwrap()
        sigma_h = sigma_ret(ec_stimpos[0])
        sigma_v = sigma_ret(ec_stimpos[1])
        summand_h = (((ec_stimpos[0]-deg_h)*(ec_stimpos[0]-deg_h)) / (2.0*sigma_h*sigma_h)).reshape(dim_h, 1)
        summand_v = (((ec_stimpos[1]-deg_v)*(ec_stimpos[1]-deg_v)) / (2.0*sigma_v*sigma_v)).reshape(1, dim_v)
        xr += strength * np.exp(-(summand_h + summand_v))

    return xr

def sigma_ret(ecc):
    '''
    width of receptive field for ret_sig
    p. 3, bottom: sigma(ecc) = b + m*ecc, b = 6.35, m = 0.0875
    ATTENTION: Manipulated b

    params: ecc -- eccentricity (in deg)

    return: sigma
    '''

    #return 0.0875*math.fabs(pos) + 6.3500
    return 0.0875*math.fabs(ecc) + 1.0

def idx_to_deg(idxs, size, vf):
    '''
    Maps neurons to corresponding positions in visual field

    params: idxs -- numpy array of indices of neurons
            size -- geometry of dimension
            vf   -- visual field

    return: numpy array of positions (in deg)
    '''

    return vf*(idxs/(size-1)-0.5)

def ret_depr(t, tau, d):
    '''
    Returns the depression factor for retinal signal
    where the differential equation is statically solved, s.t. ret_depr(0) = 1
    Eq 2+3: tau * ds/dt = 1 - s
            ret_depr = 1 - d*s

    params: t   -- timestep (in ms)
            tau -- time constant
            d   -- depression strength

    return: depression factor for ret_sig
    '''

    I = 1.0
    a = tau * math.log((I + 1.0)/I)
    return 1.0 - d * (math.exp(-t/tau) + I - I*math.exp((a-t)/tau))
