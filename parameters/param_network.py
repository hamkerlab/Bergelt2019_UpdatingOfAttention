# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 14:14:01 2018

@author: juschu

parameters for model, input signals and saccade generator
general parameters
"""

defParams = {}

###############
#### model ####
###############
## Neuron's ODEs ##
defParams['tau_XbCD'] = 10.0
defParams['tau_XbPC'] = 10.0
defParams['tau_XeCD'] = 10.0
defParams['tau_XeFEF'] = 10.0
defParams['tau_XePC'] = 10.0
defParams['tau_Xh'] = 10.0
defParams['tau_Xr'] = 10.0
# saturation
defParams['A_XbCD'] = 0.5
defParams['A_XbPC'] = 1.0
defParams['A_Xr'] = 0.5
# inhibition
defParams['D_XbCD'] = 0.1
defParams['D_XbPC'] = 0.1
defParams['D_Xh'] = 0.6
# synaptic suppression for Xh
defParams['d_dep_Xh'] = 2.2
defParams['tau_dep_Xh'] = 10000.0

## Connections (all defined as Gaussian) ##
# highest value of Gaussian
defParams['K_XbCDXh'] = 0.015
defParams['K_XbPCXh'] = 0.015
defParams['K_XbPCXr'] = 3.0
defParams['K_XeCDXeFEF'] = 5.0
defParams['K_XeFEFXbCD'] = 20.0
defParams['K_XePCXbPC'] = 1.0
defParams['K_XePCXeFEF'] = 10.0
defParams['K_XhXbCD'] = 1.3
defParams['K_XhXbPC'] = 1.3
defParams['K_XrXbCD'] = 2.0
defParams['K_XrXbPC'] = 2.
defParams['w_exc_XbPC'] = 0.6
defParams['w_exc_Xh'] = 0.4
defParams['w_inh_XbCD'] = 0.02
defParams['w_inh_XbPC'] = 0.04
defParams['w_inh_XeFEF'] = 0.2
defParams['w_inh_Xh'] = 0.1
# width of Gaussian
defParams['sigma_XbCDXh'] = 7.5
defParams['sigma_XbPCXh'] = 7.5
defParams['sigma_XbPCXr'] = 0.5
defParams['sigma_XeCDXeFEF'] = 0.5
defParams['sigma_XeFEFXbCD'] = 0.5
defParams['sigma_XePCXbPC'] = 2.0
defParams['sigma_XePCXeFEF'] = 1.0
defParams['sigma_XhXbCD'] = 2.0
defParams['sigma_XhXbPC'] = 2.0
defParams['sigma_XrXbCD'] = 0.5
defParams['sigma_XrXbPC'] = 0.5
defParams['sigma_exc'] = 0.5
defParams['sigma_exc_Xr'] = 6.0
# limitation of width of Gaussian
defParams['max_gauss_distance'] = 0     # no limitation


#######################
#### input signals ####
#######################
## corollary discharge ##
defParams['CD_range'] = 500
# threshold
defParams['CD_threshold_activity'] = 0
# strength
defParams['CD_k'] = 0.25
# width
defParams['CD_sigma'] = 1.0
# rise and decay
defParams['CD_decay'] = 50              # sigma for decay
defParams['CD_rise'] = 65               # sigma for rise
defParams['CD_peak'] = 10               # peak time relative to saccde onset

## proprioceptive eye position ##
# threshold
defParams['PC_threshold_activity'] = 0
# strength
defParams['PC_k'] = 0.3
# width
defParams['PC_sigma'] = 1.0
# update
defParams['PC_off'] = 32
defParams['PC_off_decay'] = 25
defParams['PC_on'] = 32
defParams['PC_on_buildup'] = 9000
# suppression
defParams['PC_supp_off'] = 32
defParams['PC_supp_on'] = -50
defParams['PC_supp_strength'] = 0.1
defParams['XePC_noSuppression'] = 1
defParams['XePC_suppression_not_for_FEF'] = 1
defParams['split_XePC'] = defParams['XePC_suppression_not_for_FEF']^defParams['XePC_noSuppression']

## retinal signal ##
# strength
defParams['ret_k'] = 0.3
# latency
defParams['ret_latency'] = 50
# decay
defParams['ret_decay'] = 40
defParams['ret_decay_rate'] = 0.025
# synaptic depression
defParams['ret_depression'] = 1
defParams['ret_d'] = 0.8
defParams['ret_tau'] = 40.0
# synaptic suppression
defParams['ret_suppression'] = 0
defParams['supp_begin'] = -30
defParams['supp_max'] = -10
defParams['supp_off'] = 150
defParams['supp_release'] = 100
defParams['supp_strength'] = 0.1

## top-down attention ##
# strength
defParams['att_k'] = 0.3
# width
defParams['att_sigma'] = 1.0


###########################
#### saccade generator ####
###########################
defParams['sac_endspeed'] = 22
defParams['sac_m0'] = 7.0
defParams['sac_offset_threshold'] = 0.2
defParams['sac_set_end_to_target'] = 1
defParams['sac_vpk'] = 0.525


#################
#### general ####
#################
## saving
defParams['save_eyePosition'] = 1
defParams['save_inputs'] = 1
defParams['save_stimPosition'] = 1

# visual field
defParams['vf_h'] = 40
defParams['vf_v'] = 30
## number of neurons ##
defParams['layer_size_h'] = defParams['vf_h']/2 + 1
defParams['layer_size_v'] = defParams['vf_v']/2 + 1
