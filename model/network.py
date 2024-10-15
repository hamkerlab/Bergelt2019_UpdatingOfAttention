"""
@author: juschu

definition of model
--> define:
        - Neurons including ODEs
        - Populations (= map of neurons organized in given shape)
        - Projections (= connections between populations)
          Projections are mainly calculated with own defined connection pattern
"""


#################
#### imports ####
#################
import sys

from ANNarchy import Neuron, Population, Projection
# get own-defined connection pattern
from ownConnectionPattern import gaussian2dTo4d_h, gaussian2dTo4d_v, gaussian2dTo4d_diag
from ownConnectionPattern import gaussian4dTo2d_h, gaussian4dTo2d_diag, gaussian4d_diagTo4d_v
from ownConnectionPattern import all2all_exp2d, all2all_exp4d

# load parameters
sys.path.append('../parameters/')
from param_network import defParams


##############################
#### Defining the neurons ####
##############################
# Xr
Xr_Neurons = Neuron(
    parameters="""
        A = 'A_Xr' : population
        tau = 'tau_Xr' : population
        baseline = 0.0
    """,
    equations="""
        tau * dr_change/dt + r = baseline * (1 + pos(A-r)*sum(FB)) : min=0.0, max=1.0
        r = if (r_change)<0.00001: 0 else: r_change
    """,
    extra_values=defParams
)

# Xe_PC
XePC_Neurons = Neuron(
    parameters="""
        tau = 'tau_XePC' : population
        baseline = 0.0
    """,
    equations="""
        tau * dr_change/dt + r = pos(baseline) : min=0.0, max=1.0
        r = if (r_change)<0.00001: 0 else: r_change
    """,
    extra_values=defParams
)
# Xe_CD
XeCD_Neurons = Neuron(
    parameters="""
        tau = 'tau_XeCD' : population
        baseline = 0.0
    """,
    equations="""
        tau * dr_change/dt + r = pos(baseline) : min = 0.0, max = 1.0
        r = if (r_change)<0.00001: 0 else: r_change
    """,
    extra_values=defParams
)
# Xe_FEF
XeFEF_Neurons = Neuron(
    parameters="""
        tau = 'tau_XeFEF' : population
        w_inh = 'w_inh_XeFEF' : population
        num_neurons_h = 'layer_size_h' : population
        num_neurons_v = 'layer_size_v' : population
    """,
    equations="""
        num_neurons = num_neurons_h * num_neurons_v * num_neurons_h * num_neurons_v
        inh = r * w_inh * num_neurons * mean(r)
        tau * dr_change/dt + r = sum(FF_CD)*sum(FF_PC) - inh : min = 0.0, max = 1.0
        r = if (r_change)<0.00001: 0 else: r_change
    """,
    extra_values=defParams
)

# Xb_PC
XbPC_Neurons = Neuron(
    parameters="""
        A = 'A_XbPC' : population
        D = 'D_XbPC' : population
        tau = 'tau_XbPC' : population
        w_inh = 'w_inh_XbPC' : population
        num_neurons_h = 'layer_size_h' : population
        num_neurons_v = 'layer_size_v' : population

    """,
    equations="""
        num_neurons = num_neurons_h * num_neurons_v * num_neurons_h * num_neurons_v
        inh = (r+D) * w_inh * num_neurons * mean(r)
        tau * dr_change/dt + r = sum(FF_r)*pos(A - max(r))*sum(FF_PC) + sum(FF_PC)*sum(FB) + sum(exc) - inh : min = 0.0, max = 1.0
        r = if (r_change)<0.00001: 0 else: r_change
    """,
    extra_values=defParams
)
# Xb_CD
XbCD_Neurons = Neuron(
    parameters="""
        A = 'A_XbCD' : population
        D = 'D_XbCD' : population
        tau = 'tau_XbCD' : population
        w_inh = 'w_inh_XbCD' : population
        num_neurons_h = 'layer_size_h' : population
        num_neurons_v = 'layer_size_v' : population
    """,
    equations="""
        num_neurons = num_neurons_h * num_neurons_v * num_neurons_h * num_neurons_v
        inh = (r+D) * w_inh *  num_neurons * mean(r)
        tau * dr_change/dt + r = sum(FF_r)*(1 + pos(A-r)*sum(FF_CD)) + sum(FF_CD)*sum(FB) - inh : min = 0.0, max = 1.0
        r = if (r_change)<0.00001: 0 else: r_change
    """,
    extra_values=defParams
)

# Xh
Xh_Neurons = Neuron(
    parameters="""
        D = 'D_Xh' : population
        tau = 'tau_Xh' : population
        tau_dep = 'tau_dep_Xh' : population
        d_dep = 'd_dep_Xh' : population
        w_inh = 'w_inh_Xh' : population
        num_neurons_h = 'layer_size_h' : population
        num_neurons_v = 'layer_size_v' : population
        baseline = 0.0
    """,
    equations="""
        input = sum(FF_PC) + sum(FF_CD) + baseline
        tau_dep * ds/dt + s = input
        S2 = 1 - d_dep*s : min = 0.0, max = 1.0
        num_neurons = num_neurons_h * num_neurons_v
        inh = (r+D) * w_inh * num_neurons * mean(r)
        tau * dr_change/dt + r = input*S2 + sum(exc) - inh : min = 0.0, max = 1.0
        r = if (r_change)<0.00001: 0 else: r_change
    """,
    extra_values=defParams
)


##################################
#### Creating the populations ####
##################################
# width is supposed to be greater or equal height
size_h = defParams['layer_size_h']
size_v = defParams['layer_size_v']

# Xr
Xr_Pop = Population(name='Xr', geometry=(size_h, size_v), neuron=Xr_Neurons)

# Xe_PC
XePC_Pop = Population(name='Xe_PC', geometry=(size_h, size_v), neuron=XePC_Neurons)
if defParams['split_XePC']:
    # need extra layer for the projection to Xe_FEF ('copy' of Xe_PC)
    XePC_forFEF_Pop = Population(name='Xe_PC_forFEF', geometry=(size_h, size_v),
                                 neuron=XePC_Neurons)
# Xe_CD
XeCD_Pop = Population(name='Xe_CD', geometry=(size_h, size_v), neuron=XeCD_Neurons)
# Xe_FEF
XeFEF_Pop = Population(name='Xe_FEF', geometry=(size_h, size_v, size_h, size_v),
                       neuron=XeFEF_Neurons)

# Xb_PC
XbPC_Pop = Population(name='Xb_PC', geometry=(size_h, size_v, size_h, size_v), neuron=XbPC_Neurons)
# Xb_CD
XbCD_Pop = Population(name='Xb_CD', geometry=(size_h, size_v, size_h, size_v), neuron=XbCD_Neurons)

# Xh
Xh_Pop = Population(name='Xh', geometry=(size_h, size_v), neuron=Xh_Neurons)


##################################
#### Creating the projections ####
##################################
v = float(defParams['vf_h']) # visual field
max_gauss_distance = defParams['max_gauss_distance']/v

## to Xr ##
# - FB (from Xb_PC)
XbPC_Xr = Projection(
    pre=XbPC_Pop,
    post=Xr_Pop,
    target='FB'
).connect_with_func(method=gaussian4dTo2d_h, mv=defParams['K_XbPCXr'],
                    radius=defParams['sigma_XbPCXr']/v, mgd=max_gauss_distance)

## to Xe_FEF ##
# - FF (from Xe_CD)
XeCD_XeFEF = Projection(
    pre=XeCD_Pop,
    post=XeFEF_Pop,
    target='FF_CD'
).connect_with_func(method=gaussian2dTo4d_h, mv=defParams['K_XeCDXeFEF'],
                    radius=defParams['sigma_XeCDXeFEF']/v, mgd=max_gauss_distance)
# - FF (from Xe_PC_forFEF if 'split_XePC' != 0 otherwise from Xe_PC)
XePC_XeFEF = Projection(
    pre=XePC_forFEF_Pop if defParams['split_XePC'] else XePC_Pop,
    post=XeFEF_Pop,
    target='FF_PC'
).connect_with_func(method=gaussian2dTo4d_v, mv=defParams['K_XePCXeFEF'],
                    radius=defParams['sigma_XePCXeFEF']/v, mgd=max_gauss_distance)

# to Xb_PC ##
# - FF (from Xr)
Xr_XbPC = Projection(
    pre=Xr_Pop,
    post=XbPC_Pop,
    target='FF_r'
).connect_with_func(method=gaussian2dTo4d_h, mv=defParams['K_XrXbPC'],
                    radius=defParams['sigma_XrXbPC']/v, mgd=max_gauss_distance)
# - FF (from Xe_PC)
XePC_XbPC = Projection(
    pre=XePC_Pop,
    post=XbPC_Pop,
    target='FF_PC'
).connect_with_func(method=gaussian2dTo4d_v, mv=defParams['K_XePCXbPC'],
                    radius=defParams['sigma_XePCXbPC']/v, mgd=max_gauss_distance)
# - FB (from Xh)
Xh_XbPC = Projection(
    pre=Xh_Pop,
    post=XbPC_Pop,
    target='FB'
).connect_with_func(method=gaussian2dTo4d_diag, mv=defParams['K_XhXbPC'],
                    radius=defParams['sigma_XhXbPC']/v, mgd=max_gauss_distance)
# - exc (from Xb_PC)
XbPC_exc = Projection(
    pre=XbPC_Pop,
    post=XbPC_Pop,
    target='exc'
).connect_with_func(method=all2all_exp4d, factor=defParams['w_exc_XbPC'],
                    radius=defParams['sigma_exc']/v, mgd=max_gauss_distance)

## to Xb_CD ##
# - FF (from Xr)
Xr_XbCD = Projection(
    pre=Xr_Pop,
    post=XbCD_Pop,
    target='FF_r'
).connect_with_func(method=gaussian2dTo4d_h, mv=defParams['K_XrXbCD'],
                    radius=defParams['sigma_XrXbCD']/v, mgd=max_gauss_distance)
# - FF (from Xe_FEF)
XeFEF_XbCD = Projection(
    pre=XeFEF_Pop,
    post=XbCD_Pop,
    target='FF_CD'
).connect_with_func(method=gaussian4d_diagTo4d_v, mv=defParams['K_XeFEFXbCD'],
                    radius=defParams['sigma_XeFEFXbCD']/v, mgd=max_gauss_distance)
# - FB (from Xh)
Xh_XbCD = Projection(
    pre=Xh_Pop,
    post=XbCD_Pop,
    target='FB'
).connect_with_func(method=gaussian2dTo4d_diag, mv=defParams['K_XhXbCD'],
                    radius=defParams['sigma_XhXbCD']/v, mgd=max_gauss_distance)

## to Xh ##
# - FF (from Xb_PC)
XbPC_Xh = Projection(
    pre=XbPC_Pop,
    post=Xh_Pop,
    target='FF_PC'
).connect_with_func(method=gaussian4dTo2d_diag, mv=defParams['K_XbPCXh'],
                    radius=defParams['sigma_XbPCXh']/v, mgd=max_gauss_distance)
# - FF (from Xb_CD)
XbCD_Xh = Projection(
    pre=XbCD_Pop,
    post=Xh_Pop,
    target='FF_CD'
).connect_with_func(method=gaussian4dTo2d_diag, mv=defParams['K_XbCDXh'],
                    radius=defParams['sigma_XbCDXh']/v, mgd=max_gauss_distance)
# - exc (from Xh)
Xh_exc = Projection(
    pre=Xh_Pop,
    post=Xh_Pop,
    target='exc'
).connect_with_func(method=all2all_exp2d, factor=defParams['w_exc_Xh'],
                    radius=defParams['sigma_exc']/v, mgd=max_gauss_distance)
