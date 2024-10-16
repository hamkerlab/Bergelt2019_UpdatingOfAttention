# Bergelt2019_UpdatingOfAttention

The repository contains all relevant scripts to reproduce the results and figures published in the article
"Spatial updating of attention across eye movements: A neuro-computational approach" by J. Bergelt and F. H. Hamker, sumbitted in Journal of Vision

The source code is implemented in Python 2.7. The simulations were done with the neural simulator ANNarchy 4.6.  
**Update**  
The source code is implemented in Python 3.10. The simulations were done with the neural simulator ANNarchy 4.8.

To start a simulation, run either predictiveRemapping.py or UpdatingOfAttention.py (in subfolder "model").
The neuro-computational model is defined in network.py and ownConnectionPattern.py (in subfolder "model").
The different parameters defining the experimental setups as well as the neuro-computational model are stored in subfolder "parameters".
To create the figures, run one of the scripts plot_*  (in subfolder "plotting").

## Structure of the repository
   * model
      * auxFunctions_model.py (auxiliary functions)
      * network.py (defines ANNarchy model)
      * ownConnectionPattern.py (defines connection pattern between layer)
      * predictiveRemapping.py (main script for "predictive remapping"-experiment)
      * saccadeGenerator.py (generates saccade after Van Wetter & Van Opstal (2008))
      * UpdatingOfAttention.py (main script for "updating of attention"-experiment)
      * world.py (generates input signals for model)
   * parameters
      * param_network.py (parameters for model, input signals and saccade generator)
      * param_predRemapping (parameters for setup of "predictive remapping"-experiment)
      * param_updateAtt.py (parameters for setup of "updating of attention"-experiment)
   * plotting
      * auxFunctions_plotting.py (auxiliary functions)
      * plot_attEffect(plot attentional effect for "updating of attention"-experiment; Figure 10)
      * plot_predictiveRemapping_results.py (plot/movie of simulation results for "predictive remapping"-experiment; Figure 6, Figure 7, Figure S1)
      * plot_predictiveRemapping_setup.py (plot/movie of spatial and temporal setup for "predictive remapping"-experiment; Figure 5)
      * plot_UpdatingOfAttention_results.py (plot/movie of simulation results for "updating of attention"-experiment; Figure 9, Figure 12, Figure S2, Figure S3)
      * plot_UpdatingOfAttention_setup.py (plot/movie of spatial and temporal setup for "updating of attention"-experiment; Figure 8, Figure 11)

## Dependencies

Neural Simulator ANNarchy 4.6  
python 2.7, numpy 1.14.5, scipy 1.2.0, matplotlib 2.2.3, h5py 2.6.0  
**Update**  
Neural Simulator ANNarchy 4.8  
python 3.10, numpy 1.26.1, scipy 1.14.1, matplotlib 3.9.2, h5py 3.11.0