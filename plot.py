import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
import copy
import sys
# Insert path to utilities.py
sys.path.insert(0, '/home/piariane/fs-sunray/data/Masterthesis2016/aosoppa_triplet/')
from utilities import *

#######################
#      Plotting       #
#######################

# Which methods are you interested in?
total_methods = ['RPA', 'RPA(D)', 'HRPA', 'HRPA(D)', 's-HRPA(D)', 'SOPPA', 'SOPPA(CC2)', 'SOPPA(CCSD)']
requested_methods = ['RPA', 'RPA(D)', 'HRPA', 'HRPA(D)', 's-HRPA(D)', 'SOPPA']

# Do you want to include excitation weights?
include_weights = True

# Collect data in arrays (energies read from 'total_energy' file weights from 'total_weight')
exci_dict, error_dict, abs_exci_dict, weight_dict = read_data('total_energy', 'total_weight', 'cc3_exci', total_methods, requested_methods, include_weights)

# Calculate statistical measures of errors
stats = statistics(requested_methods, error_dict, 'save', 'singlet_rpa-soppa_stats_new.tex')

# Plot stats (This appearance of this plot is not fine tuned)
#plot_hist(stats, requested_methods)

# Plot error distributions plus stats (if parameter below is set accordingly)
stat_nostat = 'stat' # if no stat is wanted, use 'nostat'
#plot_distributions_2(requested_methods, error_dict, stat_nostat, stats, 'singlet', 'save', 'singlet_rpa-soppa_dist_test.pdf', 'A4')

# Plot errors (or energies) on y-axis and the individual states on the x-axis.
#requested_methods = ['RPA', 'RPA(D)', 'SOPPA']
#plot_excitation('singlet', error_dict, requested_methods, 'show', 'rpa_rpad_soppa_excitations_test.pdf')

# The following doesn't work !!!
#plot_excitation_and_weights(error_dict, weight_dict, requested_methods)

#---------------------------------------
# Make correlation plots between methods
#
# (requested_methods_1: y-axis)
# (requested_methods_2: x-axis)
#---------------------------------------

requested_methods_1 = ['RPA(D)']
requested_methods_2 = ['RPA']
#plot_reg(requested_methods, error_dict, error_dict, requested_methods_1, requested_methods_2, -2, 2, 'error (eV)', 'show', 'half_A4', 'singlet_rpa_errors_vs_rpad_errors.pdf')

requested_methods_1 = ['SOPPA']
requested_methods_2 = ['SOPPA']
#plot_reg(requested_methods,weight_dict, error_dict, requested_methods_1, requested_methods_2, -1, 1, '117 singlet excited states', 'show', 'half_A4', 'singlet_soppa_errors_vs_soppa_weights_4.pdf')
#
requested_methods_1 = ['RPA', 'RPA(D)']
requested_methods_2 = ['SOPPA', 'SOPPA']
# lower_y and upper_y are the diemensions of the y-axis
lower_y = -2
upper_y = 7
#plot_reg(requested_methods, weight_dict, error_dict, requested_methods_1, requested_methods_2, lower_y, upper_y, '117 singlet excited states', 'show', 'half_A4', 'singlet_rpa-rpad_errors_vs_soppa_weights_5.pdf')
#plot_reg(requested_methods, exci_dict, error_dict, requested_methods_1, requested_methods_2, lower_y, upper_y, 'excitation energies (eV)', 'show', 'half_A4', 'singlet_rpa-rpad_errors_vs_soppa_exci.pdf')
##
requested_methods_1 = ['HRPA', 'HRPA(D)', 's-HRPA(D)']
requested_methods_2 = ['SOPPA', 'SOPPA', 'SOPPA']
lower_y = -2
upper_y = 7
#plot_reg(requested_methods, weight_dict, error_dict, requested_methods_1, requested_methods_2, lower_y, upper_y, '117 singlet excited states', 'save', 'half_A4', 'singlet_hrpa-hrpad-shrpad_errors_vs_soppa_weights_5.pdf')
#plot_reg(requested_methods, exci_dict, error_dict, requested_methods_1, requested_methods_2, lower_y, upper_y, 'excitation energies (eV)', 'save', 'half_A4', 'singlet_hrpa-hrpad-shrpad_errors_vs_soppa_exci.pdf')
##
requested_methods_1 = ['RPA(D)', 's-HRPA(D)', 'SOPPA']
requested_methods_2 = ['SOPPA', 'SOPPA', 'SOPPA']
lower_y = -2
upper_y = 7
#plot_reg(requested_methods, weight_dict, error_dict, requested_methods_1, requested_methods_2, lower_y, upper_y, '117 singlet excited states', 'save', 'half_A4', 'singlet_rpad-shrpad-soppa_errors_vs_soppa_weights_5.pdf')
#plot_reg(requested_methods, exci_dict, error_dict, requested_methods_1, requested_methods_2, lower_y, upper_y, 'excitation energies (eV)', 'save', 'half_A4', 'singlet_rpad-shrpad-soppa_errors_vs_soppa_exci.pdf')
#
#requested_methods_1 = ['RPA(D)', 'HRPA(D)', 's-HRPA(D)', 'SOPPA']
#requested_methods_2 = ['SOPPA', 'SOPPA', 'SOPPA', 'SOPPA']
plot_reg(requested_methods, weight_dict, error_dict, requested_methods_1, requested_methods_2, lower_y, upper_y, '117 singlet excited states', 'show', 'half_A4', 'singlet_rpad-hrpad-shrpad-soppa_errors_vs_soppa_weights_5.pdf')

#----------------------------------
# Make plots with DC contributions
#----------------------------------

# RPA(D)

dc_123_labels = ['$\omega^{\{0\}}_{\mathrm{ph}}$ \nRPA ', '+ $\omega^{\{2\}}_{\mathrm{ph}}$ \n ', '+ $\omega^{\{2\}}_{\mathrm{2p2h}}$ \n RPA(D)']
dc_dict, dc_error_dict, dc_abs_error_dict = read_dc_contributions(dc_123_labels, 'rpad_exci_contribution')
#stats = statistics(dc_123_labels, dc_error_dict, 'save', 'singlet_rpad_stats.tex')
#plot_distributions_2(dc_123_labels, dc_error_dict, 'stat', stats, 'singlet', 'save', 'singlet_rpad_contributions_stats_2.pdf', 'half_A4')
#plt.close()

# s-HRPA(D)

dc_123_labels = ['$\omega^{\{0\}}_{\mathrm{ph}}$ \nHRPA ', '+ $\omega^{\{2\}*}_{\mathrm{ph}}$ \n ', '+ $\omega^{\{2\}}_{\mathrm{2p2h}}$ \n s-HRPA(D)']
dc_dict, dc_error_dict, dc_abs_error_dict = read_dc_contributions(dc_123_labels, 'shrpad_exci_contribution')
#stats = statistics(dc_123_labels, dc_error_dict, 'save', 'singlet_shrpad_stats.tex')
#plot_distributions_2(dc_123_labels, dc_error_dict, 'stat', stats, 'singlet', 'save', 'singlet_shrpad_contributions_stats_test.pdf', 'half_A4')
#
