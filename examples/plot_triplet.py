import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
import copy
import sys
#sys.path.insert(0, '/home/zth702/data/Masterthesis2016/aosoppa_triplet/')
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
exci_dict, error_dict, abs_error_dict, weight_dict = read_data('total_energy', 'total_weight', 'cc3_exci', total_methods, requested_methods, include_weights)

#-----------------------------------------------
# Exclude imaginary RPA and RPA(D) excitations
# (resulting in 55 states)
#-----------------------------------------------
exci_dict_4, error_dict_4, abs_error_dict_4, weight_dict_4 = exclude_imag_rpa(exci_dict, error_dict, abs_error_dict, 'dc_false', weight_dict, requested_methods, 55)

#-------------------------------------------------------------------------------------
# Exclude imaginary RPA and RPA(D) exciations as well as RPA errors smaller than -3:
# (resulting in 50 states)
#-------------------------------------------------------------------------------------
#exci_dict_4, error_dict_4, abs_error_dict_4, weight_dict_4 = exclude_imag_rpa_new(exci_dict, error_dict, abs_error_dict, 'dc_false', weight_dict, requested_methods, 50)

# What is this for?
#dc_dict, dc_error_dict, dc_abs_error_dict = exclude_imag_rpa(dc_dict, dc_error_dict, dc_abs_error_dict, 'dc_true', 'dummy', dc_123_labels, 55)
#stats = statistics(dc_123_labels, dc_error_dict, 'save', 'triplet_rpad_stats.tex')
#plot_distributions_2(dc_123_labels, dc_error_dict, 'stat', stats, 'triplet', 'save', 'triplet_rpad_contributions_stats.pdf', 'half_A4')

#-----------------------------------------------
# Calculate statistical measures of errors
#-----------------------------------------------
stats = statistics(requested_methods, error_dict_4, 'save', 'triplet_rpa-soppa_stats_new_55_test.tex')

#-----------------------------------------------
# Plot stats (This appearance of this plot is not fine tuned)
#-----------------------------------------------
#plot_hist(stats, requested_methods)

#-----------------------------------------------
# Plot error distributions plus stats (if parameter below is set accordingly)
#-----------------------------------------------
stat_nostat = 'stat' # if no stat is wanted, use 'nostat'
#plot_distributions_2(requested_methods, error_dict_4, stat_nostat, stats, 'triplet', 'show', 'triplet_rpa-soppa_dist_50_new3.pdf', 'A4')

#-----------------------------------------------
# Plot errors (or energies) on y-axis and the individual states on the x-axis.
#-----------------------------------------------
#requested_methods = ['RPA', 'RPA(D)', 'SOPPA']
#plot_excitation('triplet', error_dict_4, requested_methods, 'show', 'rpa_rpad_soppa_excitations.pdf')

#-----------------------------------------------
# Plot errors (or energies) on y1-axis and weights on y2-axis and the individual states on the x-axis.
#-----------------------------------------------
requested_methods_1 = ['RPA']
requested_methods_2 = ['SOPPA']

#plot_excitation_and_weights('triplet', error_dict, weight_dict, requested_methods_1, requested_methods_2, 'save', 'triplet_soppa_errors_weights.pdf')
#plot_excitation_and_weights('triplet', error_dict_4, weight_dict_4, requested_methods_1, requested_methods_2, 'show', '')

#---------------------------------------
# Make correlation plots between methods
#
# (requested_methods_1: y-axis)
# (requested_methods_2: x-axis)
#---------------------------------------

## Backtrace cc3 energies if needed.
#cc3 = dict()
#for i in requested_methods:
#    cc3[i] = exci_dict_4[i] - error_dict_4[i]

requested_methods_1 = ['RPA(D)']
requested_methods_2 = ['RPA']
#plot_reg(requested_methods, error_dict_4, error_dict_4, requested_methods_1, requested_methods_2, -2, 2, 'error (eV)', 'save', 'half_A4', 'triplet_rpa_errors_vs_rpad_errors.pdf')

requested_methods_1 = ['SOPPA']
requested_methods_2 = ['SOPPA']
#plot_reg(weight_dict, error_dict, requested_methods_1, requested_methods_2, '71 triplet excited states', 'save', 'half_A4', 'triplet_soppa_errors_vs_soppa_weights_4.pdf')
#
requested_methods_1 = ['RPA', 'RPA(D)']
requested_methods_2 = ['SOPPA', 'SOPPA']
lower_y = -4
upper_y = 5
#plot_reg(requested_methods, weight_dict_4, error_dict_4, requested_methods_1, requested_methods_2, lower_y, upper_y, 'single excitation weights (%)', 'save', 'half_A4', 'triplet_rpa-rpad_errors_vs_soppa_weights_6.pdf')
#plot_reg(requested_methods, cc3, error_dict_4, requested_methods_1, requested_methods_2, lower_y, upper_y, 'excitation energy (eV)', 'save', 'half_A4', 'triplet_rpa-rpad_errors_vs_cc3_exci_2.pdf')
#plot_reg(requested_methods, exci_dict_4, error_dict_4, requested_methods_1, requested_methods_2, lower_y, upper_y, 'excitation energy (eV)', 'save', 'half_A4', 'triplet_rpa-rpad_errors_vs_soppa_exci.pdf')

requested_methods_1 = ['HRPA', 'HRPA(D)', 's-HRPA(D)']
requested_methods_2 = ['SOPPA', 'SOPPA', 'SOPPA']
lower_y = -2
upper_y = 7
#plot_reg(requested_methods, weight_dict_4, error_dict_4, requested_methods_1, requested_methods_2, lower_y, upper_y, 'single excitation weights (%)', 'save', 'half_A4', 'triplet_hrpa-hrpad-shrpad_errors_vs_soppa_weights_6.pdf')
#plot_reg(requested_methods, cc3, error_dict_4, requested_methods_1, requested_methods_2, lower_y, upper_y, 'excitation energy (eV)', 'save', 'half_A4', 'triplet_hrpa-hrpad-shrpad_errors_vs_cc3_exci_2.pdf')
#plot_reg(requested_methods, exci_dict_4, error_dict_4, requested_methods_1, requested_methods_2, lower_y, upper_y, 'excitation energy (eV)', 'save', 'half_A4', 'triplet_hrpa-hrpad-shrpad_errors_vs_soppa_exci.pdf')

requested_methods_1 = ['RPA(D)', 's-HRPA(D)', 'SOPPA']
requested_methods_2 = ['SOPPA', 'SOPPA', 'SOPPA']
lower_y = -4
upper_y = 5
#plot_reg(requested_methods, weight_dict_4, error_dict_4, requested_methods_1, requested_methods_2, lower_y, upper_y, 'single excitation weights (%)', 'save', 'half_A4', 'triplet_rpad-shrpad-soppa_errors_vs_soppa_weights_6.pdf')
#plot_reg(requested_methods, cc3, error_dict_4, requested_methods_1, requested_methods_2, lower_y, upper_y, 'excitation energy (eV)', 'save', 'half_A4', 'triplet_rpad-shrpad-soppa_errors_vs_cc3_exci_2.pdf')
#plot_reg(requested_methods, exci_dict_4, error_dict_4, requested_methods_1, requested_methods_2, lower_y, upper_y, 'excitation energy (eV)', 'save', 'half_A4', 'triplet_rpad-shrpad-soppa_errors_vs_soppa_exci.pdf')

#----------------------------------
# Make plots with DC contributions
#----------------------------------

# RPA(D)

dc_123_labels = ['$\omega^{\{0\}}_{\mathrm{ph}}$ \nRPA ', '+ $\omega^{\{2\}}_{\mathrm{ph}}$ \n ', '+ $\omega^{\{2\}}_{\mathrm{2p2h}}$ \n RPA(D)']
dc_dict, dc_error_dict, dc_abs_error_dict = read_dc_contributions(dc_123_labels, 'rpad_exci_contribution')
#dc_dict, dc_error_dict, dc_abs_error_dict = exclude_imag_rpa(dc_dict, dc_error_dict, dc_abs_error_dict, 'dc_true', 'dummy', dc_123_labels, 55)
dc_dict, dc_error_dict, dc_abs_error_dict = exclude_imag_rpa_new(dc_dict, dc_error_dict, dc_abs_error_dict, 'dc_true', weight_dict, dc_123_labels, 50)
stats = statistics(dc_123_labels, dc_error_dict, 'save', 'triplet_dc_rpad_stats.tex')
plot_distributions_2(dc_123_labels, dc_error_dict, 'stat', stats, 'triplet', 'show', 'triplet_dc_rpad_contributions_stats_2.pdf', 'half_A4')
#plt.close()

# s-HRPA(D)

dc_123_labels = ['$\omega^{\{0\}}_{\mathrm{ph}}$ \nHRPA ', '+ $\omega^{\{2\}*}_{\mathrm{ph}}$ \n ', '+ $\omega^{\{2\}}_{\mathrm{2p2h}}$ \n s-HRPA(D)']
dc_dict, dc_error_dict, dc_abs_error_dict = read_dc_contributions(dc_123_labels, 'shrpad_exci_contribution')
#stats = statistics(dc_123_labels, dc_error_dict, 'save', 'triplet_shrpad_stats.tex')
#plot_distributions_2(dc_123_labels, dc_error_dict, 'stat', stats, 'triplet', 'save', 'triplet_shrpad_contributions_stats_2.pdf', 'half_A4')

