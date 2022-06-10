from xspec import *
from cross_correlation_search_csv import *
import os

# xcm file containing processed data and an appropriate continuum model
xcm_file = 'f013_TBabs_diskbb_powerlaw_famodel.xcm'

# Parameters of the search
line_widths = np.array([0.025, 0.05, 0.075])
dE=0.08 
E_min = 0.4
E_max = 4.0
line_energies = np.arange(E_min,E_max,dE)
nb_simulations = 5000

path_to_burst_spectra = '/Users/lucas/Documents/IRAP/CrossCorrelationSearch/data/BURST_SPECTRA/burst_0217sig_num1_2050300110_intervals_for1000counts/grouping_step1_bin1'
# Perform the cross correlation search
cross_correlation_search(path_to_burst_spectra,xcm_file, nb_simulations, False, E_min, E_max, dE, *line_widths)
plot_results(path_to_burst_spectra, xcm_file,nb_simulations, line_energies, *line_widths)