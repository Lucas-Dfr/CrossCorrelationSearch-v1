from compare_methods import *

# xcm file containing processed data and an appropriate continuum model
xcm_file = 'f013_TBabs_diskbb_powerlaw_famodel.xcm'
path_to_burst_spectra = '/Users/lucas/Documents/IRAP/CrossCorrelationSearch/data/BURST_SPECTRA/burst_0217sig_num1_2050300110_intervals_for1000counts/grouping_step1_bin1'
path_to_compare_methods_dir = '/Users/lucas/Documents/IRAP/CrossCorrelationSearch/compare_methods'

# Number of spectra in the controlled sample
nb_spectra = 5

# Parameters of the search
line_widths = [0.05]
dE=0.08 
E_min = 0.4
E_max = 4.0
line_energies = np.arange(E_min,E_max,dE)
nb_simulations = 5000

compare_methods(path_to_compare_methods_dir, path_to_burst_spectra, xcm_file, nb_spectra, False, nb_simulations, E_min, E_max, dE, *line_widths)
compare_methods_plot(path_to_compare_methods_dir, nb_spectra+10)

