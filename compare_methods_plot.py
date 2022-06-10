import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

path_to_compare_methods_dir = '/Users/lucas/Documents/IRAP/CrossCorrelationSearch/compare_methods'
resu_dir = '/Users/lucas/Documents/IRAP/CrossCorrelationSearch/compare_methods/results'

os.chdir(path_to_compare_methods_dir)
plt.style.use(['science', 'no-latex'])

classic_search_results = pd.read_csv(os.path.join(path_to_compare_methods_dir,'classicSearch-fakeSpectrum-0.csv'),dtype=np.float64).to_numpy()
sqrt_delta_cstat = classic_search_results[:,6]

cc_search_results = pd.read_csv(os.path.join(path_to_compare_methods_dir,'ccSearch-fakeSpectrum-0.csv'),dtype=np.float64).to_numpy()
normalized_cc = cc_search_results.flatten()

fig, ax = plt.subplots()
ax.plot(sqrt_delta_cstat,sqrt_delta_cstat, color = 'black', linewidth = 1)
ax.scatter(sqrt_delta_cstat,normalized_cc, s = 7, color = 'red',marker = 'D',edgecolors='black',linewidths=0.5)
fig.savefig(os.path.join(resu_dir,'CompareMethods.eps'),format = 'eps')