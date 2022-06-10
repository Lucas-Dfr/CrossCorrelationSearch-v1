from xspec import *
from cross_correlation_search_csv import * 

file_name = 'f013_TBabs_diskbb_powerlaw_famodel.xcm'
Xset.restore(file_name)
Fit.perform()
Plot.device = '/null'
Plot.xAxis = 'keV'
Plot('data')
nb_bins = len(Plot.x())

nb_simulations = 5000
#generate_resid(nb_simulations)

#real_resid = pd.read_csv(os.path.join(output_dir, 'residuals/real-residuals.csv'))
#print(real_resid)


sigmas = np.array([0.025, 0.05, 0.075])
dE=0.08
E_min = 0.4
E_max = 6.0
E_range = np.arange(E_min,E_max,dE)

'''
generate_models(E_min, E_max, dE, *sigmas)

E_range = np.arange(E_min, E_max, dE)
nb_sigma = len(sigmas)
nb_lineE = len(E_range)
    
gaussian_lines = np.zeros((nb_lineE, nb_sigma, nb_bins))
    
for j in range(nb_sigma):
    file_name = 'gaussian-lines-sigma-' + str(j) + '.csv'\
            
    # read the file and turn it into a data frame
    df = pd.read_csv(os.path.join(models_dir, file_name))
        
    for i in range(nb_lineE):
        col_name = 'lineE-' + str(i)
        gaussian_lines[i][j] = df[col_name].to_numpy()

print(gaussian_lines[3][0])
'''

#cross_correlation_search(file_name, nb_simulations, E_min, E_max, dE, *sigmas, new_simulations=True)

plot_results(file_name,E_range, *sigmas)

'''
df= pd.read_csv(os.path.join(resid_dir, 'real-residuals.csv'))
real_resid = df['real-residuals'].to_numpy()

df2 = pd.read_csv(os.path.join(models_dir, 'gaussian-lines-sigma-1.csv'))
line = df2['lineE-14'].to_numpy()

corr = np.correlate(real_resid,line)
print(corr)

'''

'''df = pd.DataFrame({"Name": ["Braund, Mr. Owen Harris","Allen, Mr. William Henry","Bonnell, Miss. Elizabeth",],"Age": [22, 35, 58],"Sex": ["male", "male", "female"],})
for c in df:
    print(df[c])
    print(df[c].to_numpy())'''


tts = pd.read_csv(os.path.join(sig_dir, 'tts-sigma-0.csv')).to_numpy().flatten()
imaxi = np.argmax(tts)
max = np.amax(tts)
print('Maximum significance of %s sigma for width = 0.025 keV and lineE = %s keV' % (max,E_range[imaxi]))
