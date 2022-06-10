from cross_correlation_search_csv import *


def cross_correlation_search_test(xcm_file, nb_simulations: int, new_simulations : bool, E_min : float, E_max : float, dE: float, *line_widths):
    """
    This function performs a gaussian line search in an X-ray spectrum using the cross-correlation method developed by 
    Kosec et al. 2021 in this paper : https://arxiv.org/pdf/2109.14683.pdf

    Args:
        xcm_file : 
            name of the xcm file containing processed data and an appropriate continuum model. User should be in the same directory as xcm_file
            when running the program
        nb simulations (int): 
            number of simulations to perform 
        new_simulations (bool):
            Do you want to generate a new set of simulated data ? (y:True/n:False)
        E_min (float): 
            Minimum energy to search for lines, in keV
        E_max (float): 
            Maximum energy to search for lines, in keV
        dE (float): 
            Energy step to search for lines, in keV
        line_widths (float):
            Line widths to be searched for
    """

    Xset.restore(xcm_file)
    AllData.ignore('4.0-**')
    Fit.perform()
    
    m = AllModels(1)
    m.setPars({21:0.7},{22:0.025},{23:1.0})
    
    #set up the plot interface
    Plot.device = '/null'
    Plot.xAxis = 'keV'
    Plot('data')
    
    # get the number of energy bins used
    nb_bins = len(Plot.x())

    ####
    # GENERATE REAL AND FAKE RESIDUALS
    ####

    if new_simulations:
        print('### GENERATE RESID:')
        generate_resid(nb_simulations)
        print('---> SUCCESS\n')

    ####
    # GENERATE SPECTRAL MODELS
    ####
    
    print('### GENERATE MODELS:')
    generate_models(E_min, E_max, dE, *line_widths)
    print('---> SUCCESS\n')
    
    ####
    # LOAD MODELS AS A 3D ARRAY
    ####
    
    print('### LOAD GAUSSIAN LINES 3D ARRAY:')
    
    # All the gaussian lines will be loaded in a 3D array
    E_range = np.arange(E_min, E_max, dE)
    nb_line_width = len(line_widths)
    nb_lineE = len(E_range)
    
    gaussian_lines = np.zeros((nb_lineE, nb_line_width, nb_bins))
    
    for j in range(nb_line_width):
        file = 'gaussian-lines-sigma-' + str(j) + '.csv'\
            
        # read the file and turn it into a data frame
        df = pd.read_csv(os.path.join(models_dir, file))
        
        for i in range(nb_lineE):
            col_name = 'lineE-' + str(i)
            gaussian_lines[i][j] = df[col_name].to_numpy()
    
    print('---> SUCCESS\n')
    
    ####
    # COMPUTE RAW CROSS CORRELATIONS
    ####
    
    print('### CROSS CORRELATION WITH REAL RESID:')
    
    # start by cleaning the folder
    clear_cc()
    
    # Load real residuals as an array and cross correlate with all the gaussian lines
    df= pd.read_csv(os.path.join(resid_dir, 'real-residuals.csv'))
    real_resid = df['real-residuals'].to_numpy()
        
    cross_correlate(gaussian_lines, os.path.join(cc_dir, 'raw-cc-real'),False,real_resid)
    
    print('---> SUCCESS\n')
    
    # get the number of fake-resid-X.npz files (fake residuals are regrouped in 5000 simulations blocks)
    if nb_simulations % 5000 == 0:
        nb_files = nb_simulations//5000
    else:
        nb_files = nb_simulations//5000 + 1
        
    print('### CROSS CORRELATION WITH FAKE RESID:')
    
    # Compute raw cross correlation between fake residuals and all the gaussian lines
    residuals_sets = []
    for n in range(nb_files):
        df = pd.read_csv(os.path.join(resid_dir, 'fake-resid-'+ str(n) +'.csv'))
        for col_name in df:
            residuals_sets.append(df[col_name].to_numpy())
    cross_correlate(gaussian_lines, os.path.join(cc_dir, 'raw-cc-fake'), True, *residuals_sets)
    
    print('---> SUCCESS\n')
    
    ####
    # COMPUTE NORMALIZATION FACTORS
    ###
    
    print('### COMPUTE NORMALIZATION FACTORS:')
    
    # normalization factors are stored into a matrix. r_pos[i][j] is the normalization factor for positive cc at (lineE_i, sigma_j)
    r_pos = np.zeros((nb_lineE,nb_line_width))
    r_neg = np.zeros((nb_lineE,nb_line_width))
    
    # loop on the files
    for j in range(nb_line_width):
        df = pd.read_csv(os.path.join(cc_dir, 'raw-cc-fake-sigma-' + str(j) + '.csv'))
        for i in range(nb_lineE):
            n_pos = 0
            n_neg = 0
            sum_pos = 0
            sum_neg = 0
            
            # Now that we are at lineE i and sigma j, let's loop on the simulated datasets
            for col_name in df:
                c = df.at[i,col_name]
                if c >0:
                    n_pos += 1
                    sum_pos += c**2
                else: 
                    n_neg += 1
                    sum_neg += c**2
            r_pos[i][j] = np.sqrt((1/n_pos)*sum_pos)
            r_neg[i][j] = np.sqrt((1/n_neg)*sum_neg)
    
    print('---> SUCCESS\n')
    
    ####
    # NORMALIZE EVERY CROSS CORRELATION VALUE
    ###
    
    print('### NORMALIZATION OF CROSS CORRELATION:')
    
    # normalize raw cross correlation of the real and fake residuals and save them
    normalize(r_pos,r_neg)
    
    print('---> SUCCESS\n')
    
    ####
    # COMPUTE P-VALUES AND STS
    ###
    
    print('### COMPUTE STS:') 
    
    single_trial_significance(len(line_widths),len(E_range))
    
    print('---> SUCCESS\n') 
    
    ####
    # COMPUTE TRUE-P-VALUES AND TTS
    ###
    
    print('### COMPUTE TTS:') 
    
    true_trial_significance(len(line_widths),len(E_range))
    
    print('---> SUCCESS\n')




# xcm file containing processed data and an appropriate continuum model
xcm_file = 'f013_TBabs_diskbb_powerlaw_famodel.xcm'

# Parameters of the search
line_widths = np.array([0.025, 0.05, 0.075])
dE=0.08 
E_min = 0.4
E_max = 6.0
line_energies = np.arange(E_min,E_max,dE)
nb_simulations = 5000

# Perform the cross correlation search
cross_correlation_search(xcm_file, nb_simulations, True, E_min, E_max, dE, *line_widths)
plot_results(xcm_file,nb_simulations, line_energies, *line_widths)


