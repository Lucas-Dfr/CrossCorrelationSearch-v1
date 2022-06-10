from xmlrpc.client import Boolean
from pyparsing import line
from sympy import erfcinv, maximum
from xspec import *
import numpy as np
import pandas as pd
import os
from pathlib import Path
from scipy import special
import matplotlib.pyplot as plt

# Users should change this lines so that they matches the correct directory
output_dir = Path(
    '/Users/lucas/Documents/IRAP/CrossCorrelationSearch/output-files')

# No need to change this
resid_dir = os.path.join(output_dir, 'residuals')
models_dir = os.path.join(output_dir, 'models')
cc_dir = os.path.join(output_dir, 'cross-correlations')
sig_dir = os.path.join(output_dir, 'significance')


def cross_correlation_search(path_to_burst_spectra :str, xcm_file, nb_simulations: int, new_simulations : bool, E_min : float, E_max : float, dE: float, *line_widths):
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
    os.chdir(path_to_burst_spectra)


    
    Xset.restore(xcm_file)
    #AllData.ignore('4.0-**')
    Fit.perform()
    
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
        df = pd.read_csv(os.path.join(models_dir, file), dtype = np.float64)
        
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
    df= pd.read_csv(os.path.join(resid_dir, 'real-residuals.csv'), dtype = np.float64)
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
        df = pd.read_csv(os.path.join(resid_dir, 'fake-resid-'+ str(n) +'.csv'), dtype = np.float64)
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
        df = pd.read_csv(os.path.join(cc_dir, 'raw-cc-fake-sigma-' + str(j) + '.csv'), dtype = np.float64)
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
    
    # start by cleaning the folder
    clear_sig()
    
    print('### COMPUTE STS:') 
    
    single_trial_significance(len(line_widths),len(E_range))
    
    print('---> SUCCESS\n') 
    
    ####
    # COMPUTE TRUE-P-VALUES AND TTS
    ###
    
    print('### COMPUTE TTS:') 
    
    true_trial_significance(len(line_widths),len(E_range))
    
    print('---> SUCCESS\n') 

    ####
    # PLOT RESULTS
    ###
    #plot_results(file_name)
    
    






def generate_resid(nb_simulations: int):
    """
    This function generate both real and simulated residual spectra. 

    The simulated residuals are stored in files by large blocks, for example by storing 5000 individual 
    simulations in a single file. This grouping results in a large table where the columns are individual simulations
    and the rows correspond to the same energy bins

    Args:
        n (int): number of simulations to perform
    """
    # start by cleaning the folder
    clear_residuals()

    # set up the plot
    Plot.device = '/null'
    Plot.xAxis = 'keV'

    # generate real residual spectrum and save it as an .npy file
    Plot('resid')
    
    real_resid = pd.Series(Plot.y(), name='real-residuals', dtype= np.float64)
    real_resid.to_csv(os.path.join(output_dir, 'residuals/real-residuals.csv'), index = False)

    # generate n fake residual spectra and grouping them in file blocks of 5000 spectra

    # list to store all fake residual spectra
    all_fake_resid = []

    # get noticed channels in the real spectrum
    noticed_ch = AllData(1).noticedString()

    for k in range(nb_simulations):
        # create FakeitSettings object so that the file names are consistent
        name = 'fakeSpectum-' + str(k)
        fs = FakeitSettings(fileName=name)
        AllData.fakeit(1, [fs])

        # make sure that the same channels are noticed in real and fake spectra
        AllData.ignore('1-**')
        AllData.notice(noticed_ch)

        #Fit.perform() sinon le modele de reference change tout le temps comme il est fit. on veut les residuals par rapport au modele de base à chque fois

        Plot('resid')

        # get the values of the fake residuals as a column
        col_name = 'f' + str(k)
        fake_resid = pd.Series(Plot.y(), name = col_name, dtype = np.float64)

        # append the list of fake residuals to the file
        all_fake_resid.append(fake_resid)
    # group the arrays in file blocks of 5000 spectra
    count = 0
    head = 0
    tail = 5000
    if nb_simulations <= 5000:
        table = pd.concat(all_fake_resid, axis=1)
        table.to_csv(os.path.join(output_dir, 'residuals/fake-resid-' + str(count) + '.csv'), index=False)
    else:
        # groups of 5000
        for k in range(nb_simulations//5000):
            table = pd.concat(all_fake_resid[head:tail], axis=1)
            table.to_csv(os.path.join(output_dir, 'residuals/fake-resid-' + str(count) + '.csv'), index = False)
            head = tail
            tail += 5000
            count += 1

        # last group might contain less than 5000 spectra
        if len(all_fake_resid[head:]):
            table = pd.concat(all_fake_resid[head:], axis=1)
            table.to_csv(os.path.join(output_dir, 'residuals/fake-resid-' + str(count) + '.csv'), index = False)


def clear_residuals():
    """
    This function empties the output_files/residuals
    """
    for f in os.listdir(resid_dir):
        os.remove(os.path.join(resid_dir, f))



def generate_models(E_min: float, E_max: float, dE: float, *line_widths):
    """
    This function generates gaussian lines with parameters varying according to a grid. The models are then regrouped by their
    sigma values resulting in large tables where each column is a value of lineE and the rows corresponds to energy bins
    Args:
        E_min (float): 
            Minimum energy to search for lines, in keV
        E_max (float): 
            Maximum energy to search for lines, in keV
        dE (float): 
            Energy step to search for lines, in keV (should slightly over sample the spectral resolution)
        line_widths (list):
            Line widths to be searched for
    """
    # start by cleaning the folder
    clear_models()

    # set up the plot interface
    Plot.device = '/null'
    Plot.xAxis = 'keV'
    Plot('data')

    # create a gaussian line to vary the parameter
    m = Model('gaussian')

    # set energy range
    E_range = np.arange(E_min, E_max, dE)

    # spectral models stored in len(line_widths) individual csv files
    count_sig = 0
    for sig in line_widths:
        # set width of the gaussian line (remains constant in a file)
        m.setPars({2: sig})

        # list to contain all models from a file
        all_sigma_lines = []

        count_e = 0
        for e in E_range:
            # change th lineE parameter of the gaussian line
            m.setPars(e)

            Plot('data')
            # fill a an line array with the model values
            col_name = 'lineE-' + str(count_e)
            model_values = pd.Series(Plot.model(), name = col_name, dtype = np.float64)

            all_sigma_lines.append(model_values)
            count_e += 1
        
        table = pd.concat(all_sigma_lines, axis=1)
        table.to_csv(os.path.join(models_dir, 'gaussian-lines-sigma-' + str(count_sig) + '.csv'), index=False)
        count_sig += 1


def clear_models():
    """
    This function empties output-files/models directory
    """
    for f in os.listdir(models_dir):
        os.remove(os.path.join(models_dir, f))
        

def cross_correlate(gaussian_lines: np.array, fileName: str, fake = True, *residuals_sets):
    """
    This function performs the raw cross correlation between a 3D array containing all the gaussian lines and sets of residuals 
    and saves the results as a fileName-sigma-X.csv file. Each file corresponds to a value of sigma and is a large table where the 
    columns are residuals sets and the row are values of lineE. 

    Args:
        residuals_sets (np.array): 
            Arrays containing residuals values from a spectrum
        gaussian_lines (np.array): 
            3D array representing a grid of gaussian lines
        fileName (str): 
            Name of the file where to save the results of the cross correlation
        fake (bool):
            Set fake to True for simulated dataset and to false for real dataset
    """
    
    nb_line_width = len(gaussian_lines[0])
    nb_lineE = len(gaussian_lines)
    nb_columns = len(residuals_sets)
    
    for j in range(nb_line_width):
        # store all the cross correlation of a given sigma as list of pandas series
        raw_cc_all = []
        for k in range(nb_columns):
            raw_cc_values = np.zeros(nb_lineE)
            for i in range(nb_lineE):
                raw_cc_values[i] = np.correlate(residuals_sets[k], gaussian_lines[i][j])[0]
                
            # convert the array to a series and add it to the list of all series
            if fake: 
                col_name = 'f-' + str(k)
                raw_cc_all.append(pd.Series(raw_cc_values, name = col_name))
            else:
                raw_cc_all.append(pd.Series(raw_cc_values, name = 'real_residuals'))
                
        table = pd.concat(raw_cc_all, axis=1)
        table.to_csv(os.path.join(cc_dir,fileName + '-sigma-' + str(j) + '.csv'), index = False)
        
def clear_cc():
    """
    This function empties the output_files/cross-correlations
    """
    for f in os.listdir(cc_dir):
        os.remove(os.path.join(cc_dir, f))



def normalize(r_pos: np.array, r_neg: np.array):
    """
    This function normalizes the cross correlation values for both real and fake residuals, fake residuals cross correlations
    are then ordered within their energy/width bins

    Args:
        r_pos (np.array):
            Matrix containing the positive normalization factor values
        r_neg (np.array):
            Matrix containing the negative normalization factor values
    
    """
    nb_line_width = len(r_pos[0])
    nb_lineE = len(r_pos)
    
    for j in range(nb_line_width):
        
        raw_real_cc = pd.read_csv(os.path.join(cc_dir, 'raw-cc-real-sigma-' + str(j) + '.csv'), dtype = np.float64).to_numpy()
        raw_fake_cc = pd.read_csv(os.path.join(cc_dir, 'raw-cc-fake-sigma-' + str(j) + '.csv'), dtype = np.float64).to_numpy()
        
        for i in range(nb_lineE):
            
            # renormalize every cross-correlation value for fake datasets
            nb_simulations = len(raw_fake_cc[0]) 
            for k in range(nb_simulations):
                if raw_fake_cc[i][k]>0:
                    raw_fake_cc[i][k] = raw_fake_cc[i][k]/r_pos[i][j]
                else:
                    raw_fake_cc[i][k] = raw_fake_cc[i][k]/r_neg[i][j]

            # same thing for real dataset
            if raw_real_cc[i][0]:
                raw_real_cc[i][0] = raw_real_cc[i][0]/r_pos[i][j]
            else:
                raw_real_cc[i][0] = raw_real_cc[i][0]/r_neg[i][j]
        
        # Create data frames out of the matrixes and save them as csv
        norm_fake_cc = pd.DataFrame(raw_fake_cc,dtype=np.float64)
        norm_fake_cc.to_csv(os.path.join(cc_dir,'normalized-cc-fake' + '-sigma-' + str(j) + '.csv'), index = False) 

        norm_real_cc = pd.DataFrame(raw_real_cc,dtype=np.float64)  
        norm_real_cc.to_csv(os.path.join(cc_dir,'normalized-cc-real' + '-sigma-' + str(j) + '.csv'), index = False) 


def single_trial_significance(nb_line_width: int, nb_lineE : int):
    '''
    This function compute the p-value of each bin in the real dataset ((i.e. what fraction of simulated datasets 
    showed stronger correlation or anti-correlation compared with the real dataset) and derives the single trial significance
    of each bin in the real dataset. Both quantities are then saved as a single columns where the rows are lineE values
    
    Args:
        nb_line_width (int) : 
            number of sigma values tested
        nb_lineE (int) : 
            number of lineE values tested
    '''
    
    for j in range(nb_line_width):
        # load the normalized cross correlations derived from the real and simulated datasets
        norm_real_cc = pd.read_csv(os.path.join(cc_dir,'normalized-cc-real-sigma-' + str(j) + '.csv'), dtype = np.float64).to_numpy()
        norm_fake_cc = pd.read_csv(os.path.join(cc_dir,'normalized-cc-fake-sigma-' + str(j) + '.csv'), dtype = np.float64).to_numpy()
        
        nb_columns = len(norm_fake_cc[0])
        
        # compute the p-value as the fraction of simulated datasets which showed stronger correlation or anti-correlation compared with the real dataset
        p_values = []
        for i in range(nb_lineE):
            real_cc_value = norm_real_cc[i][0]
            
            # Since we ordered the normalized cross correlation values by ascending order, counting is easier
            norm_fake_cc[i] = np.sort(norm_fake_cc[i])
            count = 0
            
            if real_cc_value<0:
                col_index = 0

                while col_index<nb_columns and real_cc_value > norm_fake_cc[i][col_index]:
                    count +=1 
                    col_index +=1
                col_index = nb_columns-1
                while col_index>=0 and np.abs(real_cc_value) < norm_fake_cc[i][col_index]:
                    count+=1
                    col_index -=1
            
            if real_cc_value>0:
                col_index = nb_columns-1
                while col_index>=0 and real_cc_value < norm_fake_cc[i][col_index]:
                    count+=1
                    col_index -=1
                col_index = 0 
                while col_index<nb_columns and real_cc_value < np.abs(norm_fake_cc[i][col_index]):
                    count+=1
                    col_index +=1
                    
            p_values.append(count/nb_columns)
        sts = np.sqrt(2)*special.erfinv(1-np.array(p_values))
        
        for k in range(len(sts)):
            sts[k] = np.sign(norm_real_cc[k][0])*sts[k]
        
        # now turn our list of values to series and save them as csv
        pd.Series(p_values, name = 'p-values').to_csv(os.path.join(sig_dir,'p-values-sigma-' + str(j) + '.csv'), index = False)
        pd.Series(sts, name = 'STS').to_csv(os.path.join(sig_dir,'sts-sigma-' + str(j) + '.csv'), index = False)
        
        
def true_trial_significance(nb_line_width: int, nb_lineE : int): 
    '''
    This function compute the true p-value of each bin in the real dataset as the fraction of simulated datasets showing a 
    feature (anywhere within the searched parameter range) stronger than the real and derives the single trial significance
    of each bin in the real dataset. Both quantities are then saved as a single columns where the rows are lineE values
    
    Args:
        nb_line_width (int) : 
            number of sigma values tested
        nb_lineE (int) : 
            number of lineE values tested
    '''
    
    # load all the normalized cc values derived from the simulated datasets as a list of numpy matrixes
    df_list = []
    for j in range(nb_line_width):
        df = pd.read_csv(os.path.join(cc_dir,'normalized-cc-fake-sigma-' + str(j) + '.csv'),dtype = np.float64)
        df_list.append(df)
        
    nb_simulations = len(df_list[0].columns)
    
    all_fake_cc = df_list[0]
    for i in range(1,len(df_list)):
        np.concatenate((all_fake_cc,df_list[i]))
    
    max_cc_values = []
    min_cc_values = []
    for k in range(nb_simulations):
            # find the strongest cross correlation an anti cross correlation in the simulation over all parameter bins
            max_cc_values.append(df[str(k)].max())
            min_cc_values.append(df[str(k)].min())
    
    for j in range(nb_line_width):
        # load all the normalized cc values derived from the real datasets as a numpy matrix
        norm_real_cc = pd.read_csv(os.path.join(cc_dir,'normalized-cc-real-sigma-' + str(j) + '.csv'), dtype = np.float64).to_numpy()
        
        true_p_values = []
        for i in range(nb_lineE):
            real_cc_value = norm_real_cc[i][0]
            
            count = 0
            #loop on all the simulations:
            for k in range(nb_simulations):
                # compare it to the real_cc_value
                if real_cc_value>0 and (real_cc_value<max_cc_values[k] or real_cc_value < np.abs(min_cc_values[k])):
                    count+=1
                elif real_cc_value < 0 and (real_cc_value > min_cc_values[k] or np.abs(real_cc_value) < max_cc_values[k]):
                    count+=1
            
            true_p_values.append(count/nb_simulations)
        tts = np.sqrt(2)*special.erfinv(1-np.array(true_p_values))
        
        for k in range(len(tts)):
            tts[k] = np.sign(norm_real_cc[k][0])*tts[k]
        
        # now turn our list of values to series and save them as csv
        pd.Series(true_p_values, name = 'true-p-values').to_csv(os.path.join(sig_dir,'true-p-values-sigma-' + str(j) + '.csv'), index = False)
        pd.Series(tts, name = 'TTS').to_csv(os.path.join(sig_dir,'tts-sigma-' + str(j) + '.csv'), index = False) 
        
def clear_sig():
    """
    This function empties the output_files/significance
    """
    for f in os.listdir(sig_dir):
        os.remove(os.path.join(sig_dir, f))
    
def plot_results(path_to_burst_spectra, file_name,nb_simulations, E_range,*line_widths):
    
    os.chdir(path_to_burst_spectra)
    
    plt.style.use(['science', 'no-latex'])
    
    Xset.restore(file_name)
    #AllData.ignore('4.0-**')
    Fit.perform()
    
    #set up the plot interface
    Plot.device = '/null'
    Plot.xAxis = 'keV'
    Plot('resid')
    x = Plot.x()
    
    fig = plt.figure() # an empty figure with no Axes
    fig, ax = plt.subplots(4, sharex=True, figsize=(10,15))  # a figure with a 4x1 grid of Axes
    for axis in ax:
        axis.axhline(y=0, c = 'black', linewidth = 0.5)
    plt.subplots_adjust(hspace=0.)
    
    # Plot the real residuals with error bars
    real_resid = pd.read_csv(os.path.join(resid_dir, 'real-residuals.csv'), dtype = np.float64).to_numpy().flatten()
    ax[0].errorbar(x, real_resid, yerr =Plot.yErr(), xerr = Plot.xErr(), linestyle='')
    ax[0].set_ylabel('Data - continuum model\n$count.s^{-1}.keV^{-1}$')
    
    # Plot the single trial significance and the maximum significance given the number of simulations performed
    ax[1].set_ylabel('Single trial significance / $\sigma$')
    
    sigma_limit = np.sqrt(2)*special.erfinv(1-1/nb_simulations)
    ax[1].hlines(y=[sigma_limit,-sigma_limit], xmin =0., xmax = x[-1], color = 'red')
    
    count = 0
    for f in os.listdir(os.path.join(sig_dir)):
        if f[:3] == 'sts':
            sts = pd.read_csv(os.path.join(sig_dir, f), dtype = np.float64).to_numpy().flatten()
            
            # the value of the sts is limited by the sumber of simulations
            for s in sts:
                if s >0 and s > sigma_limit:
                    s = sigma_limit
                elif s <0 and s < -sigma_limit:
                    s = -sigma_limit
            
            ax[1].plot(E_range,sts,label='width = ' + str(line_widths[count]) + 'keV')
            count+=1
    
    # Plot the single the normalized cc value
    ax[2].set_ylabel('Renormalized cross-correlation / $\sigma$')
    
    count = 0
    for f in os.listdir(os.path.join(cc_dir)):
        if f[:15] == 'normalized-cc-r':
            norm_cc = pd.read_csv(os.path.join(cc_dir, f), dtype = np.float64).to_numpy().flatten()
            ax[2].plot(E_range,norm_cc,label='width = ' + str(line_widths[count]) + 'keV')
            count+=1
    
    # Plot the true trial significance
    ax[3].set_ylabel('True significance / $\sigma$')
    ax[3].set_xlabel('Energy (keV)')
    
    count = 0
    for f in os.listdir(os.path.join(sig_dir)):
        if f[:3] == 'tts':
            tts = pd.read_csv(os.path.join(sig_dir, f), dtype = np.float64).to_numpy().flatten()
            ax[3].plot(E_range,tts,label='width = ' + str(line_widths[count]) + 'keV')
            ax[3].grid(axis='y')
            ax[3].legend(loc='upper left')
            count+=1
    
    #plt.show()
    plt.savefig('/Users/lucas/Documents/IRAP/CrossCorrelationSearch/figures/aaa.eps',format='eps')  
    
        
                    
                    
            

                
            
            

    
    
    