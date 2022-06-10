from xspec import *
import numpy as np
import os
from pathlib import Path

# Users should change these lines so that they matches the correct directory
output_dir = Path(
    '/Users/lucas/Documents/IRAP/CrossCorrelationSearch/output-files')


def cross_correlation_search(file_name: str, nb_simulations: int, E_min, E_max, dE, *sigmas):
    """
    This function performs a gaussian line search in an X-ray spectrum using the cross-correlation method developed by 
    Kosec et al. 2021 in this paper : https://arxiv.org/pdf/2109.14683.pdf

    Args:
        file_name (str): 
            The name of the .xcm file to use to perform the search. You should be in the directory of this file
            when running the program otherwise you have to specify the path.
        nb simulations (int): 
            number of simulations to perform 
        E_min (float): 
            Minimum energy to search for lines, in keV
        E_max (float): 
            Maximum energy to search for lines, in keV
        dE (float): 
            Energy step to search for lines, in keV
        sigmas (list):
            Line widths to be searched for
    """

    Xset.restore(file_name)
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

    generate_resid(nb_simulations)

    ####
    # GENERATE SPECTRAL MODELS
    ####

    generate_models(E_min, E_max, dE, *sigmas)

    ####
    # COMPUTE RAW CROSS CORRELATIONS AND NORMALIZATION FACTORS
    ####
    
    # All the gaussian lines will be loaded in a 3D array
    E_range = np.arange(E_min, E_max, dE)
    nb_sigma = len(sigmas)
    nb_lineE = len(E_range)
    
    gaussian_lines = np.zeros((nb_lineE, nb_sigma, nb_bins))
    
    for j in range(nb_sigma):
        file_name = 'gaussian-lines-sigma-' + str(j) + '.npz'
        with np.load(os.path.join(output_dir, file_name)) as lines:
            for i in range(nb_lineE):
                arr_name = 'arr_' + str(i)
                gaussian_lines[i][j] = lines[arr_name]
    
    # Load real residuals and cross correlate with all the gaussian lines
    real_resid = np.load(os.path.join(output_dir, 'real-residuals.npy'))
    cross_correlate(gaussian_lines, os.path.join(output_dir, 'raw-cc-real'),real_resid)
    
    # get the number of fake-resid-X.npz files (fake residuals are regrouped in 5000 simulations blocks)
    if nb_simulations % 5000 == 0:
        nb_file = nb_simulations//5000
    else:
        nb_files = nb_simulations//5000 + 1
    
    # Number of each positive and negative correlation simulations (in each parameter bin), 
    # as well as the sums of squares of all positive and negative correlations in each bin.
    nb_pos_cc = 0
    nb_neg_cc = 0
    sum_pos_cc = 0
    sum_neg_cc = 0
    
    r_pos = np.zeros((nb_lineE,nb_sigma))
    r_neg = np.zeros((nb_lineE,nb_sigma))
    
    # Compute raw cross correlation between fake residuals and all the gaussian lines
    residuals_sets = []
    for n in range(nb_files):
        with np.load(os.path.join(output_dir, 'fake-resid-' + str(n) + '.npz')) as fake_resid:
            for resid in fake_resid:
                residuals_sets.append(fake_resid[resid])
    cross_correlate(gaussian_lines, os.path.join(output_dir, 'raw-cc-fake'), *residuals_sets)
    
    ####
    # NORMALIZATION OF THE CROSS CORRELATIONS
    ####
    
    # Compute the normalization factors
    r_pos = np.zeros((nb_lineE,nb_sigma))
    r_neg = np.zeros((nb_lineE,nb_sigma))
    
    # loop on the files
    for j in range(nb_sigma):
        with np.load(os.path.join(output_dir, 'raw-cc-fake-sigma-' + str(j) + '.npz')) as raw_cc_fake :
            # loop on the arrays in the file
            for i in range(nb_lineE):
                n_pos = 0
                n_neg = 0
                sum_pos = 0
                sum_neg = 0
                
                array_name = 'arr_' + str(i)
                cc_array = raw_cc_fake[array_name]
                # loop on the cross correlation values of the array (each value is the cross correlation between one set of simulated residuals and a gaussian line)
                for c in cc_array:
                    if c>0:
                        n_pos += 1
                        sum_pos += c**2
                    else: 
                        n_neg += 1
                        sum_neg += c**2
                r_pos[i][j] = np.sqrt((1/n_pos)*sum_pos)
                r_neg[i][j] = np.sqrt((1/n_neg)*sum_neg)
    
    # normalize raw cross correlation of the real residuals and save them
    normalize(r_pos,r_neg)
    
    ####
    # SINGLE TRIAL SIGNIFICANCE
    ####
    
    # p-values are stored in a matrix where the columns are width bins and the lines are energy bins
    p_values = np.zeros((nb_lineE,nb_sigma))
    
    # loop on the width bins
    for j in range(nb_sigma):
        # load real residuals normalized cross correlations values 
        with np.load(os.path.join(output_dir, 'normalized-cc-real-sigma-' + str(j) + '.npz')) as norm_cc_real:
            with np.load(os.path.join(output_dir, 'normalized-cc-fake-sigma-' + str(j) + '.npz')) as norm_cc_fake:
                # loop on the energy bins
                for i in range(nb_lineE):
                    array_name = 'arr_' + str(i)
                    # real cross correlation value for the parameter bin (lineE[i],sigmas[j])
                    cc_value = norm_cc_real[array_name]
                
                    # list of correlation values for all simulated dataset for the parameter bin (lineE[i],sigmas[j])
                    fake_cc_values = norm_cc_fake[arr_name]
                    
                    # p_values[i][j] is then computed as the fraction of simulated datasets showing stronger correlation or anti-correlation compared with the real dataset
                    count = 0.
                    imini = 0
                    imaxi = len(fake_cc_values)-1
                    
                    while imaxi>=0 and fake_cc_values[imaxi]>cc_value:
                        count+=1.
                        imaxi -=1
                    
                    while imini>=len(fake_cc_values)-1 and -cc_value>fake_cc_values[imini]:
                        count+=1.
                        imini+=1
                    
                    p_values[i][j] = count/nb_simulations
    np.save(os.path.join(output_dir, 'p-values' + '.npy'),p_values)
                    
                        
                
                
    
    
                
        
                
                
    

                        
                
                        
    
                
            
    

    
    
    


def generate_resid(n: int):
    """
    This function generate both real and simulated residual spectra. 

    The simulated residuals are stored in files by large blocks, for example by storing 5000 individual 
    simulations in a single file. This grouping results in a large  npz file where one array corresponds to an 
    individual simulation and each term in the array to a wavelength bins.

    Args:
        n (int): number of simulations to perform
    """
    # start by cleaning the folder
    clear_output_files_resid()

    # set up the plot
    Plot.device = '/null'
    Plot.xAxis = 'keV'

    # generate real residual spectrum and save it as an .npy file
    Plot('resid')

    real_resid = np.array(Plot.y())
    np.save(os.path.join(output_dir, 'real-residuals.npy'), real_resid)

    # generate n fake residual spectra and grouping them in file blocks of 5000 spectra

    # list to store all fake residual spectra
    all_fake_resid = []

    # get noticed channels in the real spectrum
    noticed_ch = AllData(1).noticedString()

    for k in range(n):
        # create FakeitSettings object so that the file names are consistant
        name = 'fakeSpectum-' + str(k)
        fs = FakeitSettings(fileName=name)
        AllData.fakeit(1, [fs])

        # make sure that the same channels are noticed in real and fake spectra
        AllData.ignore('1-**')
        AllData.notice(noticed_ch)

        Fit.perform()

        Plot('resid')

        # get the values of the fake residuals as a column
        fake_resid = np.array(Plot.y())

        # append the list of fake residuals to the file
        all_fake_resid.append(fake_resid)

    # group the arrays in file blocks of 5000 spectra
    count = 0
    head = 0
    tail = 5000
    if n <= 5000:
        np.savez(os.path.join(output_dir, 'fake-resid-' +
                 str(count) + '.npz'), *all_fake_resid)
    else:
        # groups of 5000
        for k in range(n//5000):
            np.savez(os.path.join(output_dir, 'fake-resid-' +
                     str(count) + '.npz'), *all_fake_resid[head:tail])
            head = tail
            tail += 5000
            count += 1

        # last group might contain less than 5000 spectra
        np.savez(os.path.join(output_dir, 'fake-resid-' +
                 str(count) + '.npz'), *all_fake_resid[head:])


def clear_output_files_resid():
    """
    This function deletes every file containing residuals in the output-files directory
    """
    for f in os.listdir(output_dir):
        if f[:4] == 'fake' or f[:4] == 'real':
            os.remove(os.path.join(output_dir, f))


def generate_models(E_min: float, E_max: float, dE: float, *sigmas):
    """
    This function generates gaussian lines with parameters varying according to a grid. 

    Args:
        E_min (float): 
            Minimum energy to search for lines, in keV
        E_max (float): 
            Maximum energy to search for lines, in keV
        dE (float): 
            Energy step to search for lines, in keV (should slightly over sample the spectral resolution)
        sigmas (list):
            Line widths to be searched for
    """
    # start by cleaning the folder
    clear_output_files_models()

    # set up the plot interface
    Plot.device = '/null'
    Plot.xAxis = 'keV'
    Plot('data')

    # create a gaussian line to vary the parameter
    m = Model('gaussian')

    # set energy range
    E_range = np.arange(E_min, E_max, dE)

    # spectral models stored in len(sigmas) individual npz files
    count = 0
    for sig in sigmas:
        # set width of the gaussian line (remains constant in a file)
        m.setPars({2: sig})

        # list to contain all models from a file
        all_sigma_lines = []

        for e in E_range:
            # change th lineE parameter of the gaussian line
            m.setPars(e)

            Plot('data')
            # fill a an line array with the model values
            model_values = np.array(Plot.model())

            all_sigma_lines.append(model_values)

        np.savez(os.path.join(output_dir, 'gaussian-lines-sigma-' +
                 str(count) + '.npz'), *all_sigma_lines)
        count += 1


def clear_output_files_models():
    """
    This function deletes every file containing gaussian lines in the output-files directory
    """
    for f in os.listdir(output_dir):
        if f[:4] == 'gaus':
            os.remove(os.path.join(output_dir, f))


def cross_correlate(gaussian_lines: np.array, fileName: str, *residuals_sets):
    """
    This function performs the raw cross correlation between a grid of gaussian lines and sets of residuals 
    and saves the results as an fileName.npz file. Each file corresponds to a value of sigma and contains as many arrays 
    as there are values of lineE. Each array contains as many terms as there are residuals sets. The result is similar to a large
    table where sigma is the same, each line is a value of lineE and each column a dataset. 

    Args:
        residuals_sets (np.array): 
            Arrays containing residuals values from a spectrum
        gaussian_lines (np.array): 
            3D array representing a grid of gaussian lines
        fileName (str): 
            Name of the file where to save the results of the cross correlation
    """
    
    n_sigma = len(gaussian_lines[0])
    n_lineE = len(gaussian_lines)
    n_columns = len(residuals_sets)
    
    for j in range(n_sigma):
        # list to store all the cross correlation arrays of a given sigma
        raw_cc_all = []
        for i in range(n_lineE):
            raw_cc_values = np.zeros(n_columns)
            for k in range(n_columns):
                raw_cc_values[k] = np.correlate(residuals_sets[k], gaussian_lines[i][j])[0]
            raw_cc_all.append(raw_cc_values)
        np.savez(os.path.join(output_dir, fileName + '-sigma-' + str(j) + '.npz'), *raw_cc_all)


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
    nb_sigma = len(r_pos[0])
    nb_lineE = len(r_pos)
    
    for j in range(nb_sigma):
        normalized_cc_real = []
        normalized_cc_fake = []
        
        raw_cc_real = np.load(os.path.join(output_dir, 'raw-cc-real-sigma-' + str(j) + '.npz'))
        raw_cc_fake = np.load(os.path.join(output_dir, 'raw-cc-fake-sigma-' + str(j) + '.npz'))
        
        for i in range(nb_lineE):
            array_name = 'arr_' + str(i)
            real_cc_array = raw_cc_real[array_name]
            fake_cc_array = raw_cc_fake[array_name]
            
            for c in real_cc_array:
                if c>0:
                    c = c/r_pos[i][j]
                else:
                    c = c/r_neg[i][j]
            normalized_cc_real.append(real_cc_array)
            
            for c in fake_cc_array:
                if c>0:
                    c = c/r_pos[i][j]
                else:
                    c = c/r_neg[i][j]
            # fake residuals cross correlations are sorted in ascending order within their energy/width bins
            normalized_cc_fake.append(np.sort(fake_cc_array))
            
        np.savez(os.path.join(output_dir, 'normalized-cc-real' + '-sigma-' + str(j) + '.npz'), *normalized_cc_real)
        np.savez(os.path.join(output_dir, 'normalized-cc-fake' + '-sigma-' + str(j) + '.npz'), *normalized_cc_fake)

    
# TEST
sigmas = np.array([0.025, 0.05, 0.075])
dE=0.08
E_min = 0.4
E_max = 6.0
nb_simulations = 1000
cross_correlation_search('f006_TBabs_diskbb_powerlaw_famodel.xcm',nb_simulations,E_min,E_max,dE,*sigmas)