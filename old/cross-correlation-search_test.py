from re import L
from xspec import *
import numpy as np
import os
from pathlib import Path

# Users should change this line so that head matches the correct directory
HEAD = Path('/Users/lucas/Documents/IRAP/CrossCorrelationSearch/output-files')



Xset.restore('f006_TBabs_diskbb_powerlaw_famodel.xcm')
Fit.perform()

def generate_resid(n: int):
    """
    This function generate both real and simulated residual spectra. 
    
    The simulated residuals are stored in files by large blocks, for example by storing 5000 individual 
    simulations in a single file. This grouping results in a large  npz file where one array corresponds to an 
    individual simulation and each term in the array to a wavelength bins.

    Args:
        n (int): number of simulations to perform
    """
    # set up the plot
    Plot.device = '/null'
    Plot.xAxis = 'keV'
    
    # generate real residual spectrum and save it as an .npy file
    Plot('resid')
    
    real_resid  = np.array(Plot.y())
    np.save(os.path.join(HEAD, 'real-residuals.npy'), real_resid)
    
    # generate n fake residual spectra and grouping them in file blocks of 5000 spectra
    
    # list to store all fake residual spectra
    all_fake_resid = []
    
    # get noticed channels in the real spectrum
    noticed_ch = AllData(1).noticedString()
    
    
    for k in range(n):
        # create FakeitSettings object so that the file names are consistant
        name = 'fakeSpectum-' + str(k)
        fs = FakeitSettings(fileName = name)
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
    count = 1
    head = 0
    tail = 5000 
    if n <= 5000:
         np.savez(os.path.join(HEAD, 'fake-resid-' + str(count) + '.npz'), *all_fake_resid)
    else:
        # groups of 5000
        for k in range (n//5000):
            np.savez(os.path.join(HEAD, 'fake-resid-' + str(count) + '.npz'), *all_fake_resid[head:tail])
            head = tail
            tail +=5000
            count+=1
        
        # last group might contain less than 5000 spectra
        np.savez(os.path.join(HEAD, 'fake-resid-' + str(count) + '.npz'), *all_fake_resid[head:])
        

def clear_output_files_resid():
    """
    This function deletes every file containing residuals in the output-files directory
    """
    dir = HEAD
    for f in os.listdir(dir):
        if f[:4] == 'fake' or f[:4] == 'real':
            os.remove(os.path.join(dir, f))


#Xset.restore('f006_TBabs_diskbb_powerlaw_famodel.xcm')
#Fit.perform()

def generate_models(dE, *sigmas):
    """
    This function generates gaussian lines with parameters varying according to a grid. 

    Args:
        dE (float): energy precision for the search in keV dE should slightly over sample the spectral resolution
        sigmas corresponds to a list of sigma (related to FWHM ) to be searched over
    """
    
    #set up the plot interface
    Plot.device = '/null'
    Plot.xAxis = 'keV'
    Plot('data')
    
    # create a gaussian line to vary the parameter
    m = Model('gaussian')
    
    # set energy range
    E_min = Plot.x()[0]
    E_max = Plot.x()[-1]
    E_range = np.arange(E_min, E_max, dE)
    
    # spectral models stored in len(sigmas) individual npz files
    count = 0
    for sig in sigmas:
        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
        # set width of the gaussian line (remains constant in a file)
        m.setPars({2:sig})
        
        # list to contain all models from a file
        all_sigma_lines = []
        
        for e in E_range:
            # change th lineE parameter of the gaussian line
            m.setPars(e)
            
            Plot('data')
            # fill a an line array with the model values
            model_values = np.array(Plot.model())
            
            all_sigma_lines.append(model_values)
        
        np.savez(os.path.join(HEAD, 'gaussian-lines-sigma-' + str(count) + '.npz'), *all_sigma_lines)
        count += 1
        

def clear_output_files_sigma():
    """
    This function deletes every file containing gaussian lines in the output-files directory
    """
    dir = HEAD
    for f in os.listdir(dir):
        if f[:4] == 'gaus':
            os.remove(os.path.join(dir, f))


def raw_cross_correlations():
    """
    This function computes all the raw cross correlations between both real and simulated residuals as well as spectral models
    and saves in npz files
    """
    #### LOAD ALL THE GAUSSIAN LINES IN A 3D ARRAY ####
    
    # count the number of sigma values (columns of the 3D array)
    dir = HEAD
    n_sigma = 0
    for f in os.listdir(dir):
        if f[:4]=='gaus':
            n_sigma+=1
    
    # count the number of lineE values (rows of the 3D array)
    f = np.load(os.path.join(dir,"gaussian-lines-sigma-0.npz"))
    n_lineE = len(f.files)
    nb_bins = len(f['arr_0'])
    
    # create the 3D array and fill it up (each column contains all arrays in a 'gaussian-lines-sigma-X.npz file )
    gaussian_lines = np.zeros((n_lineE,n_sigma,nb_bins))
    
    for j in range(n_sigma):
        file_name = 'gaussian-lines-sigma-' + str(j) + '.npz'
        npzfile = np.load(os.path.join(HEAD, file_name))
        
        for i in range(n_lineE):
            arr_name = 'arr_' + str(i)
            gaussian_lines[i][j] = npzfile[arr_name]
                  
    #### RAW CROSS CORRELATION BETWEEN REAL RESIDUALS AND MODELS ####
    
    # load data and create a matrix to hold cross correlation values
    real_resid = np.load(os.path.join(dir, 'real-residuals.npy'))
    Cc_real = np.zeros((n_lineE, n_sigma))
    
    # fill the matrix and save it 
    for i in range(n_lineE):
        for j in range(n_sigma):
            Cc_real[i][j] = np.correlate(real_resid, gaussian_lines[i][j])[0]
    np.save(os.path.join(HEAD, 'raw-cc-real.npy'), Cc_real)
            
    #### RAW CROSS CORRELATION BETWEEN FAKE RESIDUALS AND MODELS ####
    
    # get the number of fake-resid-X.npz files
    nb_files = 0
    for f in os.listdir(dir):
        if f[:4] == "fake":
            nb_files +=1 
    
    # list to store all the cross correlation matrixes
    Cc_fake_all = []
    
    # compute all the raw cross correlation of fake residuals and model
    for n in range(1,nb_files+1):
        with np.load(os.path.join(dir, 'fake-resid-' + str(n) + '.npz')) as fake_resid:
            for arr in fake_resid:
                Cc_fake = np.zeros((n_lineE, n_sigma))
                for i in range(n_lineE):
                    for j in range(n_sigma):
                        Cc_fake[i][j] = np.correlate(fake_resid[arr], gaussian_lines[i][j])[0]
                Cc_fake_all.append(Cc_fake)
                
    # group the matrixes in file blocks of 5000 and save them
    head = 0
    tail = 5000
    count = 1
    
    if nb_files==1:
        np.savez(os.path.join(HEAD, 'raw-cc-fake-' + str(count) + '.npz'), *Cc_fake_all)
    else :
        # first nb-1 files all contain 5000 matrixes
        for k in range(nb_files-1):
            np.savez(os.path.join(HEAD, 'raw-cc-fake-' + str(count) + '.npz'), *Cc_fake_all[head:tail])
            head = tail
            tail +=5000
            count += 1
        # last file might contain less
        np.savez(os.path.join(HEAD, 'raw-cc-fake-' + str(count) + '.npz'), *Cc_fake_all[head:])
        
        
def clear_output_files_sigma():
    """
    This function deletes every file containing raw cross correlations in the output-files directory
    """
    dir = HEAD
    for f in os.listdir(dir):
        if f[:3] == 'raw':
            os.remove(os.path.join(dir, f))
    

def normalize():
    """
    This function first compute the normalization factors for each energy and width bin and then normalizes all the raw cross-
    correlations computed previously
    """
    
    # COMPUTE NORMALIZATION FACTORS
    
    # Get the cross correlations files of the simulated datasets in the output-files directory
    dir = HEAD
    files = []
    for f in os.listdir(dir):
        if f[:11] == 'raw-cc-fake':
            files.append(f)
    
    # matrixes to store the normalization factor
    nb_lines = len(E_range)
    nb_columns = len(sigmas)
    r_pos = np.zeros((nb_lines,nb_columns))
    r_neg = np.zeros((nb_lines,nb_columns))
    
    for i in E_range:
        for j in sigmas:
            n_pos = 0
            n_neg = 0
            sum_pos = 0
            sum_neg = 0
            for f in files:
                with np.load(os.path.join(dir, f)) as cc_fake :
                    for arr in cc_fake:
                        cc_values = cc_fake[arr]
                        if cc_values[i][j]>0:
                            n_pos +=1
                            sum_pos += (cc_values[i][j])**2
                        else:
                            n_neg +=1
                            sum_neg += (cc_values[i][j])**2
            r_pos[i][j] = np.sqrt((1/n_pos)*sum_pos)
            r_neg[i][j] = np.sqrt((1/n_neg)*sum_neg)
            
    # NORMALIZE RAW CROSS CORRELATIONS OF THE REAL DATASET
    
    # load the raw cross correlation into memory
    with np.load(os.path.join(dir, 'raw-cc-real.npy')) as cc_real :
        for i in range(nb_lines):
            for j in range(nb_columns):
                if cc_real[i][j]>0:
                    cc_real[i][j] = cc_real[i][j]/r_pos[i][j]
                else:
                    cc_real[i][j] = cc_real[i][j]/r_neg[i][j]
        np.save(os.path.join(HEAD, 'normalized-cc-real.npy'), cc_real)
    
    # NORMALIZE RAW CROSS CORRELATIONS OF THE REAL DATASET
    for f in files:
        with np.load(os.path.join(dir, f)) as cc_fake :
            for arr in cc_fake:
                cc_values = cc_fake[arr]
                for i in range(nb_lines):
                    for j in range(nb_columns):
                        if cc_values[i][j]>0:
                            cc_values[i][j] = cc_values[i][j]/r_pos[i][j]
                        else:
                            cc_values[i][j] = cc_values[i][j]/r_neg[i][j]
                
                
        
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
                
    
    
    
#clear_output_files_resid()
#generate_resid(1000)

#npzfile = np.load(os.path.join(HEAD,'fake-resid-1.npz'))
#print(npzfile.files)
#print(npzfile['arr_3'])

# NICER HAS A SPECTRAL RESOLUTION OF 85 eV at 1 keV and 137 eV at 6 keV so we choose a precision of 80 eV
#dE=0.08
#sigmas = np.array([0.025, 0.05, 0.075])
#generate_models(dE,*sigmas)
#npzfile = np.load(os.path.join(HEAD,'gaussian-lines-sigma-1.npz'))
#print(npzfile.files)
#print(npzfile['arr_0'])

# set energy range
#Plot.device = '/null'
#Plot.xAxis = 'keV'
#Plot('data')
#E_min = Plot.x()[0]
#E_max = Plot.x()[-1]
#E_range = np.arange(E_min, E_max, dE)

#raw_cross_correlations()

