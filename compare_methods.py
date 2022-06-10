from cross_correlation_search_csv import * 
from classic_search import * 
import shutil

compare_dir = '/Users/lucas/Documents/IRAP/CrossCorrelationSearch/compare_methods'

def compare_methods(path_to_results : str,path_to_burst_spectra : str, xcm_file : str, nb_spectra : int, plot :bool, nb_simulations : int, E_min : float, E_max : float, dE : float ,*line_widths):
    '''
    This function verifies that the cross correlation methods yields the same results as the classic search method. More 
    precisely, the function plots RC for each gaussian parameter bin against sqrt(delta_C-stat). The result should be close 
    to y=x.
    
    Args:
        path_to_results (str):
            string representing the path to where you want the csv files to be saved
        path_to_bust_spectra (str):
            string representing the path to the burst spectra
        xcm_file (str) :
            xcm file containing the data and the model on which the controlled sample will be based
        nb_spectra (int):
            number of fake spectra to generate in the controlled sample
        plot (bool):
            should the function plot the graph ?
        nb simulations (int): 
            number of simulations to perform 
        E_min (float): 
            Minimum energy to search for lines, in keV
        E_max (float): 
            Maximum energy to search for lines, in keV
        dE (float): 
            Energy step to search for lines, in keV
        line_widths (float):
            Line widths to be searched for
    '''
    # start by cleaning the folder
    #clear_compare_dir()
    
    # Create nb xcm files containing the fake data and the model. Controlled sample will be generated based on xcm_file
    Xset.allowPrompting = False
    for k in range(nb_spectra):
        os.chdir(path_to_burst_spectra)
    
        # Restore the file of reference
        Xset.restore(xcm_file)
        Fit.perform()
    
        # get noticed channels in the reference spectrum
        noticed_ch = AllData(1).noticedString()
    
        # create FakeitSettings object so that the file names are consistent
        name = 'CompareMethods-FakeSpectum-' + str(k+10) + '.fak'
        fs = FakeitSettings(fileName=name)
        AllData.fakeit(1, [fs])
    
        # make sure that the same channels are noticed in real and fake spectra
        AllData.ignore('1-**')
        AllData.notice(noticed_ch)

        # Save the spectrum as a xcm file
        Xset.save('CompareMethods-FakeSpectrum-' + str(k+10) + '.xcm')
    

    for k in range (nb_spectra):
        file_name = 'CompareMethods-FakeSpectrum-' + str(k+10) + '.xcm'
    
        # Perform the cross correlation search on the file of the controlled sample
        cross_correlation_search(path_to_burst_spectra,file_name,nb_simulations,True,E_min,E_max,dE,*line_widths)
    
        # Save of copy of the file containing the normalized cc in the compare_methods_dir
        original = os.path.join(cc_dir,'normalized-cc-real-sigma-0.csv')
        target = os.path.join(path_to_results,'ccSearch-fakeSpectrum-' + str(k+10) + '.csv')
    
        shutil.copyfile(original,target)
    
        # Perform the cross classic search on the file of the controlled sample
        db_scan_residuals_for_lines(path_to_burst_spectra,file_name,True,line_widths[0],E_min,E_max,dE,False)
    
        # Move the result file to the compare_methods_dir and rename it
        original = os.path.join(classic_dir,'CompareMethods-FakeSpectrum-' + str(k+10) + '_LineSearch.csv')
        target = os.path.join(path_to_results,'classicSearch-fakeSpectrum-' + str(k+10) + '.csv')
    
        shutil.move(original,target)
    
    if plot : 
        compare_methods_plot(path_to_results, nb_spectra)
        

def compare_methods_plot(path_to_results : str, nb_spectra : int):
    
    os.chdir(path_to_results)
    plt.style.use(['science', 'no-latex'])
    
    fig, ax = plt.subplots()

    for k in range(nb_spectra):

        # Read and load the file containing the values of sqrt(delta_Cstat)
        classic_search_results = pd.read_csv(os.path.join(path_to_results,'classicSearch-fakeSpectrum-' + str(k) +'.csv'),dtype=np.float64).to_numpy()
        sqrt_delta_cstat = classic_search_results[:,6]

        # Read and load the file containing the values of the normalized cc
        cc_search_results = pd.read_csv(os.path.join(path_to_results,'ccSearch-fakeSpectrum-' + str(k) +'.csv'),dtype=np.float64).to_numpy()
        normalized_cc = cc_search_results.flatten()

        # Plot everything on the graph
        ax.scatter(sqrt_delta_cstat,normalized_cc, s = 7, color = 'red',marker = 'D',edgecolors='black',linewidths=0.5)
    
    # Plot y=x
    lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    # now plot both limits against each other
    ax.plot(lims, lims, 'k-', zorder=0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
        
    fig.savefig(os.path.join(path_to_results,'CompareMethods-sampleSize-' + str(nb_spectra) + '.eps'),format = 'eps')

def clear_compare_dir():
    for f in os.listdir(compare_dir):
        if 'fakeSpectrum' in f:
            os.remove(os.path.join(compare_dir,f))