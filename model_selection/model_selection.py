from numpy import linspace
from xspec import *
from cstat_deviation import *
import matplotlib.pyplot as plt
import numpy as np

def compute_deviation(file_name): 
    """
    This function computes the cstat deviation from an xcm file

    Args:
        file_name (.xcm): xcm file
    
    Returns: 
        the cstat deviation for the given xcm file
    """
    Xset.restore(file_name)
    Fit.perform() 
    xspec_cstat = Fit.statistic
    dof = Fit.dof

    # data values should be in counts and not counts/s
    t_e = AllData(1).exposure
    data_values = []
    for i in range (len(AllData(1).values)):
        # Alldata(1).values returns a tuple and I want a list
        data_values.append(AllData(1).values[i]*t_e)
    
    # model values should be in counts and limited to noticed bins
    Plot.device = '/null'
    Plot.xAxis = 'channel'
    Plot('counts')
    model_values = Plot.model()

    return db_compute_goodness_of_the_fit_from_cstat_v1(data_values,model_values,dof,xspec_cstat, verbose=False)
    
    
def select_models(file_names_list, deviation_min): 
    """
    This function selects the models among a list that have a deviation > deviation_min

    Args:
        file_names_list (list): array of xcm file names to be loaded 
        deviation_min (float): minimum deviation in unit of sigma
    
    Returns:
        An list of xcm file names where deviation > deviation_min
    """
    
    selected_models = []
    for file in file_names_list: 
        cstat_dev = compute_deviation(file)
        
        if cstat_dev >= deviation_min: 
            selected_models.append(file)
            
    return selected_models

        
def select_models_plot(file_names_list, deviation_min):
    """_summary_

    Args:
        file_names_list (_type_): _description_
        deviation_min (_type_): _description_
    """
    
    # list of deviation and file numbers
    cstat_deviations = []
    file_numbers = []
    for file in file_names_list:
        cstat_deviations.append(compute_deviation(file))
        file_numbers.append(int(file[1:4]))
        
    dev_min = [deviation_min]*len(file_names_list)
        
    plt.style.use(['science','no-latex'])
    
    plt.xlabel('model ID')
    plt.ylabel('deviation / sigma')
    
    plt.plot(file_numbers, cstat_deviations, 'ro')
    plt.plot(file_numbers, dev_min, '-b')
    plt.show()
        
        
        
        
        
    
    

