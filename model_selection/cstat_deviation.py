import numpy as np
import math
import sys
from scipy.stats import norm

# From Kaastra(2017) https://ui.adsabs.harvard.edu/abs/2017A%26A...605A..51K/abstract
# Cash (1979) : https://ui.adsabs.harvard.edu/abs/1979ApJ...228..939C/abstract

def db_compute_ce_cv(mu):
    """
    This function computes the Ce and Cv factors relative to one bin using approximations of infinite series 

    Args:
        mu (int): expected number of counts in the relevant bin for the tested model
    
    Returns:
        Ce and Cv
    """

    def f0(mu, k):
        """
        This function computes the factor P_k(mu)*[mu-k+k*ln(k/mu)]

        Args:
            mu (int): expected number of counts in the relevant bin for the tested model
            k (int): index of the sum

        Returns:
            P_k(mu)*[mu-k+k*ln(k/mu)]
        """
        
        pk_mu=(np.exp(-mu)*(mu**k))/math.factorial(k)
        if k > 0 :
            pk_mu=pk_mu*(mu-k+k*np.log(k/mu))**2.
        if k == 0 :
            pk_mu=pk_mu*(mu)**2.

        return pk_mu


    ce = 0. ; cv= 0.
   
    if mu <= 0.5 : ce=-0.25*mu**3. + 1.38*mu**2. - 2.*mu*np.log(mu)
    if mu >0.5 and mu <= 2. : ce=-0.00335*mu**5 + 0.04259*mu**4. - 0.27331*mu**3. + 1.381*mu**2. - 2.*mu*np.log(mu)
    if mu >2 and mu <= 5. : ce = 1.019275 + 0.1345*mu**(0.461-0.9*np.log(mu))
    if mu > 5 and mu <= 10. : ce = 1.00624 + 0.604/mu**1.68
    if mu > 10 : ce= 1.+0.1649/mu+0.226/mu**2.


    if mu >= 0 and mu <= 0.1 : cv=4.*(f0(mu,0.)+f0(mu,1.)+f0(mu,2.)+f0(mu,3.)+f0(mu,4.))-ce**2.
    if mu > 0.1 and mu <= 0.2 : cv=-262.*mu**4. +195.*mu**3. -51.24*mu**2. + 4.34*mu + 0.77005
    if mu > 0.2 and mu <= 0.3 : cv=4.23*mu**2. - 2.8254*mu + 1.12522
    if mu > 0.3 and mu <= 0.5 : cv=-3.7*mu**3. + 7.328*mu**2 - 3.6926*mu + 1.20641
    if mu > 0.5 and mu <= 1. : cv = 1.28*mu**4. - 5.191 * mu**3 + 7.666*mu**2. - 3.5446*mu + 1.15431
    if mu > 1 and mu <= 2. : cv = 0.1125*mu**4. - 0.641 * mu**3 + 0.859*mu**2. + 1.0914*mu - 0.05748
    if mu > 2 and mu <= 3. : cv = 0.089*mu**3. - 0.872*mu**2.+ 2.8422*mu - 0.67539
    if mu > 3 and mu <= 5. : cv = 2.12336 + 0.012202*mu**(5.717-2.6*np.log(mu))
    if mu > 5 and mu <= 10. : cv = 2.05159 + 0.331*mu**(1.343-np.log(mu))
    if mu > 10 : cv=12./mu**3. + 0.79/mu**2. + 0.6747/mu + 2.

    if ce == 0. or cv == 0. : sys.exit("value of "+str(mu)+" not supported, please go back to Kaastra (2017)")

    return ce,cv

def db_compute_goodness_of_the_fit_from_cstat_v1(data,model,dof,xspec_cstat,verbose = True):
    """
    This function returns 

    Args:
        data (array): tuple of floats containing the spectrum rates for noticed channels in counts
        model (array): list of the model values in counts
        dof (int): degree of freedom of the fit
        xspec_cstat (float): Cstat computed by xspec
        
    Returns:
        The cstat deviation in unit of sigma. 
    
    """

    if verbose : print ("Total number of data_values bins=",len(data))
    if verbose : print ("degree of freedom=",dof)
    chi2bfit=0.
    cstat=0.
    ce_sum=0.
    cv_sum=0.
    nbneg=0. ; sumnegdata=0.
    truncated_xspec_value=1.
    for i in range(len(data)) :
        if data[i] <=0 :
            nbneg+=1.
            sumnegdata+=data[i]
        if model[i] < 0 : model[i]=1.0E-10
        if data[i] > 0. :  cstat+=model[i]-data[i]-data[i]*np.log(model[i])+data[i]*np.log(data[i])
        if data[i] <= 0. : cstat+=model[i]-data[i]-data[i]*np.log(model[i])+data[i]*truncated_xspec_value
        if data[i] >0 : chi2bfit+=((data[i]-model[i])**2)/data[i]
        ce,cv=db_compute_ce_cv(model[i])
        ce_sum+=ce ; cv_sum+=cv
    cstat=2.*cstat
    if verbose : print ("Own cstat =",cstat)
    if verbose : print ("Chi2 of the best fit =",chi2bfit)
    if verbose : print ("Difference between Xspec ",xspec_cstat,"and my own estimate=",cstat,"=",xspec_cstat-cstat)
    if verbose : print ("1 sigma=",100.*norm.sf(1.),(100.-2.*100.*norm.sf(1.)),"% 2 sigma=",100.*norm.sf(2.),(100.-2.*100.*norm.sf(2.)),"% 3 sigma=",100.*norm.sf(3.))
    if verbose : print ("% Probability to get cstat ",xspec_cstat," out of the expected cstat ",ce_sum,"with sigma",np.sqrt(cv_sum),"=", 100.*norm.sf(np.abs((xspec_cstat-ce_sum)/np.sqrt(cv_sum))),"% - devation =",(xspec_cstat-ce_sum)/np.sqrt(cv_sum),"sigma")

    return (xspec_cstat-ce_sum)/np.sqrt(cv_sum)