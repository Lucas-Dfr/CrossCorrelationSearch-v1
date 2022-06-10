#import db_spectral_analysis_library as dsal
from optparse import OptionParser
import os
import sys
import astropy.io.fits as pyfits
from optparse import OptionParser
from datetime import datetime
from astropy.io import fits
import numpy as np
import pandas as pd
import xspec
from xspec import AllData,AllModels,Xset,Spectrum,Fit,Plot,Model
import glob
#import db_tools as dt
#import db_datagraph_tools as ddt
#import db_yml_tools as dyt
from distutils.util import strtobool
#import db_manipulate_xspec_models as dmxm
import optparse
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
    
def db_scan_residuals_for_line_and_edges(path_to_burst_spectra,xcmfile, scan_line=True, emin_scan_line=0.3,emax_scan_line=3.0, step_scan_line=0.1, scan_edge=True, emin_scan_edge=0.3,emax_scan_edge=3.0, step_scan_edge=0.1):
    os.chdir(path_to_burst_spectra)

    Fit.query="no"
    if scan_line :
        minEnergy=emin_scan_line
        maxEnergy=emax_scan_line
        dE=step_scan_line
        lineE_list = np.arange(minEnergy,maxEnergy+dE,dE)

        minNorm = 0.
        maxNorm=1.0
        n=20
        dNorm=(maxNorm-minNorm)/np.float(n+1)
        Norm_tested=np.linspace(maxNorm,minNorm,n+1)
        
        Xset.restore(xcmfile)
        Fit.perform()

        cstat_bf=Fit.statistic
        AllModels.show()
        cstat = []
        input("Enter")
        Plot.device = "/xs"
        str_tp=""
        
        for energy in lineE_list:
            Xset.restore(xcmfile)
            Fit.perform()
            AllModels(1).gaussian.Sigma.frozen=True
            AllModels(1).gaussian.LineE.frozen=True
            AllModels(1).gaussian.norm.frozen=False
            AllModels(1).gaussian.LineE.values = (energy,-1)
            AllModels(1).gaussian.Sigma.values = (0.05)
            AllModels(1).gaussian.norm.values = (0,0.001,-100,-10,10,100)
            AllModels.show()
            
            Fit.steppar(f"{AllModels(1).gaussian.norm.index} {maxNorm} {minNorm} {n}")
            cstat.append(Fit.stepparResults("statistic"))
            cstat_array_int=Fit.stepparResults("statistic")
            print(cstat_array_int,"cstat-16",cstat_bf-16)
            
            if np.min(cstat_array_int) < cstat_bf - 9. :
                Plot.xAxis = "KeV"
                Plot("data residuals")

                print("Minimum reached at Energy ",energy," norm = ",Norm_tested[np.argmin(cstat_array_int)], " Delta-Cstat=",np.min(cstat_array_int) - cstat_bf)
                AllModels(1).gaussian.LineE.frozen=True
                AllModels(1).gaussian.Sigma.frozen=True
                AllModels(1).gaussian.norm.frozen=False
                AllModels(1).gaussian.norm.values = (Norm_tested[np.argmin(cstat_array_int)])
                Fit.perform()
                
                print("Leaving all parameters free : ", AllModels(1).gaussian.LineE.values[0], AllModels(1).gaussian.Sigma.values[0], AllModels(1).gaussian.norm.values[0], "cstat=",Fit.statistic,"Previous statistic=",cstat_bf)
                str_tp+=",".join(str(tp) for tp in [AllModels(1).gaussian.LineE.values[0], AllModels(1).gaussian.Sigma.values[0], AllModels(1).gaussian.norm.values[0], Norm_tested[np.argmin(cstat_array_int)], Fit.statistic, np.min(cstat_array_int),Fit.statistic-cstat_bf,cstat_bf])+"\n"
#                input("Enter")
                Plot("data residuals")
#                input("Enter")

        print(str_tp)
        if len(str_tp) > 0 :
            f = open(xcmfile.replace(".xcm","")+"_Emission_LineSearch_plot.bfit", 'a')
            f.write(str_tp)
            f.close()

#        input("Enter")
        c = cstat_bf-np.array(cstat)
        c[c<0] = 0
        data_file = xcmfile.replace(".xcm","")+"_Emission_LineSearch_plot.npy"
        x,y = np.meshgrid(lineE_list,np.linspace(maxNorm,minNorm,n+1))
        np.save(data_file,np.array([x,y,c],dtype=object))
        x,y,c = np.load(data_file,allow_pickle=True)

        fig,ax = plt.subplots(1,1,figsize=(12,6))
        contour = ax.contourf(x,y,c.T,cmap="Blues")
        plt.xlabel("Energy (keV)")
        plt.ylabel("Norm")
        cbar = plt.colorbar(contour)
        cbar.set_label(r'$\Delta$ '+Fit.statMethod)
        plt.hlines(0,minEnergy,maxEnergy,color="k",linestyle="--",alpha=0.5)
        ax.set_xlim(minEnergy,maxEnergy)
        fig.tight_layout()
        fig.savefig(xcmfile.replace(".xcm","")+"_Emission_LineSearch_plot.pdf")

# Do the same for the absorption line

        str_tp=""
        minNorm = -1.0 ; maxNorm=0.0 ; n=20
        dNorm=(maxNorm-minNorm)/np.float(n+1)
        Norm_tested=np.linspace(maxNorm,minNorm,n+1)
        Xset.restore(xcmfile)
        Fit.perform()

        cstat_bf=Fit.statistic
        AllModels.show()
        cstat = []

        AllModels(1).gaussian.norm.frozen=False
        AllModels(1).gaussian.norm.values = (0,0.001,-100,-10,10,100)
                    
        for energy in lineE_list:
            Xset.restore(xcmfile)
            Fit.perform()
            AllModels(1).gaussian.Sigma.frozen=True
            AllModels(1).gaussian.LineE.frozen=True
            AllModels(1).gaussian.norm.frozen=False
            AllModels(1).gaussian.LineE.values = (energy,-1)
            AllModels(1).gaussian.Sigma.values = (0.05)
            AllModels(1).gaussian.norm.values = (0,0.001,-100,-10,10,100)
            AllModels.show()
            Fit.steppar(f"{AllModels(1).gaussian.norm.index} {maxNorm} {minNorm} {n}")
            cstat.append(Fit.stepparResults("statistic"))
            cstat_array_int=Fit.stepparResults("statistic")
            print(cstat_array_int,"cstat-16",cstat_bf-16)
            if np.min(cstat_array_int) < cstat_bf - 9. :
                Plot.xAxis = "KeV"
                Plot("data residuals")
                print("Minimum reached at Energy ",energy," norm = ",Norm_tested[np.argmin(cstat_array_int)], " Delta-Cstat=",np.min(cstat_array_int) - cstat_bf)
                AllModels(1).gaussian.LineE.frozen=True
                AllModels(1).gaussian.Sigma.frozen=True
                AllModels(1).gaussian.norm.frozen=False
                AllModels(1).gaussian.norm.values = (Norm_tested[np.argmin(cstat_array_int)])
                Fit.perform()
                
                print("Leaving all parameters free : ", AllModels(1).gaussian.LineE.values[0], AllModels(1).gaussian.Sigma.values[0], AllModels(1).gaussian.norm.values[0], "cstat=",Fit.statistic,"Previous statistic=",cstat_bf)
                str_tp+=",".join(str(tp) for tp in [AllModels(1).gaussian.LineE.values[0], AllModels(1).gaussian.Sigma.values[0], AllModels(1).gaussian.norm.values[0], Norm_tested[np.argmin(cstat_array_int)], Fit.statistic,np.min(cstat_array_int), Fit.statistic-cstat_bf,cstat_bf])+"\n"
#                input("Enter")
                Plot("data residuals")
        print("Your results")
        print(str_tp)
#        input("Enter")

        if len(str_tp) > 0 :
            f = open(xcmfile.replace(".xcm","")+"_Absorption_LineSearch_plot.bfit", 'a')
            f.write(str_tp)
            f.close()
            
#        input("Enter")

        c = cstat_bf-np.array(cstat)
        c[c<0] = 0
        data_file = xcmfile.replace(".xcm","")+"_Absorption_LineSearch_plot.npy"

        x,y = np.meshgrid(lineE_list,np.linspace(maxNorm,minNorm,n+1))
        np.save(data_file,np.array([x,y,c],dtype=object))
        x,y,c = np.load(data_file,allow_pickle=True)

        fig,ax = plt.subplots(1,1,figsize=(12,6))
        contour = ax.contourf(x,y,c.T,cmap="Blues")
        plt.xlabel("Energy (keV)")
        plt.ylabel("Norm")
        cbar = plt.colorbar(contour)
        cbar.set_label(r'$\Delta$ '+Fit.statMethod)
        plt.hlines(0,minEnergy,maxEnergy,color="k",linestyle="--",alpha=0.5)
        ax.set_xlim(minEnergy,maxEnergy)
        fig.tight_layout()
        fig.savefig(xcmfile.replace(".xcm","")+"_Absorption_LineSearch_plot.pdf")


    if scan_edge :
        # Do the same for the edge
        str_tp=""
        minEnergy=emin_scan_edge ; maxEnergy=emax_scan_edge ; dE=step_scan_line
        lineE_list = np.arange(minEnergy,maxEnergy+dE,dE)
        
        minNorm = 0.1 ; maxNorm=2.5 ; n=20
        Edge_tested=np.linspace(maxNorm,minNorm,n+1)

        data_file = xcmfile.replace(".xcm","")+"_EdgeSearch_plot.npy"
        Xset.restore(xcmfile)
        Fit.perform()
        print("The fit statistics is ",Fit.statistic)
        cstat_bf=Fit.statistic
        AllModels.show()
        cstat = []

                    
        for energy in lineE_list:
            Xset.restore(xcmfile)
            Fit.perform()
            AllModels(1).zedge.MaxTau.frozen=False
            AllModels(1).zedge.MaxTau.values = (0,0.001,-1,0,100,1000)
            AllModels(1).zedge.edgeE.values = (energy,-1)
            AllModels.show()
            Fit.steppar(f"{AllModels(1).zedge.MaxTau.index} {maxNorm} {minNorm} {n}")
            cstat.append(Fit.stepparResults("statistic"))
            cstat_array_int=Fit.stepparResults("statistic")
            print(cstat_array_int,"cstat-16",cstat_bf-16)
            if np.min(cstat_array_int) < cstat_bf - 9. :
                print("Minimum reached at Energy ",energy," norm = ",Edge_tested[np.argmin(cstat_array_int)], " Delta-Cstat=",np.min(cstat_array_int) - cstat_bf)
                AllModels(1).zedge.edgeE.frozen=True
                AllModels(1).zedge.MaxTau.values = (Edge_tested[np.argmin(cstat_array_int)])
                Fit.perform()
                str_tp+=",".join(str(tp) for tp in [AllModels(1).zedge.edgeE.values[0], AllModels(1).zedge.MaxTau.values[0],Fit.statistic,np.min(cstat_array_int), Fit.statistic-cstat_bf,cstat_bf])+"\n"

        if len(str_tp) > 0 :
            f = open(xcmfile.replace(".xcm","")+"_EdgeSearch_plot.bfit", 'a')
            f.write(str_tp)
            f.close()

        c = cstat_bf-np.array(cstat)
        c[c<0] = 0

        x,y = np.meshgrid(lineE_list,np.linspace(maxNorm,minNorm,n+1))
        np.save(data_file,np.array([x,y,c],dtype=object))
        x,y,c = np.load(data_file,allow_pickle=True)

        fig,ax = plt.subplots(1,1,figsize=(12,6))
        contour = ax.contourf(x,y,c.T,cmap="Blues")
        plt.xlabel("Energy (keV)")
        plt.ylabel("Norm")
        cbar = plt.colorbar(contour)
        cbar.set_label(r'$\Delta$ '+Fit.statMethod)
        plt.hlines(0,minEnergy,maxEnergy,color="k",linestyle="--",alpha=0.5)
        ax.set_xlim(minEnergy,maxEnergy)
        fig.tight_layout()
        fig.savefig(xcmfile.replace(".xcm","")+"_EdgeSearch_plot.pdf")
        
    
    
db_scan_residuals_for_line_and_edges('/Users/lucas/Documents/IRAP/CrossCorrelationSearch/data/BURST_SPECTRA/burst_0217sig_num1_2050300110_intervals_for1000counts/grouping_step1_bin1','f013_TBabs_diskbb_powerlaw_famodel.xcm', scan_line=True, emin_scan_line=0.3,emax_scan_line=3.0, step_scan_line=0.04, scan_edge=False, emin_scan_edge=0.3,emax_scan_edge=3.0, step_scan_edge=0.1)