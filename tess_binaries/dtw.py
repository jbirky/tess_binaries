import numpy as np
import pandas as pd
from astropy import units as u
from dtaidistance import dtw
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib import rc
plt.style.use('classic')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('figure', facecolor='w')
rc('xtick', labelsize=20)
rc('ytick', labelsize=20)

import tess_binaries as tb

__all__ = ['preprocessData', 'plotDTW']


def preprocessData(dataframe, period, tsteps=100, fold=True, scale=True):
    
    pharr = np.linspace(0,1,tsteps)
    
    if fold == True:
        phase_fold_df = dataframe.fold(period=period*u.day)
        binned_flux = tb.binData(phase_fold_df, tsteps)
    else:
        binned_flux = np.array(dataframe['pdcsap_flux'])
        
    shift_flux = np.roll(np.array(binned_flux), tsteps-np.argmin(binned_flux))

    if scale == True:
        scaler = MinMaxScaler()
        phase_flux_array = np.vstack([pharr, shift_flux]).T
        scaler.fit(phase_flux_array)
        scaled_flux = scaler.transform(phase_flux_array).T[1]

        return scaled_flux
    
    else:
        return shift_flux
        

def plotDTW(flux1, flux2, **kwargs):
    
    tsteps = len(flux1)
    pharr = np.linspace(0,1,tsteps)
    
    window = kwargs.get('window', tsteps)
    types  = kwargs.get('types', ['Source 1', 'Source 2'])
    
    d, paths = dtw.warping_paths(flux1, flux2, window=window, psi=2)
    matches = dtw.best_path(paths)

    plt.figure(figsize=[16,8])
    plt.plot(pharr, flux1, label=types[0])
    plt.plot(pharr, flux2, label=types[1])

    for m in matches:
        plt.plot((pharr[m[0]], pharr[m[1]]), (flux1[m[0]], flux2[m[1]]), color='k')
    plt.title(f'DTW Distance: {str(np.round(d,3))}', fontsize=18)
    plt.legend(loc='lower right', fontsize=16)
    plt.ticklabel_format(useOffset=False)
    plt.show()