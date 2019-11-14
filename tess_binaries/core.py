import numpy as np
import math
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

__all__ = ['LightCurve', 'PowerSpectrum', 'Sample']


class LightCurve():
    """
    Class to store and perform operations on individual light curves
    """
    def __init__(self, *args, **kwargs):

        self.tic_id = kwargs.get('tic_id')          # TESS input catalog ID
        self.type   = kwargs.get('type', None)

        if ('time' in kwargs) and ('flux' in kwargs):
            self.time 		= kwargs.get('time')
            self.flux 		= kwargs.get('flux')
            self.flux_err 	= kwargs.get('flux_err')

        else:
            lc = tb.loadLightCurve(self.tic_id)
            self.time 		= np.array(lc[0])					# time in MJD
            self.flux 		= np.array(lc[1])					# PDCSAP flux
            self.flux_err 	= np.array(lc[2])					# PDCSAP flux error

    def powerSpectrum(self, **kwargs):
        
        method  = kwargs.get('method', 'ls')
        self.ps = tb.loadPowerSpectrum(self.tic_id)
        
        return self.ps

    def phaseFold(self, period):

        self.period = period

        lc_t0 = min(self.time)
        lc_tm = max(self.time)

        phase = self.time/period - np.floor(self.time/period)
        sort_idx = np.argsort(phase)

        self.phase = phase[sort_idx]
        self.fold_flux = self.flux[sort_idx]
        self.fold_flux_err = self.flux_err[sort_idx]

        return np.vstack([self.phase, self.fold_flux, self.fold_flux_err])

    def plot(self, **kwargs):

        plt.figure(figsize=[16,8])
        plt.ticklabel_format(useOffset=False)
        plt.plot(self.time, self.flux/np.nanmedian(self.flux), color='k', linewidth=.5)
        plt.ylabel('PDCSAP Flux', fontsize=18)
        plt.xlabel('Julian Date', fontsize=18)
        plt.xlim(min(self.time), max(self.time))
        # plt.legend(loc='upper right', frameon=False, fontsize=16)
        plt.minorticks_on()

        if 'save_dir' in kwargs:
            save_dir = kwargs.get('save_dir')
            plt.savefig(f'{save_dir}/TIC_{self.tic_id}.png')
        plt.show() 


class PowerSpectrum():
    """
    Class to store power spectra and select periods/harmonics
    """
    def __init__(self, *args, **kwargs):

    	self.tic_id = kwargs.get('tic_id')

    def bestPeriod():
    	return None

    def returnPeaks():
    	return None


class Sample():
    """
    Class to store light curves of training/test samples
    """
    def __init__(self, *args, **kwargs):

    	self.tic_id = kwargs.get('tic_id')
        
    def computePeriods(**kwargs):
        return None

    def preprocess(fold=True, shift=True, scale=True):
        
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
        
    def dmatrix(**kwargs):
        return None
        
    def dtw_knn(k=1):
        
        k = kwargs.get('k', 1)