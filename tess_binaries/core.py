import numpy as np
import pandas as pd
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
            self.norm_flux  = self.flux/np.nanmedian(self.flux)

        self.baseline = max(self.time) - min(self.time)

    def powerSpectrum(self, method='ls'):
        
        self.ps = tb.loadPowerSpectrum(self.tic_id, ps_type=method)
        self.best_period = tb.bestPeriod(self.ps)
        
        return self.ps, self.best_period

    def phaseFold(self, **kwargs):

        self.period = kwargs.get('period', self.best_period)

        lc_t0 = min(self.time)
        lc_tm = max(self.time)

        phase = self.time/self.period - np.floor(self.time/self.period)
        sort_idx = np.argsort(phase)

        self.phase = phase[sort_idx]
        self.phase_flux = self.flux[sort_idx]
        self.phase_flux_err = self.flux_err[sort_idx]
        self.norm_phase_flux = self.norm_flux[sort_idx]

        return np.vstack([self.phase, self.phase_flux, self.phase_flux_err])

    def smoothData(self, method='rolling_median', window=128):

        bin_flux = []
        if method == 'rolling_median':
            for i in range(len(self.phase_flux)):
                bin_flux.append(np.nanmedian(self.phase_flux[i-window:i+window]))
            # self.bin_flux = pd.Series(self.norm_phase_flux, center=True).rolling(window).median()
        self.bin_flux = np.array(bin_flux)
        self.norm_bin_flux = self.bin_flux/np.nanmedian(self.bin_flux)

        return self.bin_flux

    def plot(self, **kwargs):

        opt = kwargs.get('opt') 

        if opt == 'lc':
            plt.figure(figsize=[16,8])
            plt.ticklabel_format(useOffset=False)
            plt.plot(self.time, self.flux/np.nanmedian(self.flux), color='k', linewidth=.5)
            plt.ylabel('PDCSAP Flux', fontsize=18)
            plt.xlabel('Julian Date', fontsize=18)
            plt.xlim(min(self.time), max(self.time))
            # plt.legend(loc='upper right', frameon=False, fontsize=16)
            plt.minorticks_on()

        elif opt == 'ps':
            plt.figure(figsize=[16,8])
            plt.ticklabel_format(useOffset=False)
            plt.axvline(self.best_period, color='r', linewidth=.5, alpha=.6, \
                label=f'Period: {round(self.best_period,3)}')
            plt.plot(self.ps[0], self.ps[1], color='k', linewidth=1)
            plt.ylabel('Power [normalized]', fontsize=18)
            plt.xlabel('Period [days]', fontsize=18)
            plt.xlim(min(self.ps[0]), max(self.ps[0]))
            plt.ylim(0,1)
            plt.legend(loc='upper right', frameon=False, fontsize=16)
            plt.minorticks_on()
            plt.xscale('log')

        elif opt == 'phase':
            plt.figure(figsize=[16,8])
            plt.ticklabel_format(useOffset=False)
            plt.scatter(self.phase, self.norm_phase_flux, color='k', s=1)
            plt.ylabel('PDCSAP Flux', fontsize=18)
            plt.xlabel('Phase', fontsize=18)
            plt.xlim(min(self.phase), max(self.phase))
            plt.minorticks_on()

        elif opt == 'smooth':
            plt.figure(figsize=[16,8])
            plt.ticklabel_format(useOffset=False)
            plt.scatter(self.phase, self.norm_phase_flux, color='k', s=1)
            plt.plot(self.phase, self.norm_bin_flux, color='r')
            plt.ylabel('PDCSAP Flux', fontsize=18)
            plt.xlabel('Phase', fontsize=18)
            plt.xlim(min(self.phase), max(self.phase))
            plt.minorticks_on()
        
        if 'title' in kwargs:
            plt.title(kwargs.get('title'), fontsize=20)
        if 'save_dir' in kwargs:
            save_dir = kwargs.get('save_dir')
            plt.savefig(save_dir)
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