import numpy as np

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

		self.tic_id = kwargs.get('tic_id')			# TESS input catalog ID
        self.type   = kwargs.get('type', None)

		if ('time' in kwargs) and ('flux' in kwargs):
			self.time 		= kwargs.get('time')		
			self.flux 		= kwargs.get('flux')		
			self.flux_err 	= kwargs.get('flux_err')	

		else:
			lc = tb.loadLightCurve(self.tic_id)
			self.time 		= lc[0]					# time in MJD
			self.flux 		= lc[1]					# PDCSAP flux
			self.flux_err 	= lc[2]					# PDCSAP flux error
            
    def powerSpectrum(**kwargs):
        
        method  = kwargs.get('method', 'ls')
        self.ps = tb.loadPowerSpectrum(self.tic_id)
        
        return self.ps

	def phaseFold(period):

		self.period = period

		lc_t0 = min(self.time)
		lc_tm = max(self.time)

		fold_flux = None

		return fold_flux
    
    def plot(**kwargs):
    
        plt.figure(figsize=[16,8])
        plt.ticklabel_format(useOffset=False)
        if self.type != None:
            plt.plot(self.time, self.flux/np.nanmedian(self.flux), \
                     color='k', linewidth=.5, label=f'{label}'.replace('_', ' '))
        else:
            plt.plot(self.time, self.flux/np.nanmedian(self.flux), color='k', linewidth=.5)
        plt.ylabel('PDCSAP Flux', fontsize=18)
        plt.xlabel('Julian Date', fontsize=18)
        plt.xlim(min(self.time), max(self.time))
        plt.legend(loc='upper right', frameon=False, fontsize=16)
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
		retrun None


class Sample():
	"""
	Class to store light curves of training/test samples
	"""
	def __init__(self, *args, **kwargs):

		self.tic_id = kwargs.get('tic_id')