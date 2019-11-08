import tess_binaries as tb

__all__ = ['LightCurve', 'PowerSpectrum', 'Sample']


class LightCurve():
	"""
	Class to store and perform operations on individual light curves
	"""
	def __init__(self, *args, **kwargs):

		self.tic_id 	= kwargs.get('tic_id')		# TESS input catalog ID

		if ('time' in kwargs) and ('flux' in kwargs):
			self.time 		= kwargs.get('time')		
			self.flux 		= kwargs.get('flux')		
			self.flux_err 	= kwargs.get('flux_err')	

		else:
			lc = tb.loadLightCurve(self.tic_id)
			self.time 		= lc[0]					# time in MJD
			self.flux 		= lc[1]					# PDCSAP flux
			self.flux_err 	= lc[2]					# PDCSAP flux error

	def phaseFold():

		return fold_flux


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