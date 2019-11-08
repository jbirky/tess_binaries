import tess_binaries as tb

__all__ = ['LightCurve', 'PowerSpectrum', 'Sample']


class LightCurve():
	"""
	Class to store and perform operations on individual light curves
	"""
	def __init__(self, *args, **kwargs):

		self.tic_id = kwargs.get('tic_id')		# TESS input catalog ID
		self.time = kwargs.get('time')			# time in MJD
		self.flux = kwargs.get('flux')			# PDCSAP flux
		self.flux_err = kwargs.get('flux_err')	# PDCSAP flux error

	def loadLC():
		data, end_times = tb.readSourceFiles(self.tic_id)

		self.time = np.array(data.time.jd)
		self.flux = np.array(data['pdcsap_flux'])
		self.flux_err = np.array(data['pdcsap_flux_err'])

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