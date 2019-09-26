import numpy as np
import pandas as pd
import math
import os
import glob
import time
import pickle

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib import rc
plt.style.use('classic')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('figure', facecolor='w')
rc('xtick', labelsize=20)
rc('ytick', labelsize=20)

from astropy import units as u
from astropy.io import fits, ascii
from astropy.table import Table, vstack
from astropy.timeseries import TimeSeries, LombScargle, BoxLeastSquares, aggregate_downsample

tess_dir = os.environ['TESS_DATA']
lc_dir = tess_dir + '/light_curves'
sb_dir = tess_dir + '/simbad'
ps_dir = tess_dir + '/combined_sector_power_spectra'


if __name__ == "__main__":
	print('a')