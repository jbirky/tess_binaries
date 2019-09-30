from .data import readSourceFiles
from .period import computePowerSpectra
from .plot import plotTimeSeries, plotLightCurve

import os

cat_dir  = '../catalogs'
lc_dir   = os.environ['TESS_DATA']
ps_dir   = '../results/combined_sector_power_spectra'
plot_dir = '../results/plots'
