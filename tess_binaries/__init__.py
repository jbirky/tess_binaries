from .read import *
from .period import *
from .plot import *
from .dtw import *

import os

tb_path  = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))		# tess_binaries code directory
data_dir = os.environ['TESS_DATA']											# raw tess fits data files
cat_dir  = tb_path + '/catalogs'											# catalog files (csv)
lc_dir   = tb_path + '/results/light_curves'								# combined sector light curves
ps_dir   = tb_path + '/results/power_spectra'								# combined sector power spectra
plot_dir = tb_path + '/results/plots'


for directory in [cat_dir, lc_dir, ps_dir, plot_dir]:		# create directories to read/store results
	if os.path.exists(directory) == False:
		os.system(f'mkdir {directory}')