from .read import *
from .period import *
from .plot import *
from .dtw import *

import os

data_dir = os.environ['TESS_DATA']
cat_dir  = '../catalogs'
ps_dir   = '../results/combined_sector_power_spectra'
plot_dir = '../results/plots'
