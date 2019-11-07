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

from tess_binaries import *

def loadSampleFromHDF5(fname)
    ff = h5py.File(fname, mode="r")

    df = {}
    for key in list(ff):
        try:
            df[key] = ff[key].value
        except:
            df[key] = np.array(ff[key].value, dtype='str')

    ff.close()
    sample = pd.DataFrame(data=df)
    
    return sample


def loadPickle(ID):
    ls_period, bls_period = [], []

    infile = open(f'{tb.ps_dir}/{ID}_ps.pkl','rb')
    ps_dict = pickle.load(infile)
    infile.close()
    
    return ps_dict['ls_best_period']


if __name__ == "__main__":
	sample = loadSampleFromHDF5(f'{tb.cat_dir}/asassn_tess_inspected.hdf5')
    