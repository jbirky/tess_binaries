import numpy as np
import pandas as pd
import h5py
import time

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


if __name__ == "__main__":
    ff = h5py.File(f'{tb.cat_dir}/asassn_tess_inspected.hdf5', mode="r")

    df = {}
    for key in list(ff):
        if key == 'type':
            df[key] = np.array(ff[key].value, dtype='str')
        else:
            df[key] = ff[key].value

    ff.close()
    sample = pd.DataFrame(data=df)

    tb.sampleLoadLightCurves(list(sample['tic_id']))
