import numpy as np
import pandas as pd
import math
import os
import glob
import time
import pickle

from astropy.timeseries import LombScargle

from .data import readSourceFiles
from .__init__ import lc_dir, ps_dir


def computePowerSpectra(tic_id, **kwargs):

    data, end_times = readSourceFiles(tic_id)
    ts = data[~np.isnan(data['pdcsap_flux'])]
    
    #Compute lomb-scargle power series
    ls0 = time.time()
    freq, power = LombScargle(ts.time.jd, ts['pdcsap_flux'], dy=ts['pdcsap_flux_err']).autopower()
    ls_time = time.time() - ls0
    ls_pwr_spec  = np.vstack([np.array(1/freq), np.array(power/max(power))])
    print(f'Lomb-Scargle compute time: {ls_time}')
    
    #find best lomb-scargle period
    ls_max_pwr = np.argmax(ls_pwr_spec, axis=1)[1]
    ls_best_period = ls_pwr_spec[0][ls_max_pwr]

    ps_output = {'tic_id':tic_id, 'data': data, 'ls_pwr_spec': ls_pwr_spec, 'bls_pwr_spec': None, \
                 'ls_best_period': ls_best_period, 'bls_best_period': None, \
                 'ls_time': ls_time, 'bls_time': None, 'end_times': end_times}
    
    fname = f'{ps_dir}/{tic_id}_ps.pkl'
    
    print(f'Saving power spectra to {ps_dir}.')
    output = open(fname, 'wb')
    pickle.dump(ps_output, output)
    
    return ps_output