import numpy as np
import pandas as pd
import math
import os
import glob
import time
import pickle

from astropy import units as u
from astropy.timeseries import LombScargle

from .data import readSourceFiles
lc_dir   = os.environ['TESS_DATA']
ps_dir   = '../results/combined_sector_power_spectra'

__all__ = ['computePowerSpectra', 'binData']

def computePowerSpectra(tic_id, **kwargs):

    data, end_times = readSourceFiles(tic_id)
    ts = data[~np.isnan(data['pdcsap_flux'])]
    
    #Compute lomb-scargle power series
    ls0 = time.time()
    ls =  LombScargle(ts.time.jd, ts['pdcsap_flux'], dy=ts['pdcsap_flux_err'], \
                          normalization='standard')
    ls_freq, ls_power = ls.autopower(minimum_frequency=.001, \
                          maximum_frequency=100, samples_per_peak=10)
    ls_time = time.time() - ls0
    ls_pwr_spec  = np.vstack([np.array(1/ls_freq), np.array(ls_power)])
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


def binData(data, tsteps):
    
    tbins = np.linspace(min(data.time.jd), max(data.time.jd), tsteps+1)
    bin_width = (tbins[1] - tbins[0])/2
    bin_flux = []
    for i in range(tsteps):
        bin_ind = np.where((data.time.jd > tbins[i]) & (data.time.jd < tbins[i+1]))[0]
        bin_flux.append(np.nanmedian(data['pdcsap_flux'][bin_ind])/u.electron*u.second)
    bin_flux = np.array(bin_flux)/np.nanmedian(np.array(bin_flux))
    
    return bin_flux