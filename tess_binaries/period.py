import numpy as np
import pandas as pd
import math
import os
import glob
import time
import pickle

from astropy import units as u
from astropy.timeseries import BoxLeastSquares, LombScargle

import tess_binaries as tb

__all__ = ['bestPeriod', 'computePowerSpectra', 'binData']


def bestPeriod(ps):
    max_pwr = np.argmax(ps, axis=1)[1]
    best_period = ps[0][max_pwr]

    return best_period


def computePowerSpectra(tic_id, **kwargs):
    
    load_dir = kwargs.get('load_dir', tb.ps_dir)
    method  = kwargs.get('method', 'ls')

    mjd, flux, flux_err = tb.loadLightCurve(tic_id)
    data = pd.DataFrame(data={'time':mjd, 'flux':flux, 'flux_err':flux_err})
    ts = data[~np.isnan(data['flux'])]
    
    if method == 'ls':
        # Compute lomb-scargle power series
        ls0 = time.time()
        ls =  LombScargle(ts['time'] * u.day, ts['flux'], dy=ts['flux_err'], \
                          normalization='standard')
        ls_freq, ls_power = ls.autopower(minimum_frequency=.001, \
                                         maximum_frequency=100, samples_per_peak=10)
        ls_time = time.time() - ls0

        ls_pwr_spec  = np.vstack([np.array(1/ls_freq), np.array(ls_power)])
        print(f'Lomb-Scargle compute time: {ls_time}')

        # Save power spectrum to numpy file
        ls_fname = f'{load_dir}/{tic_id}_ps_ls.npy'
        print(f'Saving power spectra to {ls_fname}.')
        np.save(ls_fname, ls_pwr_spec)

        power_spectra = ls_pwr_spec
    
    ### NOT WORKING ###
    elif method == 'bls': 
        # Compute box-least-squares power series
        bls0 = time.time()
        psamp = 10**3
        rng = [ls_best_period/5, ls_best_period*5]          # center BLS grid around LS best period
        model = BoxLeastSquares(ts['time'] * u.day, ts['flux'], dy=ts['flux_err'])
        periods = np.linspace(rng[0], rng[1], psamp) * u.day
        periodogram = model.power(periods, rng[0]/2)
        bls_time = time.time() - bls0
        
        bls_pwr_spec = np.vstack([np.array(periodogram.period), np.array(periodogram.power)])
        print(f'Box-Least-Squares compute time: {bls_time}')

        # Save power spectrum to numpy file
        bls_fname = f'{load_dir}/{tic_id}_ps_bls.npy'
        print(f'Saving power spectra to {bls_fname}.')
        np.save(bls_fname, bls_pwr_spec)

        power_spectra = bls_pwr_spec
    
    return power_spectra


def binData(data, tsteps):
    
    tbins = np.linspace(min(data.time.jd), max(data.time.jd), tsteps+1)
    bin_width = (tbins[1] - tbins[0])/2
    bin_flux = []
    for i in range(tsteps):
        bin_ind = np.where((data.time.jd > tbins[i]) & (data.time.jd < tbins[i+1]))[0]
        bin_flux.append(np.nanmedian(data['pdcsap_flux'][bin_ind])/u.electron*u.second)
    bin_flux = np.array(bin_flux)/np.nanmedian(np.array(bin_flux))
    
    return bin_flux


# =======================================================================
# Old functions
# =======================================================================

def _computePowerSpectra(tic_id, **kwargs):

    data, end_times = tb.loadSourceFromFits(tic_id)
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
    
    fname = f'{tb.ps_dir}/{tic_id}_ps.pkl'
    
    print(f'Saving power spectra to {tb.ps_dir}.')
    output = open(fname, 'wb')
    pickle.dump(ps_output, output)
    
    return ps_output