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
from astropy.timeseries import aggregate_downsample

lc_dir = os.environ['TESS_DATA']
ps_dir = '../results/combined_sector_power_spectra'
plot_dir = '../results/plots'


def plotTimeSeries(ps_dict, **kwargs):

    label = str(kwargs.get('label', 'Data'))
    save_dir = kwargs.get('save_dir', plot_dir)
    
    #upack dictionary
    tic_id = str(ps_dict['tic_id'])
    data = ps_dict['data']
    ts = data[~np.isnan(data['pdcsap_flux'])]
    end_times = ps_dict['end_times']
    ls_pwr_spec = ps_dict['ls_pwr_spec']
    bls_pwr_spec = ps_dict['bls_pwr_spec']

    fig, (ax1,ax2) = plt.subplots(2,1, figsize=[16,16])
    ax1.ticklabel_format(useOffset=False)
    ax1.plot(data.time.jd, data['pdcsap_flux']/np.nanmedian(data['pdcsap_flux']), \
             color='k', linewidth=.5, label=f'{label}'.replace('_', ' '))
    if len(end_times) > 1:
        for t in end_times:
            ax1.axvline(x=t, color='r', alpha=.5)
    if kwargs.get('binned', False) == True:
        ts_binned = aggregate_downsample(data, time_bin_size=.01*u.day) 
        ax1.plot(ts_binned.time_bin_start.jd, ts_binned['pdcsap_flux']/np.nanmedian(ts_binned['pdcsap_flux']), \
                 label='Binned', color='r', linewidth=.5)
    
    ax1.set_title('TIC' + tic_id, fontsize=25)
    ax1.set_ylabel('PDCSAP Flux', fontsize=18)
    ax1.set_xlabel('Julian Date', fontsize=18)
    ax1.set_xlim(min(data.time.jd), max(data.time.jd))
    ax1.legend(loc='upper right', frameon=False, fontsize=16)
    ax1.minorticks_on()
    
    #find best lomb-scargle period
    ls_max_pwr = np.argmax(ls_pwr_spec, axis=1)[1]
    ls_best_period = ls_pwr_spec[0][ls_max_pwr]
    
    ax2.plot(ls_pwr_spec[0], ls_pwr_spec[1], label=r'Lomb Scargle: $P={}$'.format(round(ls_best_period,3)))
    
    #optional: plot box-least-squares power spectrum
    if type(bls_pwr_spec) is np.ndarray:
        bls_max_pwr = np.argmax(bls_pwr_spec, axis=1)[1]
        bls_best_period = bls_pwr_spec[0][bls_max_pwr]
        ax2.plot(bls_pwr_spec[0], bls_pwr_spec[1], label=r'Box Least Squares: $P={}$'.format(round(bls_best_period,3)))
        
    ax2.set_xscale('log')
    ax2.set_xlim(.01,100)
    ax2.set_ylim(0,1)
    ax2.legend(loc='upper right', frameon=False, fontsize=16)
    ax2.set_xlabel('Period (days)', fontsize=18)
    ax2.set_ylabel('Power (normalized)', fontsize=18)  
    ax2.minorticks_on()
    
    plt.savefig(f'{save_dir}/TIC_{tic_id}.png')
    plt.show() 