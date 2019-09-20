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


def download(**kwargs):
    tic_id = str(kwargs.get('tic_id'))
    sector = kwargs.get('sector')
    dl_dir = kwargs.get('dl_dir', lc_dir)

    mast_url = 'https://mast.stsci.edu/api/v0.1/Download/file/?uri=mast:TESS/product/'
    zfmt = tic_id.rjust(16, '0')

    if sector == 1:
        fname = f'tess2018206045859-s0001-{zfmt}-0120-s_lc.fits'
    elif sector == 2:
        fname = f'tess2018234235059-s0002-{zfmt}-0121-s_lc.fits'
    elif sector == 3:
        fname = f'tess2018263035959-s0003-{zfmt}-0123-s_lc.fits'
    elif sector == 4:
        fname = f'tess2018292075959-s0004-{zfmt}-0124-s_lc.fits'
    elif sector == 5:
        fname = f'tess2018319095959-s0005-{zfmt}-0125-s_lc.fits'
    elif sector == 6:
        fname = f'tess2018349182459-s0006-{zfmt}-0126-s_lc.fits'
    elif sector == 7:
        fname = f'tess2019006130736-s0007-{zfmt}-0131-s_lc.fits'
    elif sector == 8:
        fname = f'tess2019032160000-s0008-{zfmt}-0136-s_lc.fits'
    elif sector == 9:
        fname = f'tess2019058134432-s0009-{zfmt}-0139-s_lc.fits'
    elif sector == 10:
        fname = f'tess2019085135100-s0010-{zfmt}-0140-s_lc.fits'
    elif sector == 11:
        fname = f'tess2019112060037-s0011-{zfmt}-0143-s_lc.fits'
    elif sector == 12:
        fname = f'tess2019140104343-s0012-{zfmt}-0144-s_lc.fits'
    elif sector == 13:
        fname = f'tess2019169103026-s0013-{zfmt}-0146-s_lc.fits'
    else:
        raise Exception(f'{sector} is not a valid TESS sector.')

    if os.path.exists(f'{dl_dir}/{fname}') == False:
        try:
            os.system(f'wget -O {dl_dir}/{fname} {mast_url}{fname}')
        except:
            raise Exception(f'Failed to down {mast_url}{fname}')
            
        if os.path.exists(f'{dl_dir}/{fname}') == True:
            print(f'Downloaded {dl_dir}/{fname}')
        else:
            raise Exception(f'Failed to down {mast_url}{fname}')
    else:
        print(f'{fname} already downloaded.')
        
        
def readSourceFiles(tic_id, **kwargs):
    
    tic_id = str(tic_id)
    files = glob.glob(f'{lc_dir}/*{tic_id}*.fits')

    data_frames = []
    end_times = []

    try:
        for file in files:
            fcontents = TimeSeries.read(file, format='tess.fits')
            
            #normalize flux
            pdcsap_flux = fcontents['pdcsap_flux']
            fcontents['pdcsap_flux_norm'] = pdcsap_flux/np.nanmedian(pdcsap_flux) 
            
            data_frames.append(fcontents)
            end_times.append(max(fcontents.time.jd))

        print(f'Stacking {len(files)} light curve(s).')
    
    except:
        raise Exception(f'{tic_id} not downloaded.')

    data = vstack(data_frames) 
    
    #un-normalize flux
    unit_flux = data['pdcsap_flux_norm'] * np.nanmedian(data['pdcsap_flux'])
    data.remove_column('pdcsap_flux')
    data.remove_column('pdcsap_flux_norm')
    data['pdcsap_flux'] = unit_flux
    
    return data, end_times
    
        
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

    #Compute box-least-squares power series
    if len(end_times) < 4:
        bls0 = time.time()
        model = BoxLeastSquares(ts.time.jd, ts['pdcsap_flux'], dy=ts['pdcsap_flux_err'])
        periodogram = model.autopower(0.2)
        bls_time = time.time() - bls0
        print(f'Box-Least-Squares compute time: {bls_time}')
    
        bls_pwr_spec = np.vstack([np.array(periodogram.period), np.array(periodogram.power/max(periodogram.power))])
        
        bls_max_pwr = np.argmax(bls_pwr_spec, axis=1)[1]
        bls_best_period = bls_pwr_spec[0][bls_max_pwr]

        ps_output = {'tic_id':tic_id, 'data': data, 'ls_pwr_spec': ls_pwr_spec, 'bls_pwr_spec': bls_pwr_spec, \
                     'ls_best_period': ls_best_period, 'bls_best_period': bls_best_period, \
                     'ls_time': ls_time, 'bls_time': bls_time, 'end_times': end_times}
    else:
        ps_output = {'tic_id':tic_id, 'data': data, 'ls_pwr_spec': ls_pwr_spec, 'bls_pwr_spec': None, \
                     'ls_best_period': ls_best_period, 'bls_best_period': None, \
                     'ls_time': ls_time, 'bls_time': None, 'end_times': end_times}
    
    if 'save_dir' in kwargs:
        save_dir = kwargs.get('save_dir')
        fname = f'{save_dir}/{tic_id}_ps.pkl'
        
        print(f'Saving power spectra to {save_dir}.')
        output = open(fname, 'wb')
        pickle.dump(ps_output, output)
    
    return ps_output

        
def plotTimeSeries(ps_dict, **kwargs):

    save_dir = kwargs.get('save_dir', f'/plots')
    label = str(kwargs.get('label', 'Data'))
    
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
    if len(end_times) > 3:
        ax2.set_xlim(.01,1000)
    else:
        ax2.set_xlim(.01,100)
    ax2.set_ylim(0,1)
    ax2.legend(loc='upper right', frameon=False, fontsize=16)
    ax2.set_xlabel('Period (days)', fontsize=18)
    ax2.set_ylabel('Power (normalized)', fontsize=18)  
    ax2.minorticks_on()
    
    plt.savefig(f'{save_dir}/TIC_{tic_id}.png')
    plt.show() 