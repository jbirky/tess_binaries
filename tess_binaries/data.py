import numpy as np
import pandas as pd
import math
import os
import glob
import time
import pickle

from astropy import units as u
from astropy.io import fits, ascii
from astropy.table import Table, vstack
from astropy.timeseries import TimeSeries

lc_dir = os.environ['TESS_DATA']
ps_dir = '../results/combined_sector_power_spectra'
        
        
def readSourceFiles(tic_id, **kwargs):
    
    tic_id = str(tic_id)
    
    if 'sector' in kwargs:
        sec = str(kwargs.get('sector')).rjust(3, '0')
        files = glob.glob(f'{lc_dir}/sector{sec}/*{tic_id}*.fits')
    else:
        files = glob.glob(f'{lc_dir}/sector*/*{tic_id}*.fits')

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


def loadPowerSpectrum(tic_id, **kwargs):
    
    return None