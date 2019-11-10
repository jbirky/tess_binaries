import numpy as np
import pandas as pd
import glob
import h5py
import warnings

from astropy import units as u
from astropy.table import Table, vstack
from astropy.timeseries import TimeSeries

import tess_binaries as tb

__all__ = ['loadSourceFromFits', 'loadLightCurve', 'loadPowerSpectrum', 'loadSampleFromHDF5']     

   
# =======================================================================
# Read raw individual source files
# =======================================================================

def loadSourceFromFits(tic_id, **kwargs):
    """
    Load combined sector light curve from TESS fits files
    """
    load_dir = kwargs.get('load_dir', tb.data_dir)
    tic_id = str(tic_id)
    
    if 'sector' in kwargs:
        sec = str(kwargs.get('sector')).rjust(3, '0')
        files = glob.glob(f'{load_dir}/sector{sec}/*{tic_id}*.fits')
    else:
        files = glob.glob(f'{load_dir}/sector*/*{tic_id}*.fits')

    data_frames = []
    end_times = []

    try:
        for file in files:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")             #suppress astropy warnings
                fcontents = TimeSeries.read(file, format='tess.fits')
            
            #normalize flux
            pdcsap_flux = fcontents['pdcsap_flux']
            fcontents['pdcsap_flux_norm'] = pdcsap_flux/np.nanmedian(pdcsap_flux) 
            
            data_frames.append(fcontents)
            end_times.append(max(fcontents.time.jd))

        print(f'Stacking {len(files)} light curve(s).')
    
    except:
        raise Exception(f'{tic_id} not downloaded.')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")             #suppress astropy warnings
        data = vstack(data_frames) 
    
    #un-normalize flux
    unit_flux = data['pdcsap_flux_norm'] * np.nanmedian(data['pdcsap_flux'])
    data.remove_column('pdcsap_flux')
    data.remove_column('pdcsap_flux_norm')
    data['pdcsap_flux'] = unit_flux
    
    return data, end_times


# =======================================================================
# Read saved individual files
# =======================================================================

def loadLightCurve(tic_id, **kwargs):
    """
    Load combined sector light curve from numpy array containing [mjd, flux, flux_err]
    """
    load_dir = kwargs.get('load_dir', tb.lc_dir)
    tic_id = str(tic_id)

    try:
        lc = np.load(f'{load_dir}/{tic_id}_lc.npy')

    except:
        print('Loading from FITS files...')
        data, end_times = loadSourceFromFits(tic_id, load_dir=tb.data_dir)
        mjd = np.array(data.time.jd)
        flux = np.array(data['pdcsap_flux'])
        flux_err = np.array(data['pdcsap_flux_err'])

        lc = np.vstack([mjd, flux, flux_err])
        np.save(f'{load_dir}/{tic_id}_lc.npy', lc)

    return lc


def loadPowerSpectrum(tic_id, **kwargs):
    """
    Load power spectra from numpy array 
    """
    load_dir = kwargs.get('load_dir', tb.ps_dir)
    ps_type  = kwargs.get('ps_type', 'ls')
    tic_id = str(tic_id)
    
    try:
        ps = np.load(f'{load_dir}/{tic_id}_ps_{ps_type}.npy')
    except:
        ps = tb.computePowerSpectra(tic_id)
    
    return ps


# =======================================================================
# Read saved sample files
# =======================================================================

def loadSampleFromHDF5(fname):
    print(f'Loading {fname}')
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
