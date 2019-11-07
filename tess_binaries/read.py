import numpy as np
import glob

from astropy import units as u
from astropy.table import Table, vstack
from astropy.timeseries import TimeSeries

import tess_binaries as tb

__all__ = ['readSourceFiles']     

        
def readSourceFiles(tic_id, **kwargs):
    
    tic_id = str(tic_id)
    
    if 'sector' in kwargs:
        sec = str(kwargs.get('sector')).rjust(3, '0')
        files = glob.glob(f'{tb.lc_dir}/sector{sec}/*{tic_id}*.fits')
    else:
        files = glob.glob(f'{tb.lc_dir}/sector*/*{tic_id}*.fits')

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


def loadPowerSpectrum(tic_id, **kwargs):
    
    return None