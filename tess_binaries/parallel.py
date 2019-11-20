import pandas as pd
import multiprocessing as mp
import time
import warnings

import tess_binaries as tb

__all__ = ['sampleSaveLightCurves', 'sampleComputePowerSpectra', 'samplePDM']


# =======================================================================
# Execute functions in parallel using python multiprocessing
# =======================================================================

def sampleSaveLightCurves(tic_ids):
    pool = mp.Pool(mp.cpu_count())

    t0 = time.time()
    result = pool.map(tb.loadLightCurve, tic_ids)
    t1 = time.time() - t0
    print(f'Loaded {len(tic_ids)} light curves: {t1}')
    
    return result
    

def sampleComputePowerSpectra(tic_ids):
    pool = mp.Pool(mp.cpu_count())

    t0 = time.time()
    result = pool.map(tb.computePowerSpectra, tic_ids)
    t1 = time.time() - t0
    
    print(f'Computed {len(tic_ids)} power spectra: {t1}')
    
    return result


def samplePDM(tic_ids):
    pool = mp.Pool(mp.cpu_count())

    t0 = time.time()

    light_curves = []
    for ID in tic_ids:
        lc = tb.LightCurve(tic_id=ID)

        #cut to only 1 sector!
        if len(lc.time) > 20000:
            lc.time = lc.time[0:20000]
            lc.flux = lc.flux[0:20000]
            lc.flux_err = lc.flux_err[0:20000]

        lc.powerSpectrum()
        light_curves.append(lc)

    #Default values: p0=lc.best_period, factors=[1,2,4], bound=.1, window=200
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") 
        result = pool.map(tb.pdm, light_curves)
    t1 = time.time() - t0

    print(f'Computed {len(tic_ids)} PDM periods: {t1}')

    return result