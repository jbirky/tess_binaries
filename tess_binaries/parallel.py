import pandas as pd
import multiprocessing as mp
import time

import tess_binaries as tb

__all__ = ['sampleSaveLightCurves', 'sampleComputePowerSpectra']


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