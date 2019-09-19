import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib import rc
plt.style.use('classic')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('figure', facecolor='w')
rc('xtick', labelsize=20)
rc('ytick', labelsize=20)
import math, os

from astropy.io import fits, ascii
from astropy.table import Table
from astropy.timeseries import TimeSeries, LombScargle, BoxLeastSquares

tess_dir = os.environ['TESS_DATA']
lc_dir = tess_dir + '/light_curves'
sb_dir = tess_dir + '/simbad'


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

        
def plotTimeSeries(tic_id, **kwargs):

    save_dir = kwargs.get('save_dir', f'/plots')
    label = str(kwargs.get('label', 'Data'))
    tic_id = str(tic_id)

    try:
        fpath = glob.glob(f'{lc_dir}/*{tic_id}*.fits')[0]
        fname = fpath.split('/')[-1].split('.fits')[0]
    except:
        raise Exception(f'{tic_id} not downloaded.')

    data = TimeSeries.read(fpath, format='tess.fits')  
    ts = data[~np.isnan(data['pdcsap_flux'])]

    fig, (ax1,ax2) = plt.subplots(2,1, figsize=[16,16])
    ax1.ticklabel_format(useOffset=False)
    ax1.plot(data.time.jd, data['pdcsap_flux']/np.nanmedian(data['pdcsap_flux']), \
             color='k', linewidth=.5, label=label)
    ax1.set_title('TIC' + tic_id, fontsize=25)
    ax1.set_ylabel('PDCSAP Flux', fontsize=18)
    ax1.set_xlabel('Julian Date', fontsize=18)
    ax1.set_xlim(min(data.time.jd), max(data.time.jd))
    ax1.legend(loc='upper right', frameon=False)

    freq, power = LombScargle(ts.time.jd, ts['pdcsap_flux'], dy=ts['pdcsap_flux_err']).autopower()
    model = BoxLeastSquares(ts.time.jd, ts['pdcsap_flux'], dy=ts['pdcsap_flux_err'])
    periodogram = model.autopower(0.2)

    ax2.plot(periodogram.period, periodogram.power/max(periodogram.power), label='Box Least Squares')
    ax2.plot(1/freq, power/max(power), label='Lomb Scargle')
    ax2.set_xscale('log')
    ax2.set_xlim(.01,100)
    ax2.set_ylim(0,1)
    ax2.legend(loc='upper right', frameon=False)
    ax2.set_xlabel('Period (days)', fontsize=18)
    ax2.set_ylabel('Power (normalized)', fontsize=18)  

    plt.minorticks_on()
    plt.savefig(f'{save_dir}/{fname}.png')
    plt.show() 