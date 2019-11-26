import numpy as np
import pandas as pd
from astropy import units as u
from dtaidistance import dtw
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib import rc
plt.style.use('classic')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('figure', facecolor='w')
rc('xtick', labelsize=20)
rc('ytick', labelsize=20)

import tess_binaries as tb

__all__ = ['preprocessData', 'plotDTW']


def preprocessData(dataframe, period, tsteps=100, fold=True, scale=True):
    
    pharr = np.linspace(0,1,tsteps)
    
    if fold == True:
        phase_fold_df = dataframe.fold(period=period*u.day)
        binned_flux = tb.binData(phase_fold_df, tsteps)
    else:
        binned_flux = np.array(dataframe['pdcsap_flux'])
        
    shift_flux = np.roll(np.array(binned_flux), tsteps-np.argmin(binned_flux))

    if scale == True:
        scaler = MinMaxScaler()
        phase_flux_array = np.vstack([pharr, shift_flux]).T
        scaler.fit(phase_flux_array)
        scaled_flux = scaler.transform(phase_flux_array).T[1]

        return scaled_flux
    
    else:
        return shift_flux
        

def plotDTW(flux1, flux2, **kwargs):
    
    tsteps = len(flux1)
    pharr = np.linspace(0,1,tsteps)
    
    window = kwargs.get('window', tsteps)
    types  = kwargs.get('types', ['Source 1', 'Source 2'])
    
    d, paths = dtw.warping_paths(flux1, flux2, window=window, psi=2)
    matches = dtw.best_path(paths)

    plt.figure(figsize=[16,8])
    plt.plot(pharr, flux1, label=types[0])
    plt.plot(pharr, flux2, label=types[1])

    for m in matches:
        plt.plot((pharr[m[0]], pharr[m[1]]), (flux1[m[0]], flux2[m[1]]), color='k')
    plt.title(f'DTW Distance: {str(np.round(d,3))}', fontsize=18)
    plt.legend(loc='lower right', fontsize=16)
    plt.ticklabel_format(useOffset=False)
    plt.show()


def plotConfusionMatrix(cmatrix, tnames, **kwargs):
    
    k = kwargs.get('k', 'k')
    show_percent = kwargs.get('percent', False)
    nlabel = len(tnames)

    fig, ax = plt.subplots(figsize=[12,12])
    if show_percent == True:
        cmatrix = np.round(cmatrix/np.sum(cmatrix, axis=0), 2)
    
    im = ax.matshow(cmatrix, cmap='Blues', norm=colors.PowerNorm(gamma=0.6))

    ax.set_xticks(np.arange(nlabel))
    ax.set_yticks(np.arange(nlabel))
    ax.set_xticklabels(tnames)
    ax.set_yticklabels(tnames)

    ax.xaxis.set_ticks_position('bottom')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(nlabel):
        for j in range(nlabel):
            val = cmatrix[i, j]
            if val > .65*(np.max(cmatrix) - np.min(cmatrix)):
                text = ax.text(j, i, val, ha="center", va="center", color="w")
            else:
                text = ax.text(j, i, val, ha="center", va="center", color="k")

    ax.set_title(f"{k}-NN Confusion Matrix", fontsize=25)
    ax.set_xlabel('Reference Label', fontsize=20)
    ax.set_ylabel(f'{k}-NN Label', fontsize=20)
    fig.tight_layout()
    if 'save_dir' in kwargs:
        plt.savefig(kwargs.get('save_dir'))
    
    return fig
    

def returnKNN(dmatrix, k):
    
    nn1 = []
    for i in range(len(dmatrix)):
        min_dist = sorted(dmatrix[i])[1:k+1]
        min_dist_ind = np.array([np.where(dmatrix[i] == d)[0][0] for d in min_dist])
        votes = np.array(subsamp['type'])[min_dist_ind]
        best = Counter(votes).most_common(1)[0][0]
        nn1.append(best)

    train = list(subsamp['type'])
    test = nn1
    
    return confusion_matrix(test, train, labels=tnames)