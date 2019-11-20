### Authors:

* Jessica Birky (UW)
* Jim Davenport (UW)

### Documentation (under construction!):
(Book-keeping for myself to maybe become something someday...)

#### Period Analysis
Read combined-sector light curve (given a TICID):
```
import tess_binaries as tb

lc = tb.LightCurve(tic_id='117544915')
lc.plot(opt='lc')
```
![](/docs/117544915_lc.png)

Compute Lomb-Scargle power spectrum and plot:
```
lc.powerSpectrum()
lc.plot(opt='ps')
```
![](/docs/117544915_ps.png)

Plot the phase-folded light curve of the best Lomb-Scargle period:
```
lc.phaseFold(period=lc.best_period)
lc.smoothData(window=100)
lc.plot(opt='smooth')
```
![](/docs/117544915_ls_phase.png)

Run phase-dispersion minimization around different factors of the best Lomb-Scargle period:
```
pdm_period = tb.pdm(lc, p0=lc.best_period, factors=[1,2,4], bound=.1, window=100)
lc.smoothData(window=100)
lc.plot(opt='smooth')
```
![](/docs/117544915_pdm_phase.png)

#### Parallelized functions (for N objects):
Reading in a file and computing power spectra for a set of sources:
```
sample = pd.read_csv(f'{tb.cat_dir}/combined_sectors_1_15.csv.gz')

tb.sampleSaveLightCurves(list(sample['tic_id']))
tb.sampleComputePowerSpectra(list(sample['tic_id']))
```
Computing phase-dispersion minimization for a set of sources:
```
pdm_periods = tb.samplePDM(sample['tic_id'])
```