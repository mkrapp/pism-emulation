# PISM emulator

We use *Gaussian Processes* to emulate global sea-level contributions from PISM under various climate forcings (*RCP2.6* and *RCP8.5*) and model parameter combinations.

## Getting started

We need at least Python in version 3 and as a virtual environment we recommend using `conda`.

```
conda create --name pism-emu
conda activate pism-emu
conda install pandas matplotlib tqdm scikit-learn netCDF4
```

## Workflow

1. **Pre-processing:** Convert time series data and store them for later use (we use Python's `pickle` a lot)
2. **Training:** Run the *Gaussian Process Regression* using four PISM parameters ($q$, $E_{ssa}$, $E_{sia}$, $\varphi$), global mean temperature (GMT) (as direct forcing term), cumulative GMT (as committed forcing term), and time since last change in GMT (as a kind of relaxation measure)
3. **Predictions:** Create parameters samples (we use [*Latin Hypercube Sampling*](https://en.wikipedia.org/wiki/Latin_hypercube_sampling)) for each forcing scenario (*RCP2.6* and *RCP8.5*) and check if the emulator prediciton matches the observed record (aka *history matching*)

The emulator predictions (that match observations) from the parameter sampling are stored as CSV file along with the sampled parameter combinations.

## Other data

Global mean temperature forcing from *NorESM1-M* (as surrogate for the spatially explicit PISM forcing) can found [here](http://climexp.knmi.nl/CMIP5/Tglobal/).
