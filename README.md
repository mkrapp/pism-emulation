# PISM emulator

We use a non-linear regression model to emulate Antarctic sea-level contributions from PISM under various climate forcings (*RCP2.6* and *RCP8.5*) and model parameter combinations.

## Getting started

Before we start, we create a virtual environment to install all our required libraries using `conda`:

```
conda env create --file environment.yml
conda activate pism-emu
```

If you want to use some of the code in a Jupyter notebook, you need to register this virtual environment as a kernel:

```
python -m ipykernel install --name pism-emu
```

Then simply type `jupter notebook` and you're good to go.

## Workflow

Our workflow management tool is [`Snakemake`](https://snakemake.readthedocs.io/en/stable/). It takes care of dependencies between different parts of the workflow, such as downloading and pre-processing of input data, running the emulator, and visualizing results. Basically it follows these steps:

1. **Pre-processing:** Convert time series data and store them for later use (we use Python's `pickle` a lot)
2. **Training:** Run the *Gaussian Process Regression* using four PISM parameters ($q$, $E_{ssa}$, $E_{sia}$, $\varphi$), global mean temperature (GMT) (as direct forcing term), cumulative GMT (as committed forcing term), and time since last change in GMT (as a kind of relaxation measure)
3. **Predictions:** Create parameters samples (we use [*Latin Hypercube Sampling*](https://en.wikipedia.org/wiki/Latin_hypercube_sampling)) for each forcing scenario (*RCP2.6* and *RCP8.5*) and check if the emulator prediciton matches the observed record (aka *history matching*)
4. **Visualizations:** Create figures and animations for the manuscript and supplementary information

The emulator predictions (that match observations) from the parameter sampling are stored as CSV file along with the sampled parameter combinations.


To run the whole workflow, call
```
snakemake run_all --cores 1
```

## Other data

Global mean temperature forcing from *NorESM1-M* (as surrogate for the spatially explicit PISM forcing) can found [here](http://climexp.knmi.nl/CMIP5/Tglobal/) and will be downloaded automatically.


## Citation

Lowry, D.P., Krapp, M., Golledge, N.R. et al. The influence of emissions scenarios on future Antarctic ice loss is unlikely to emerge this century. Commun Earth Environ 2, 221 (2021). https://doi.org/10.1038/s43247-021-00289-2
