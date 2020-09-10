import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import sys
sys.path.append("src/models/")
from gaussian_process_emulator import GaussianProcessEmulator
from scipy.signal import savgol_filter
import cmocean

rng = np.random.default_rng(seed=123)

def load(fnm):
    df = pd.read_csv(fnm,delim_whitespace=True,comment='#',header=None,index_col=0).mean(axis=1)
    return df

def main():
    # here goes the main part
    fnm_in = sys.argv[1]
    with open(fnm_in, "rb") as f:
        [_,parameters,time,y_name,miny,maxy,_,_,df] = pickle.load(f)

    miny = -0.1
    maxy = 0.7

    #print(gpr.kernel.get_params())
    fnm_in = sys.argv[2]
    with open(fnm_in, "rb") as f:
        _,_,scenarios = pickle.load(f)
    start_year = 1970
    end_year   = 2100
    time0 = 1992
    print(time)
    nrandom = int(sys.argv[3])
    time = scenarios['rcp26'].loc[start_year:end_year].index
    print(time)
    rcp26 = scenarios['rcp26'].loc[start_year:end_year]
    rcp85 = scenarios['rcp85'].loc[start_year:end_year]

    dt = time[1]-time[0]

    nt = len(rcp26)
    n_params = len(parameters)

    print(y_name)
    gpe = GaussianProcessEmulator()
    gpe.load("./models/")
    def model_update(c,scen):
        X = np.zeros((nt,n_params+3))
        this_forc = scen["global_mean_temperature"]
        x1 = this_forc
        x2 = this_forc.cumsum()
        x2 -= x2.iloc[0]
        x3 = (this_forc.groupby((this_forc != this_forc.shift(1)).cumsum()).cumcount()+1)*dt # years since last temperature change
        for i,t in enumerate(time):
            X[i,:n_params] = c
            X[i,n_params:] = [x1.loc[t],x2.loc[t],x3.loc[t]]
        y_pred, _ = gpe.predict(X)
        idx_diff = time0-start_year
        y_pred = y_pred[:,0] - y_pred[idx_diff,0]

        return np.array(y_pred)

    # Antarctica only; https://www.nature.com/articles/s41586-020-2591-3/tables/1
    t_start = time==1993
    t_end = time==2018
    dslr_obs = 0.44e-3
    # https://www.pnas.org/content/116/4/1095#T2
    t_start = time==1979
    t_end = time==2017
    dslr_obs = 15.9e-3
    # IMBIE team; abstract; https://www.nature.com/articles/s41586-018-0179-y
    t_start = time==1992
    t_end = time==2017
    dslr_obs_mean = 7.6*1e-3
    dslr_obs_max  = (7.6+3.9)*1e-3
    dslr_obs_min  = (7.6-3.9)*1e-3

    # get warming levels from NorESM-M

    gwl = np.arange(1,4.01,0.05)
    fnm_gcm = "data/external/gmt/global_tas_Amon_NorESM1-M_rcp85_r1i1p1.dat"
    df_gcm = load(fnm_gcm).loc[:2100]
    df_filtered = df_gcm*1.0
    order = 3
    pts = 51
    df_filtered.loc[:] = savgol_filter(df_filtered, pts, order)
    mean = df_filtered.loc[1850:1900].mean()

    gwl_dict = {}
    for g in gwl:
        x = df_filtered[df_filtered-mean>=g]
        if len(x)>0:
            year = x.index[0]
            gwl_dict[year] = g
        else:
            year = "NaN"
        print(year)

    fig1, ax1 = plt.subplots(1,1)

    n_combinations = int(sys.argv[3])
    nparams = len(df.T)
    params = np.array([df[df.columns[1:]].min(axis=0).values,df[df.columns[1:]].max(axis=0).values]).T
    print(params)
    pmin = params.min(axis=1)
    pmax = params.max(axis=1)
    n = 0

    nbins = 41
    bins = np.linspace(miny,maxy,nbins)
    print(bins)

    n_gwl = len(gwl_dict.keys())
    X = np.zeros((nbins-1,n_gwl))
    Y = np.zeros((n_gwl,n_combinations))

    gwl_time_idx = [i for i in range(nt) if time[i] in gwl_dict.keys()]
    print(time[gwl_time_idx])

    while n<n_combinations:
        this_params = pmin + (pmax-pmin) * rng.uniform(low=0,high=1,size=nparams-1)
        y2 = model_update(this_params,rcp85)
        if (dslr_obs_min <= y2[t_end]-y2[t_start] <= dslr_obs_max):
            Y[:,n] = y2[gwl_time_idx]
            n += 1
        print("n=%d"%n,end="\r")
        #if (n%10==0): print("n=%d"%n,end="\r")
    print("n=%d"%n)
    # get histograms for binning
    Y50 = np.zeros((n_gwl))
    Y95_lo = np.zeros((n_gwl))
    Y95_hi = np.zeros((n_gwl))
    for i in range(n_gwl):
        freq, _ = np.histogram(Y[i,:], bins)
        freq = freq/(1.*sum(freq))
        X[:,i] = freq
        Y50[i] = np.percentile(Y[i,:],50)
        Y95_lo[i] = np.percentile(Y[i,:],2.5)
        Y95_hi[i] = np.percentile(Y[i,:],97.5)

    #X = np.where(X==0,np.nan,X)

    levels = list(gwl_dict.values())
    im = ax1.imshow(X,origin='lower',extent=[levels[0],levels[-1],bins[0],bins[-1]],cmap=cmocean.cm.dense,aspect='auto',vmin=0,vmax=1)
    ax1.plot(levels,Y50,'k-',lw=2,alpha=0.5)
    ax1.plot(levels,Y95_lo,'k-',lw=1,alpha=0.5)
    ax1.plot(levels,Y95_hi,'k-',lw=1,alpha=0.5)
    plt.colorbar(im,ax=ax1,shrink=0.8)
    ax1.set_xlabel("global warming level (in K)")
    ax1.set_ylabel("Antarctic SLE (in m)")

    plt.show()
    # save plot
    fig1.tight_layout()
    fig1.savefig("reports/figures/warming_level_probabilities.png",dpi=300, bbox_inches='tight', pad_inches = 0.01)

if __name__ == "__main__":
    main()

