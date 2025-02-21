import argparse
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from tqdm import tqdm
from numpy.random import default_rng

rng = default_rng(42)

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

plt.rcParams.update({
    "pdf.fonttype" : 42
    })

rcp_colors = {
        'AR6-RCP-8.5': '#980002',
        'AR6-RCP-6.0': '#c37900',
        'AR6-RCP-4.5': '#709fcc',
        'AR6-RCP-2.6': '#003466'}

def main():
    # here goes the main part
    parser = argparse.ArgumentParser(
            prog='ranking',
            description='Ranking samples.')
    parser.add_argument('--model_output', type=str, required=True, help="Output from regression model (to pickle from)")
    parser.add_argument('--input', type=str, required=True, help="Input file with scenarios (to pickle from)")
    parser.add_argument('--model', type=str, required=True, help="Model type", choices=["mlp","gp","rf"])
    #parser.add_argument('--nrandom', type=int, required=True, help="Number of random samples")
    args = parser.parse_args()
    fnm_model_output = args.model_output
    with open(fnm_model_output, "rb") as f:
        [_,parameters,time_train,y_name,miny,maxy,ys,_,df] = pickle.load(f)

    t0_train = time_train[0]

    fnm_in = args.input
    with open(fnm_in, "rb") as f:
        _,_,scenarios = pickle.load(f)
    start_year = 1970
    end_year   = 2300
    time0 = 1992
    print(time_train)
    time = scenarios['rcp26'].loc[start_year:end_year].index
    print(time)
    rcp26 = scenarios['rcp26'].loc[start_year:end_year]
    rcp85 = scenarios['rcp85'].loc[start_year:end_year]

    #with open("./models/gp_exact.pkl", "rb") as f:
    fnm_model = f"./models/{args.model}.pkl"
    with open(fnm_model, "rb") as f:
        model = pickle.load(f)
    def model_update(c,scen):
        X = np.zeros((nt,n_params+3))
        this_forc = scen["global_mean_temperature"]
        x1 = this_forc
        x2 = this_forc.cumsum()
        x2 -= x2.loc[t0_train] # cumulative warming starts with first years of training data
        x3 = (this_forc.groupby((this_forc != this_forc.shift(1)).cumsum()).cumcount()+1)*dt # years since last temperature change
        for i,t in enumerate(time):
            X[i,:n_params] = c
            X[i,n_params:] = [x1.loc[t],x2.loc[t],x3.loc[t]]
        y_pred = model.predict(X)
        idx_diff = time0-start_year
        y_pred = y_pred - y_pred[idx_diff]

        return np.array(y_pred).flatten()

    # custom scenarios (linear warming from 2000 to 2100)
    other_scenarios = {}
    temp_2000 = rcp85["global_mean_temperature"].loc[2000]
    temp_start = rcp85["global_mean_temperature"].loc[start_year]

    dt = time[1]-time[0]

    nt = len(rcp26)
    n_params = len(parameters)

    uncert_factor = 3.
    # Antarctica only; https://www.nature.com/articles/s41586-020-2591-3/tables/1
    t_start = time==1993
    t_end = time==2018
    dslr_obs_max  = 0.44*25.*1e-3 # x25yr because value is goven as rate (mm/yr)
    dslr_obs_min  = 0.21*25.*1e-3
    # Rignot et al (2019)
    ## https://www.pnas.org/content/116/4/1095#T2
    #t_start = time==1979
    #t_end = time==2017
    #dslr_obs_max = (13.9+uncert_factor*2.0)*1e-3
    #dslr_obs_min = (13.9-uncert_factor*2.0)*1e-3
    ## IMBIE team; abstract; https://www.nature.com/articles/s41586-018-0179-y
    t_start = time==1992
    t_end = time==2017
    dslr_obs_mean  = (7.6)
    dslr_obs_max  = (7.6+uncert_factor*3.9)
    dslr_obs_min  = (7.6-uncert_factor*3.9)
    dslr_unc      = uncert_factor*3.9

    target = dslr_obs_mean*1e-3

    # plot median and 2.5-95% CI from ensembles
    n = int(len(ys)/2)
    time_train = time_train[time_train<=time[-1]]
    ys = np.array(ys)
    print(ys.shape)
    y_rcp26 = ys[:n,:len(time_train)]
    y_rcp85 = ys[n:,:len(time_train)]

    df_hist_matched = pd.read_csv(f"data/processed/{args.model}_emulator_runs_pism_matched.csv",index_col=0).T
    print(df_hist_matched.index)

    print(df)
    X = []
    Y = []
    for i,row in df[df["scenario"]=="rcp26"][parameters].iterrows():
        y1 = model_update(row.values,rcp26)
        diff = y1[time==2017][0] - y1[time==1992][0]
        X.append(i)
        Y.append(1.e3*np.abs(diff-target))

    Y = np.array(Y)
    idx = np.argsort(Y)

    fig,ax = plt.subplots(1,1,figsize=(3,8))
    n_best = 0 # 0 - strictly just the combinations that match historical constraints
    for m,i in enumerate(idx):
        color = 'lightgray'
        if m < n_best or (X[i]+1 in df_hist_matched.index.values.astype(int)):
            color = 'lightgreen'
            if X[i]+1 in df_hist_matched.index.values.astype(int):
                color = 'forestgreen'
                n_best += 1
        ax.barh("%d"%(X[i]+1),Y[i],color=color) # expid +1
    ax.axvline(np.mean(Y),lw=1,color='k',zorder=10)
    ax.text(np.mean(Y), 20, 'mean', rotation=90, va='center', ha='center', backgroundcolor='w',zorder=11)
    ax.axvline(dslr_unc,lw=1,color='k',zorder=10)
    ax.text(dslr_unc, 35, 'IMBIE (2018)', rotation=90, va='center', ha='center', backgroundcolor='w',zorder=11)
    ax.yaxis.set_tick_params(labelsize=6)
    ax.set_xscale('log')
    ax.set_xlabel("Absolute error (in mm)")
    ax.set_ylabel("Experiment ID")
    ax.set_ylim(-1,81)#Y[idx[0]],Y[idx[-1]])

    rcp26_matches = idx[:n_best]
    print(rcp26_matches+1)
    df_matched = df[df["scenario"]=="rcp26"][parameters].iloc[rcp26_matches]
    print(df_matched)
    matched = df_matched.values

    ## PLOTTING ###
    fig1, axes = plt.subplots(2,3,figsize=(10,4))#,sharex=True,sharey=True)
    fig1.subplots_adjust(wspace=0.05)
    ax10,ax11,ax12,ax20,ax21,ax22 = axes.flatten()

    #rcp26_matches = []
    #for i,row in df_matched.iterrows():
    #    this_df = df[(df[parameters] == row).all(1)]
    #    idx = this_df.index
    #    rcp26_matches.append(idx[0])
    #sys.exit()
    #rcp26_matches = [25,29,38,56,70,74,79]
    #rcp26_matches = [2,4,5,11,13,14,20,22,23,29,31,32,38,40,41,47,49,50,56,58,59,65,67,68,74,76,77,79]

    color0 = "gray"
    color1 = rcp_colors["AR6-RCP-2.6"]
    color2 = rcp_colors["AR6-RCP-8.5"]
    for i in range(n):
        label1 = None
        label2 = None
        if i == rcp26_matches[0]:
            label1 = "RCP 2.6 (all)"
            label2 = "RCP 8.5 (all)"
        ax11.plot(time_train,1e3*np.gradient(y_rcp26[i,:]),c=color0,alpha=0.5,lw=0.75,zorder=-10,label=label1)
        ax21.plot(time_train,1e3*np.gradient(y_rcp85[i,:]),c=color0,alpha=0.5,lw=0.75,zorder=-10,label=label2)
        ax12.plot(time_train,y_rcp26[i,:],c=color0,alpha=0.5,lw=0.75,zorder=-10,label=label1)
        ax22.plot(time_train,y_rcp85[i,:],c=color0,alpha=0.5,lw=0.75,zorder=-10,label=label2)
    for i in rcp26_matches:
        label1 = None
        label2 = None
        if i == rcp26_matches[0]:
            label1 = "RCP 2.6 (match)"
            label2 = "RCP 8.5 (match)"
        ax11.plot(time_train,1e3*np.gradient(y_rcp26[i,:]),c=color1,alpha=0.85,lw=1,zorder=10,label=label1)
        ax21.plot(time_train,1e3*np.gradient(y_rcp85[i,:]),c=color2,alpha=0.85,lw=1,zorder=10,label=label2)
        ax12.plot(time_train,y_rcp26[i,:],c=color1,alpha=0.85,lw=1,zorder=10,label=label1)
        ax22.plot(time_train,y_rcp85[i,:],c=color2,alpha=0.85,lw=1,zorder=10,label=label2)
    for i,c in enumerate(matched):
        label1 = None
        label2 = None
        if i == 0:
            label1 = "RCP 2.6 (emulator)"
            label2 = "RCP 8.5 (emulator)"
        y1 = model_update(c,rcp26)
        ax10.plot(time,1e3*(y1-y1[time==2017]),ls='--',c=color1,alpha=0.75,lw=1,zorder=-10,label=label1)
        #ax12.plot(time,y1-y1[time==2018],c=color1,ls='--',alpha=0.5,lw=0.75,zorder=-10)
        y2 = model_update(c,rcp85)
        ax20.plot(time,1e3*(y2-y2[time==2017]),ls='--',c=color2,alpha=0.75,lw=1,zorder=-10,label=label2)
        #ax22.plot(time,y2-y2[time==2018],c=color2,ls='--',alpha=0.5,lw=0.75,zorder=-10)

    ax10.set_xlim(1990,2019)
    ax20.set_xlim(1990,2019)
    ax11.set_xlim(2018,2102)
    ax21.set_xlim(2018,2102)
    ax10.set_ylabel("SLR\n(in mm/a)")
    ax20.set_ylabel("SLR\n(in mm/a)")
    ax11.set_ylabel("$\Delta$ SLR\n(in mm/a)")
    ax21.set_ylabel("$\Delta$ SLR\n(in mm/a)")
    ax12.set_ylabel("SLR\n(in m)")
    ax22.set_ylabel("SLR\n(in m)")
    #ax12.set_xlim(2100,2300)
    #ax22.set_xlim(2100,2300)

    time = np.arange(1992,2018)
    y_min = (time-time[time==2017])*dslr_obs_min/25.
    y_max = (time-time[time==2017])*dslr_obs_max/25.
    color3 = "black"
    ax10.fill_between(time,y_min,y_max,color=color3,alpha=0.25,lw=0,zorder=-10,label='IMBIE (2018)')
    ax20.fill_between(time,y_min,y_max,color=color3,alpha=0.25,lw=0,zorder=-10,label='IMBIE (2018)')
    fs = 8
    ax10.legend(loc=2,fontsize=fs)
    ax20.legend(loc=2,fontsize=fs)
    ax11.legend(loc=2,fontsize=fs)
    ax21.legend(loc=2,fontsize=fs)
    ax12.legend(loc=2,fontsize=fs)
    ax22.legend(loc=2,fontsize=fs)
    #ax10.plot(time,y_max,'k--',lw=1)
    #ax20.plot(time,y_min,'k--',lw=1)
    #ax20.plot(time,y_max,'k--',lw=1)
    miny0 = -22
    maxy0 = 15
    ax10.set_ylim(miny0,maxy0)
    ax20.set_ylim(miny0,maxy0)
    miny1 = -5
    maxy1 = 20
    ax11.set_ylim(miny1,maxy1)
    ax21.set_ylim(miny1,maxy1)
    miny2 = -0.3
    maxy2 = 4.7
    ax12.set_ylim(miny2,maxy2)
    ax22.set_ylim(miny2,maxy2)

    ## hide the spines between ax and ax2
    #ax11.spines["right"].set_visible(False)
    #ax12.spines["left"].set_visible(False)
    #ax21.spines["right"].set_visible(False)
    #ax22.spines["left"].set_visible(False)
    #ax12.yaxis.tick_right()
    #ax22.yaxis.tick_right()
    ##ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ##ax2.xaxis.tick_bottom()

    #ax1.plot(time_train,np.percentile(y_rcp26,50,axis=0),ls='--',c='k',alpha=0.75,lw=1,zorder=10,label='PISM (median)')
    #ax1.plot(time_train,np.percentile(y_rcp85,50,axis=0),ls='--',c='k',alpha=0.75,lw=1,zorder=10)
    #for pct in [0]:#[2.5,5,10,25]:
    #    ax1.fill_between(time_train,np.percentile(y_rcp26,pct,axis=0),np.percentile(y_rcp26,100-pct,axis=0),lw=0,color='grey',alpha=0.2,zorder=0)
    #    ax1.fill_between(time_train,np.percentile(y_rcp85,pct,axis=0),np.percentile(y_rcp85,100-pct,axis=0),lw=0,color='grey',alpha=0.2,zorder=0)

    #ax1.plot(time,[np.nan]*len(time),c="C0",lw=1,alpha=0.75,label='RCP2.6')
    #ax1.plot(time,[np.nan]*len(time),c="C1",lw=1,alpha=0.75,label='RCP8.5')
    #for n in range(n_matched):
    #    ax1.plot(time,y1_matched[n],lw=1,color='C0',alpha=0.25,zorder=1)
    #    ax1.plot(time,y2_matched[n],lw=1,color='C1',alpha=0.25,zorder=1)
    #ax1.legend(loc=2)

    fig1.tight_layout()
    fig1.savefig(f"reports/figures/{args.model}_constrain_slr_pism.png",dpi=300, bbox_inches='tight', pad_inches = 0.01)
    fig.savefig(f"reports/figures/{args.model}_constrain_slr_pism_ranking.png",dpi=300, bbox_inches='tight', pad_inches = 0.01)
    plt.show()

if __name__ == "__main__":
    main()

