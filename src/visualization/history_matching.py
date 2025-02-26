import argparse
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from numpy.random import default_rng


rcp_colors = {
        'AR6-RCP-8.5': '#980002',
        'AR6-RCP-6.0': '#c37900',
        'AR6-RCP-4.5': '#709fcc',
        'AR6-RCP-2.6': '#003466'}

rng = default_rng(42)

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

def lhssample(n=10,p=2,centered=True):
    x = rng.uniform(size=[n,p])
    for i in range(0,p):
        if centered:
            x[:,i] = (np.argsort(x[:,i])+0.5)/n
        else:
            x[:,i] = (np.argsort(x[:,i])+0.5)/n + (x[:,i]-0.5)/n
    return x

def main():
    # here goes the main part
    parser = argparse.ArgumentParser(
            prog='history_matching',
            description='Run history matching.')
    parser.add_argument('--model_output', type=str, required=True, help="Output from regression model (to pickle from)")
    parser.add_argument('--input', type=str, required=True, help="Input file with scenarios (to pickle from)")
    parser.add_argument('--model', type=str, required=True, help="Model type", choices=["mlp","gp","rf"])
    parser.add_argument('--nrandom', type=int, required=True, help="Number of random samples")
    args = parser.parse_args()
    model_type = args.model
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
    rcps = {"rcp26": rcp26, "rcp85": rcp85}

    #miny = -0.05
    #maxy = 0.7

    nrandom = args.nrandom
    with open("./data/interim/other_scenarios.pkl", "rb") as f:
        [other_scenarios,other_rcps] = pickle.load(f)
    all_rcps = {**rcps, **other_rcps}

    dt = time[1]-time[0]

    nt = len(rcp26)
    n_params = len(parameters)

    print(y_name)
    #with open("./models/gp_exact.pkl", "rb") as f:
    fnm_model = f"./models/{model_type}.pkl"
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
    dslr_obs_max  = (7.6+uncert_factor*3.9)*1e-3
    dslr_obs_min  = (7.6-uncert_factor*3.9)*1e-3
    ##ddslr_obs_mean = (109)/360.e3 # Gt/a -> m/a
    ##ddslr_obs_min  = (109 - uncert_factor*56)/360.e3 # Gt/a -> m/a
    ##ddslr_obs_max  = (109 + uncert_factor*56)/360.e3 # Gt/a -> m/a


    limits = {}
    limits["sia"] = (1.2,4.8)
    limits["ssa"] = (0.42,0.8)
    limits["q"] = (0.25,0.75)
    limits["phi"] = (5,15)

    # plot median and 2.5-95% CI from ensembles
    n = int(len(ys)/2)
    time_train = time_train[time_train<=time[-1]]
    ys = np.array(ys)
    print(ys.shape)
    y_rcp26 = ys[:n,:len(time_train)]
    y_rcp85 = ys[n:,:len(time_train)]


    d = {"sia": [], "ssa": [], "q": [], "phi": []}
    y_matched = {k: [] for k in all_rcps.keys()}
    y_other_scenarios_matched = {k: [] for k in other_scenarios.keys()}
    def update(p,save_output=True):
        [sia,ssa,q,phi]  = p
        y1 = model_update([sia,ssa,q,phi],rcp26)
        y2 = model_update([sia,ssa,q,phi],rcp85)
        hist_matching = ((dslr_obs_min <= y1[t_end]-y1[t_start] <= dslr_obs_max) and
                       (dslr_obs_min <= y2[t_end]-y2[t_start] <= dslr_obs_max))
        if hist_matching:
            if save_output:
                for k,scen in other_scenarios.items():
                    y_this_scen = model_update([sia,ssa,q,phi],scen)
                    y_other_scenarios_matched[k].append(y_this_scen)
                d["sia"].append(sia)
                d["ssa"].append(ssa)
                d["q"].append(q)
                d["phi"].append(phi)
                y_matched["rcp26"].append(y1)
                y_matched["rcp85"].append(y2)
                for k,scen in other_rcps.items():
                    y_this_scen = model_update([sia,ssa,q,phi],scen)
                    y_matched[k].append(y_this_scen)
            return True
        else:
            return False


    #for _ in tqdm(range(nrandom)):
    #    p = []
    #    for s in ["sia","ssa","q","phi"]:
    #        x = np.random.uniform(low=limits[s][0],high=limits[s][1])
    #        p.append(x)
    #    #update(p)
    #    print(p)
    df_matched = []
    df.index += 1 # expid starts with 1
    for i,row in df.iterrows():
        if row.iloc[0] == "rcp26":
            matched = update(row[1:].values,save_output=False)
            if matched:
                df_matched.append(row[1:])
    df_matched = pd.concat(df_matched,axis=1)
    fnm_out = f"data/processed/{model_type}_emulator_runs_pism_matched.csv"
    df_matched.index.name = "expid"
    df_matched.to_csv(fnm_out)

    lhs_params  = lhssample(nrandom,len(d.keys()),centered=False)
    for i,s in enumerate(["sia","ssa","q","phi"]):
        lhs_params[:,i] = (limits[s][1]-limits[s][0])*lhs_params[:,i] + limits[s][0]
    for n in tqdm(range(nrandom)):
        update(lhs_params[n,:])

    ### SAVE matched runs to CSV file
    #df_rcp26 = pd.DataFrame({"GMT": rcp26["global_mean_temperature"]} ,index=time)
    #df_rcp85 = pd.DataFrame({"GMT": rcp85["global_mean_temperature"]} ,index=time)
    #df_rcp45 = pd.DataFrame({"GMT": rcp45["global_mean_temperature"]} ,index=time)
    #df_rcp60 = pd.DataFrame({"GMT": rcp60["global_mean_temperature"]} ,index=time)
    #for i in range(len(y1_matched)):
    #    df_rcp26[i] = y1_matched[i]
    #    df_rcp85[i] = y2_matched[i]
    #    df_rcp45[i] = y3_matched[i]
    #    df_rcp60[i] = y4_matched[i]
    #df_rcp26.index.name = "year"
    #df_rcp85.index.name = "year"
    #df_rcp26["GMT"] = rcp26["global_mean_temperature"]
    #df_rcp85["GMT"] = rcp85["global_mean_temperature"]
    #df_rcp45["GMT"] = rcp45["global_mean_temperature"]
    #df_rcp60["GMT"] = rcp60["global_mean_temperature"]
    #df_params = pd.DataFrame(d)
    #fnm_out = "data/processed/emulator_runs_rcp26.csv"
    #df_rcp26.to_csv(fnm_out)
    #fnm_out = "data/processed/emulator_runs_rcp85.csv"
    #df_rcp85.to_csv(fnm_out)
    #fnm_out = "data/processed/emulator_runs_rcp45.csv"
    #df_rcp45.to_csv(fnm_out)
    #fnm_out = "data/processed/emulator_runs_rcp60.csv"
    #df_rcp60.to_csv(fnm_out)
    for k,scen in all_rcps.items():
        this_df = pd.DataFrame({"GMT": scen["global_mean_temperature"]} ,index=time)
        for i in range(len(y_matched[k])):
            this_df[i] = y_matched[k][i]
        this_df.index.name = "year"
        this_df["GMT"] = scen["global_mean_temperature"]
        fnm_out = f"data/processed/{model_type}_emulator_runs_{k}.csv"
        this_df.to_csv(fnm_out)

    df_params = pd.DataFrame(d)
    df_params.index.name = "run_id"
    fnm_out = f"data/processed/{model_type}_emulator_runs_parameters.csv"
    df_params.to_csv(fnm_out)
    # save our scenarios
    for k,scen in other_scenarios.items():
        this_df = pd.DataFrame({"GMT": scen["global_mean_temperature"]} ,index=time)
        for i in range(len(y_other_scenarios_matched[k])):
            this_df[i] = y_other_scenarios_matched[k][i]
        this_df.index.name = "year"
        this_df["GMT"] = scen["global_mean_temperature"]
        fnm_out = f"data/processed/{model_type}_emulator_runs_{k}.csv"
        this_df.to_csv(fnm_out)

    ### PLOTTING ###
    fig1, ax1 = plt.subplots(1,1)
    fig1r, ax1r = plt.subplots(1,1)
    fig2, ax2 = plt.subplots(3,3)
    columns = ["sia","ssa","q","phi"]
    colors = {0: "#7bc043", 1: "#ee4035"}
    for i,y in enumerate(columns[1:]):
        for j,x in enumerate(columns[:-1]):
            if i>=j:
                ax2[i,j].set_xlim(limits[x])
                ax2[i,j].set_ylim(limits[y])
            else:
                ax2[i,j].remove()
            if i==2:
                ax2[i,j].set_xlabel(x)
            if j==0:
                ax2[i,j].set_ylabel(y)

    ax1.plot(time_train,np.percentile(y_rcp26,50,axis=0),ls='--',c='k',alpha=0.75,lw=1,zorder=10,label='PISM (median)')
    ax1r.plot(time_train,np.percentile(np.gradient(y_rcp26,axis=1),50,axis=0),ls='--',c='k',alpha=0.75,lw=1,zorder=10,label='PISM (median)')
    ax1.plot(time_train,np.percentile(y_rcp85,50,axis=0),ls='--',c='k',alpha=0.75,lw=1,zorder=10)
    ax1r.plot(time_train,np.percentile(np.gradient(y_rcp85,axis=1),50,axis=0),ls='--',c='k',alpha=0.75,lw=1,zorder=10)
    for pct in [0]:#[2.5,5,10,25]:
        ax1.fill_between(time_train,np.percentile(y_rcp26,pct,axis=0),np.percentile(y_rcp26,100-pct,axis=0),lw=0,color='grey',alpha=0.2,zorder=0)
        ax1.fill_between(time_train,np.percentile(y_rcp85,pct,axis=0),np.percentile(y_rcp85,100-pct,axis=0),lw=0,color='grey',alpha=0.2,zorder=0)
        ax1r.fill_between(time_train,np.percentile(np.gradient(y_rcp26,axis=1),pct,axis=0),np.percentile(np.gradient(y_rcp26,axis=1),100-pct,axis=0),lw=0,color='grey',alpha=0.2,zorder=0)
        ax1r.fill_between(time_train,np.percentile(np.gradient(y_rcp85,axis=1),pct,axis=0),np.percentile(np.gradient(y_rcp85,axis=1),100-pct,axis=0),lw=0,color='grey',alpha=0.2,zorder=0)

    n_matched = len(y_matched["rcp26"])
    C0 = rcp_colors["AR6-RCP-2.6"]
    C1 = rcp_colors["AR6-RCP-8.5"]
    ax1.plot(time,[np.nan]*len(time),c=C0,lw=1,alpha=0.75,label='RCP2.6')
    ax1.plot(time,[np.nan]*len(time),c=C1,lw=1,alpha=0.75,label='RCP8.5')
    ax1r.plot(time,[np.nan]*len(time),c=C0,lw=1,alpha=0.75,label='RCP2.6')
    ax1r.plot(time,[np.nan]*len(time),c=C1,lw=1,alpha=0.75,label='RCP8.5')
    for n in range(n_matched):
        ax1.plot(time,y_matched["rcp26"][n],lw=1,color=C0,alpha=0.25,zorder=1)
        ax1.plot(time,y_matched["rcp85"][n],lw=1,color=C1,alpha=0.25,zorder=1)
        ax1r.plot(time,np.gradient(y_matched["rcp26"][n]),lw=1,color=C0,alpha=0.25,zorder=1)
        ax1r.plot(time,np.gradient(y_matched["rcp85"][n]),lw=1,color=C1,alpha=0.25,zorder=1)
    #ax1.plot(time,np.percentile(y1_matched,50,axis=0),ls='--',c=C0,alpha=0.75,lw=2,zorder=10,label='emulator (median)')
    #ax1.plot(time,np.percentile(y2_matched,50,axis=0),ls='--',c=C1,alpha=0.75,lw=2,zorder=10,label='emulator (median)')
    ax1.legend(loc=2)
    ax1r.legend(loc=2)

    for i,x in enumerate(columns[1:]):
        for j,y in enumerate(columns[:-1]):
            if i>=j:
                ax2[i,j].plot(d[y],d[x],ls='',marker='.',color=colors[1],alpha=0.5,mew=0)

    fig2.tight_layout()
    fig1.tight_layout()
    fig1.savefig(f"reports/figures/{model_type}_constrain_slr.png",dpi=300, bbox_inches='tight', pad_inches = 0.01)
    fig1r.savefig(f"reports/figures/{model_type}_constrain_dslr.png",dpi=300, bbox_inches='tight', pad_inches = 0.01)
    fig2.savefig(f"reports/figures/{model_type}_constrain_parameter.png",dpi=300, bbox_inches='tight', pad_inches = 0.01)
    plt.show()

if __name__ == "__main__":
    main()

