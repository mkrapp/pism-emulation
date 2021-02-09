import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import sys
from tqdm import tqdm
from numpy.random import default_rng

rng = default_rng(42)

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
    fnm_in = sys.argv[1]
    with open(fnm_in, "rb") as f:
        [_,parameters,time_train,y_name,miny,maxy,ys,_,_] = pickle.load(f)

    t0_train = time_train[0]

    #miny = -0.05
    #maxy = 0.7

    fnm_in = sys.argv[2]
    with open(fnm_in, "rb") as f:
        _,_,scenarios = pickle.load(f)
    start_year = 1970
    end_year   = 2299
    time0 = 1992
    print(time_train)
    nrandom = int(sys.argv[3])
    time = scenarios['rcp26'].loc[start_year:end_year].index
    print(time)
    rcp26 = scenarios['rcp26'].loc[start_year:end_year]
    rcp85 = scenarios['rcp85'].loc[start_year:end_year]
    # load RCP4.5
    fnm_gcm = "data/external/gmt/global_tas_Amon_NorESM1-M_rcp45_r1i1p1.dat"
    df_gcm = pd.read_csv(fnm_gcm,delim_whitespace=True,comment='#',header=None,index_col=0).mean(axis=1).loc[:2100]
    df_filtered = df_gcm*1.0
    order = 3
    pts = 51
    df_filtered.loc[:] = savgol_filter(df_filtered, pts, order)
    # save the mean as reference for GWL
    mean = df_filtered.loc[1850:1900].mean()
    rcp45 = rcp85*0
    rcp45["global_mean_temperature"] = df_filtered.loc[rcp85.index[0]:2100]
    rcp45["global_mean_temperature"].loc[2100:] = df_filtered.loc[2100]
    # load RCP6.0
    fnm_gcm = "data/external/gmt/global_tas_Amon_NorESM1-M_rcp60_r1i1p1.dat"
    df_gcm = pd.read_csv(fnm_gcm,delim_whitespace=True,comment='#',header=None,index_col=0).mean(axis=1).loc[:2100]
    df_filtered = df_gcm*1.0
    order = 3
    pts = 51
    df_filtered.loc[:] = savgol_filter(df_filtered, pts, order)
    rcp60 = rcp85*0
    rcp60["global_mean_temperature"] = df_filtered.loc[rcp85.index[0]:2100]
    rcp60["global_mean_temperature"].loc[2100:] = df_filtered.loc[2100]
    # custom scenarios (linear warming from 2000 to 2100)
    other_scenarios = {}
    temp_2000 = rcp85["global_mean_temperature"].loc[2000]
    temp_start = rcp85["global_mean_temperature"].loc[start_year]
    for GWL in list(range(1,6)):#+[1.25,1.5,1.75,2.25,2.5,2.75,3.25,3.5,3.75]:
        this_scen = rcp85*0
        #this_scen["global_mean_temperature"].loc[:2000] = rcp85["global_mean_temperature"].loc[:2000]
        dx = 2100 - start_year
        dy = GWL - (temp_start - mean)
        for y in range(start_year,2101):
            this_scen["global_mean_temperature"].loc[y] = temp_start + dy/dx*(y-start_year)
        this_scen["global_mean_temperature"].loc[2100:] = this_scen["global_mean_temperature"].loc[2100]
        #other_scenarios["%.2fK"%GWL] = this_scen
        other_scenarios["%dK"%GWL] = this_scen
    # custom scenarios (same warming level but reached at different decades (2040,2060,2080,2100)
    GWL = 2.
    for decade in [2020,2040,2060,2080,2100]:
        this_scen = rcp85*0
        #this_scen["global_mean_temperature"].loc[:2000] = mean#rcp85["global_mean_temperature"].loc[:2000]
        for y in range(start_year,decade+1):
            dx = decade - start_year
            dy = GWL - (temp_start - mean)
            this_scen["global_mean_temperature"].loc[y] = temp_start + dy/dx*(y-start_year)
        this_scen["global_mean_temperature"].loc[decade:] = this_scen["global_mean_temperature"].loc[decade]
        other_scenarios["%dK-%d"%(GWL,decade)] = this_scen


    fig_gmt,ax_gmt = plt.subplots(1,1)
    scenarios = ["RCP%s"%s for s in ["2.6","8.5","4.5","6.0"]]
    for i,scen in enumerate([rcp26,rcp85,rcp45,rcp60]):
        scen = scen.loc[1000:2110] - 273.15
        ax_gmt.plot(scen.index,scen["global_mean_temperature"],alpha=0.75,lw=2,label=scenarios[i])
    ax_gmt.legend(ncol=2,loc=2)
    ax_gmt.set_xlabel("time [in years]")
    ax_gmt.set_ylabel(u"global mean temperature [in \u2103]")

    fig_gmt2, ax_gmt2 = plt.subplots(1,1)
    fig_gmt3, ax_gmt3 = plt.subplots(1,1)
    for scenario,scen in other_scenarios.items():
        scen = scen.loc[1000:2110] - 273.15
        if "-" in scenario:
            label = "%d"%int(scenario.split("-")[-1])
            ax_gmt2.plot(scen.index,scen["global_mean_temperature"],alpha=0.75,lw=2,label=label)
        else:
            label = u"+%.d\u2103"%int(scenario[:-1])
            ax_gmt3.plot(scen.index,scen["global_mean_temperature"],alpha=0.75,lw=2,label=label)
    ax_gmt2.legend(ncol=2,loc=2)
    ax_gmt2.set_xlabel("time [in years]")
    ax_gmt2.set_ylabel(u"global mean temperature [in \u2103]")
    ax_gmt3.legend(ncol=2,loc=2)
    ax_gmt3.set_xlabel("time [in years]")
    ax_gmt3.set_ylabel(u"global mean temperature [in \u2103]")

    # set same range for y-axis
    ax_ymin = min(ax_gmt.get_ylim()[0],ax_gmt2.get_ylim()[0],ax_gmt3.get_ylim()[0])
    ax_ymax = max(ax_gmt.get_ylim()[1],ax_gmt2.get_ylim()[1],ax_gmt3.get_ylim()[1])
    for this_ax in (ax_gmt,ax_gmt2,ax_gmt3):
        this_ax.set_ylim(ax_ymin,ax_ymax)

    dt = time[1]-time[0]

    nt = len(rcp26)
    n_params = len(parameters)

    print(y_name)
    with open("./models/gp_exact.pkl", "rb") as f:
        gpe = pickle.load(f)
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
        y_pred = gpe.predict(X)
        idx_diff = time0-start_year
        y_pred = y_pred - y_pred[idx_diff]

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
    uncert_factor = 3.
    dslr_obs_max  = (7.6+uncert_factor*3.9)*1e-3
    dslr_obs_min  = (7.6-uncert_factor*3.9)*1e-3
    ddslr_obs_mean = (109)/360.e3 # Gt/a -> m/a
    ddslr_obs_min  = (109 - uncert_factor*56)/360.e3 # Gt/a -> m/a
    ddslr_obs_max  = (109 + uncert_factor*56)/360.e3 # Gt/a -> m/a


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
    y1_matched = []
    y2_matched = []
    y3_matched = []
    y4_matched = []
    y_other_scenarios_matched = {k: [] for k in other_scenarios.keys()}
    def update(p):
        [sia,ssa,q,phi]  = p
        y1 = model_update([sia,ssa,q,phi],rcp26)
        y2 = model_update([sia,ssa,q,phi],rcp85)
        hist_matching = ((dslr_obs_min <= y1[t_end]-y1[t_start] <= dslr_obs_max) and
                       (dslr_obs_min <= y2[t_end]-y2[t_start] <= dslr_obs_max))
        if hist_matching:
            y3 = model_update([sia,ssa,q,phi],rcp45)
            y4 = model_update([sia,ssa,q,phi],rcp60)
            for k,scen in other_scenarios.items():
                y_this_scen = model_update([sia,ssa,q,phi],scen)
                y_other_scenarios_matched[k].append(y_this_scen)
            d["sia"].append(sia)
            d["ssa"].append(ssa)
            d["q"].append(q)
            d["phi"].append(phi)
            y1_matched.append(y1)
            y2_matched.append(y2)
            y3_matched.append(y3)
            y4_matched.append(y4)


    #for _ in tqdm(range(nrandom)):
    #    p = []
    #    for s in ["sia","ssa","q","phi"]:
    #        x = np.random.uniform(low=limits[s][0],high=limits[s][1])
    #        p.append(x)
    #    #update(p)
    #    print(p)
    lhs_params  = lhssample(nrandom,len(d.keys()),centered=False)
    for i,s in enumerate(["sia","ssa","q","phi"]):
        lhs_params[:,i] = (limits[s][1]-limits[s][0])*lhs_params[:,i] + limits[s][0]
    for n in tqdm(range(nrandom)):
        update(lhs_params[n,:])

    ### SAVE matched runs to CSV file
    df_rcp26 = pd.DataFrame({"GMT": rcp26["global_mean_temperature"]} ,index=time)
    df_rcp85 = pd.DataFrame({"GMT": rcp85["global_mean_temperature"]} ,index=time)
    df_rcp45 = pd.DataFrame({"GMT": rcp45["global_mean_temperature"]} ,index=time)
    df_rcp60 = pd.DataFrame({"GMT": rcp60["global_mean_temperature"]} ,index=time)
    for i in range(len(y1_matched)):
        df_rcp26[i] = y1_matched[i]
        df_rcp85[i] = y2_matched[i]
        df_rcp45[i] = y3_matched[i]
        df_rcp60[i] = y4_matched[i]
    df_rcp26.index.name = "year"
    df_rcp85.index.name = "year"
    df_rcp26["GMT"] = rcp26["global_mean_temperature"]
    df_rcp85["GMT"] = rcp85["global_mean_temperature"]
    df_rcp45["GMT"] = rcp45["global_mean_temperature"]
    df_rcp60["GMT"] = rcp60["global_mean_temperature"]
    df_params = pd.DataFrame(d)
    fnm_out = "data/processed/emulator_runs_rcp26.csv"
    df_rcp26.to_csv(fnm_out)
    fnm_out = "data/processed/emulator_runs_rcp85.csv"
    df_rcp85.to_csv(fnm_out)
    fnm_out = "data/processed/emulator_runs_rcp45.csv"
    df_rcp45.to_csv(fnm_out)
    fnm_out = "data/processed/emulator_runs_rcp60.csv"
    df_rcp60.to_csv(fnm_out)
    fnm_out = "data/processed/emulator_runs_parameters.csv"
    df_params.index.name = "run_id"
    df_params.to_csv(fnm_out)
    # save our scenarios
    for k,scen in other_scenarios.items():
        this_df = pd.DataFrame({"GMT": scen["global_mean_temperature"]} ,index=time)
        for i in range(len(y1_matched)):
            this_df[i] = y_other_scenarios_matched[k][i]
        this_df.index.name = "year"
        this_df["GMT"] = scen["global_mean_temperature"]
        fnm_out = "data/processed/emulator_runs_%s.csv"%k
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

    n_matched = len(y1_matched)
    ax1.plot(time,[np.nan]*len(time),c="C0",lw=1,alpha=0.75,label='RCP2.6')
    ax1.plot(time,[np.nan]*len(time),c="C1",lw=1,alpha=0.75,label='RCP8.5')
    ax1r.plot(time,[np.nan]*len(time),c="C0",lw=1,alpha=0.75,label='RCP2.6')
    ax1r.plot(time,[np.nan]*len(time),c="C1",lw=1,alpha=0.75,label='RCP8.5')
    for n in range(n_matched):
        ax1.plot(time,y1_matched[n],lw=1,color='C0',alpha=0.25,zorder=1)
        ax1.plot(time,y2_matched[n],lw=1,color='C1',alpha=0.25,zorder=1)
        ax1r.plot(time,np.gradient(y1_matched[n]),lw=1,color='C0',alpha=0.25,zorder=1)
        ax1r.plot(time,np.gradient(y2_matched[n]),lw=1,color='C1',alpha=0.25,zorder=1)
    #ax1.plot(time,np.percentile(y1_matched,50,axis=0),ls='--',c='C0',alpha=0.75,lw=2,zorder=10,label='emulator (median)')
    #ax1.plot(time,np.percentile(y2_matched,50,axis=0),ls='--',c='C1',alpha=0.75,lw=2,zorder=10,label='emulator (median)')
    ax1.legend(loc=2)
    ax1r.legend(loc=2)

    for i,x in enumerate(columns[1:]):
        for j,y in enumerate(columns[:-1]):
            if i>=j:
                ax2[i,j].plot(d[y],d[x],ls='',marker='.',color=colors[1],alpha=0.5,mew=0)

    fig2.tight_layout()
    fig1.tight_layout()
    fig1.savefig("reports/figures/gp_constrain_slr.png",dpi=300, bbox_inches='tight', pad_inches = 0.01)
    fig1r.savefig("reports/figures/gp_constrain_dslr.png",dpi=300, bbox_inches='tight', pad_inches = 0.01)
    fig2.savefig("reports/figures/gp_constrain_parameter.png",dpi=300, bbox_inches='tight', pad_inches = 0.01)
    fig_gmt.savefig("reports/figures/timeseries_scenarios.png",dpi=300, bbox_inches='tight', pad_inches = 0.01)
    fig_gmt2.savefig("reports/figures/timeseries_linear_scenarios_1.png",dpi=300, bbox_inches='tight', pad_inches = 0.01)
    fig_gmt3.savefig("reports/figures/timeseries_linear_scenarios_2.png",dpi=300, bbox_inches='tight', pad_inches = 0.01)
    plt.show()

if __name__ == "__main__":
    main()

