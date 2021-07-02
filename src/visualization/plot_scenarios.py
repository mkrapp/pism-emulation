import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import sys

plt.rcParams.update({
    "pdf.fonttype" : 42
    })

rcp_colors = {
        'AR6-RCP-8.5': '#980002',
        'AR6-RCP-6.0': '#c37900',
        'AR6-RCP-4.5': '#709fcc',
        'AR6-RCP-2.6': '#003466'}

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

def main():
    # here goes the main part
    fnm_in = sys.argv[1]
    with open(fnm_in, "rb") as f:
        [_,parameters,time_train,y_name,miny,maxy,ys,_,df] = pickle.load(f)

    t0_train = time_train[0]
    fnm_in = sys.argv[2]
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
    scenarios = ["RCP%s"%s for s in ["2.6","4.5","6.0","8.5"]]
    colors    = [rcp_colors["AR6-RCP-%s"%s] for s in ["2.6","4.5","6.0","8.5"]]
    for i,scen in enumerate([rcp26,rcp45,rcp60,rcp85]):
        scen = scen.loc[1000:2110] - 273.15
        ax_gmt.plot(scen.index,scen["global_mean_temperature"],c=colors[i],alpha=0.75,lw=2,label=scenarios[i])
    ax_gmt.legend(loc=2)
    ylabel = u"Global mean temperature (in \u2103)"
    xlabel = "Years"
    ax_gmt.set_ylabel(ylabel)
    ax_gmt.set_xlabel(xlabel)

    fig_gmt2, ax_gmt2 = plt.subplots(1,1)
    fig_gmt3, ax_gmt3 = plt.subplots(1,1)
    with open("./data/interim/other_scenarios.pkl", "wb") as f:
        pickle.dump([other_scenarios,rcp45,rcp60],f)
    for scenario,scen in other_scenarios.items():
        scen = scen.loc[1000:2110] - 273.15
        if "-" in scenario:
            label = "%d"%int(scenario.split("-")[-1])
            ax_gmt2.plot(scen.index,scen["global_mean_temperature"],alpha=0.75,lw=2,label=label)
        else:
            label = u"+%.d\u2103"%int(scenario[:-1])
            ax_gmt3.plot(scen.index,scen["global_mean_temperature"],alpha=0.75,lw=2,label=label)
    ax_gmt2.legend(loc=2)
    ax_gmt2.set_xlabel(xlabel)
    ax_gmt2.set_ylabel(ylabel)
    ax_gmt3.legend(loc=2)
    ax_gmt3.set_xlabel(xlabel)
    ax_gmt3.set_ylabel(ylabel)

    # set same range for y-axis
    ax_ymin = min(ax_gmt.get_ylim()[0],ax_gmt2.get_ylim()[0],ax_gmt3.get_ylim()[0])
    ax_ymax = max(ax_gmt.get_ylim()[1],ax_gmt2.get_ylim()[1],ax_gmt3.get_ylim()[1])
    for this_ax in (ax_gmt,ax_gmt2,ax_gmt3):
        this_ax.set_ylim(ax_ymin,ax_ymax)

    fig_gmt.savefig("reports/figures/timeseries_scenarios.png",dpi=300, bbox_inches='tight', pad_inches = 0.01)
    fig_gmt2.savefig("reports/figures/timeseries_linear_scenarios_1.png",dpi=300, bbox_inches='tight', pad_inches = 0.01)
    fig_gmt3.savefig("reports/figures/timeseries_linear_scenarios_2.png",dpi=300, bbox_inches='tight', pad_inches = 0.01)

if __name__ == "__main__":
    main()

