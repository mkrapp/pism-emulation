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
        'rcp85': '#980002',
        'rcp60': '#c37900',
        'rcp45': '#709fcc',
        'rcp26': '#003466'}

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

def main():
    # here goes the main part
    start_year = 1970
    end_year = 2300
    order = 3
    pts = 51
    fig_gmt,ax_gmt = plt.subplots(1,1,figsize=(6,3))
    labels = ["RCP%s"%s for s in ["2.6","4.5","6.0","8.5"]]
    for i,scen in enumerate(["rcp26","rcp45","rcp60","rcp85"]):
        fnm_gcm = "data/external/gmt/global_tas_Amon_NorESM1-M_%s_r1i1p1.dat"%scen
        df = pd.read_csv(fnm_gcm,delim_whitespace=True,comment='#',header=None,index_col=0).mean(axis=1).loc[:2100]
        df -= 273.15
        ax_gmt.plot(df.index,df,c=rcp_colors[scen],alpha=0.5,lw=1,zorder=-1)
        df.loc[:] = savgol_filter(df, pts, order)
        for t in range(2100,2120):
            df.loc[t] = df.loc[2100]
        ax_gmt.plot(df.index,df,c=rcp_colors[scen],alpha=0.75,lw=2,label=labels[i])
    ylim0 = ax_gmt.get_ylim()[0]
    #xlim1 = ax_gmt.get_xlim()[1]
    ax_gmt.plot(start_year,ylim0,ls='',marker='^',markersize=10,color='k',mew=0)
    #ax_gmt.plot([start_year,end_year],[ylim0]*2,ls='-',lw=1,color='k')

    ax_gmt.legend(loc=2)
    ylabel = u"GMT (in \u2103)"
    xlabel = "Years"
    ax_gmt.set_ylabel(ylabel)
    ax_gmt.set_xlabel(xlabel)
    #ax_gmt.set_ylim(ylim0,None)
    #ax_gmt.set_xlim(None,xlim1)


    fig_gmt.savefig("reports/figures/timeseries_scenarios_filtered.png",dpi=300, bbox_inches='tight', pad_inches = 0.01)
    plt.show()

if __name__ == "__main__":
    main()

