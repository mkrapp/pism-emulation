import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from scipy.signal import savgol_filter

def load(fnm):
    df = pd.read_csv(fnm,delim_whitespace=True,comment='#',header=None,index_col=0).mean(axis=1)
    return df

def main():
    # here goes the main part

    fig, ax = plt.subplots(1,1)
    order = 3
    pts = 51
    fnms = sys.argv[1:]
    gwl = [1.5,2,3,4]
    cmap = plt.get_cmap("Paired")
    for g in gwl:
        ax.axhline(g,lw=0.5,color='k',alpha=0.5,zorder=0)
    for i,fnm in enumerate(fnms):
        df = load(fnm).loc[:2100]
        df_filtered = df*1.0
        df_filtered.loc[:] = savgol_filter(df_filtered, pts, order)
        mean = df_filtered.loc[1850:1900].mean()

        label, scen = fnm.split("/")[-1].split("_")[3:5]
        color = cmap(i)#"C%d"%i
        l = (df-mean).plot(ax=ax,color=color,label="",alpha=0.25)
        (df_filtered-mean).plot(ax=ax,lw=2,c=color,label="%s (%s)"%(label,scen),alpha=0.75)
        for g in gwl:
            x = df_filtered[df_filtered-mean>=g]
            if len(x)>0:
                year = x.index[0]
                #ax.axvline(year,lw=0.5,color=color,alpha=0.5,zorder=0)
                ax.plot(year,df_filtered.loc[year]-mean,ls='',marker='o',color=color,alpha=0.75,zorder=0)
            else:
                year = "NaN"
            print(g,label,scen,year)

    ax.set_xlabel("year")
    ax.set_ylabel("GSAT")
    ax.set_title("Savitzky–Golay filtered (%d,%d)"%(pts,order))
    ax.legend(loc=2)
    plt.show()
    fnm_out = 'reports/figures/global_warming_levels.png'
    fig.savefig(fnm_out,dpi=300, bbox_inches='tight', pad_inches = 0.01)

if __name__ == "__main__":
    main()

