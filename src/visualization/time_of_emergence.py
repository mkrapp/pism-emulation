import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy.stats import gamma, norm, ttest_ind
import sys

def main():
    # here goes the main part
    fnm1 = sys.argv[1]
    df1 = pd.read_csv(fnm1,index_col=0).drop("GMT",axis=1)
    fnm2 = sys.argv[2]
    df2 = pd.read_csv(fnm2,index_col=0).drop("GMT",axis=1)

    start_year = 2000

    df1 = df1.loc[start_year:]
    df2 = df2.loc[start_year:]

    q = 0.95
    alpha = (1-q)/2.
    df1_ci_lo = df1.quantile(alpha,axis=1)
    df2_ci_lo = df2.quantile(alpha,axis=1)
    alpha = 1 - alpha
    df1_ci_hi = df1.quantile(alpha,axis=1)
    df2_ci_hi = df2.quantile(alpha,axis=1)

    df1_ci95 = df1_ci_hi-df1_ci_lo
    df2_ci95 = df2_ci_hi-df2_ci_lo

    fig, ax = plt.subplots(1,1)

    l, = ax.plot(df1.index,df1.median(axis=1),label='RCP2.6')
    ax.fill_between(df1.index,df1_ci_lo,df1_ci_hi,lw=0,alpha=0.25,color=l.get_color())
    l, = ax.plot(df2.index,df2.median(axis=1),label='RCP8.5')
    ax.fill_between(df2.index,df2_ci_lo,df2_ci_hi,lw=0,alpha=0.25,color=l.get_color())
    p_values = []
    different = []
    for y in df2.index:
        #res = ttest_ind(df1.loc[y],df2.loc[y])
        diff = np.abs(df2.loc[y].median()-df1.loc[y].median())>df1_ci95.loc[y]
        different.append(diff)
        #p_values.append(res.pvalue)
    #ax2 = ax.twinx()
    #ax2.plot(df1.index,p_values,'k-',zorder=0,label="p-values (t-test)")
    #ax2.axhline(0.05,ls='--',lw=1,color="k",alpha=0.25,label='p=0.05')
    #ax3 = ax.twinx()
    idx = np.argmax(np.diff(np.asarray(different)))
    year = df1.index[idx]
    #ax2.axvline(year,ls='--',lw=1,color="k",alpha=0.25,label='ToE = year %d'%year)
    #ax2.fill_between(df1.index,different,lw=0,color='k',alpha=0.25,label=r'CI$_{%d}$ (yr=%d)'%(int(q*100),year))
    ax.axvline(year,lw=2,ls='--',color='k',alpha=0.5,label='ToE (%d%% CI) = %d'%(int(q*100),year),zorder=0)
    #ax2.legend(loc=1)

    # PISM ranges
    fnm_in = sys.argv[3]
    with open(fnm_in, "rb") as f:
        [_,_,time_train,_,_,_,ys,_,_] = pickle.load(f)
    #time_train = time_train[time_train<=time[-1]]
    n = int(len(ys)/2)
    y_rcp26 = ys[:n,:len(time_train)]
    y_rcp85 = ys[n:,:len(time_train)]
    for pct in [0]:#[2.5,5,10,25]:
        ax.fill_between(time_train,np.percentile(y_rcp26,pct,axis=0),np.percentile(y_rcp26,100-pct,axis=0),lw=0,color='grey',alpha=0.2,zorder=0,label='PISM (full ensemble)')
        ax.fill_between(time_train,np.percentile(y_rcp85,pct,axis=0),np.percentile(y_rcp85,100-pct,axis=0),lw=0,color='grey',alpha=0.2,zorder=0)

    ax.legend(loc=2)
    ax.set_ylabel('sea level rise (in m)')
    ax.set_xlabel('year')
    fig.savefig("reports/figures/toe.png",dpi=300, bbox_inches='tight', pad_inches = 0.01)

    plt.show()


if __name__ == "__main__":
    main()

