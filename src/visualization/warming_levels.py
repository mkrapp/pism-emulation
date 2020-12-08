import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import sys

fnm_gcm = "data/external/gmt/global_tas_Amon_NorESM1-M_rcp85_r1i1p1.dat"
df_gcm = pd.read_csv(fnm_gcm,delim_whitespace=True,comment='#',header=None,index_col=0).mean(axis=1).loc[:2100]
df_filtered = df_gcm*1.0
order = 3
pts = 51
df_filtered.loc[:] = savgol_filter(df_filtered, pts, order)
mean = df_filtered.loc[1850:1900].mean()

fnms = sys.argv[1:]
fig, ax = plt.subplots(1,1)

#divider = make_axes_locatable(ax)
## below height and pad are in inches
#ax2 = divider.append_axes("right", 0.1, pad=0.1, sharey=ax)
#ax2.spines['right'].set_visible(False)
#ax2.spines['top'].set_visible(False)
#ax2.spines['left'].set_visible(False)
##ax2.yaxis.set_ticks([])
##ax2.yaxis.set_ticklabels([])
#ax2.spines['bottom'].set_visible(False)
#ax2.xaxis.set_ticks([])
#ax2.xaxis.set_ticklabels([])
#ax2.xaxis.set_ticks_position('bottom')

for fnm in fnms:
    df = pd.read_csv(fnm,index_col=1)
    df_last = df.set_index("year")
    df = df[np.logical_and(df["year"]<=2100,df["year"]>2015)].drop("year",axis=1)
    df.index -= mean

    t_range = np.arange(df.index[0],df.index[-1],0.1)

    new_idx = list(df.index) + list(t_range)
    df = df.reindex(new_idx).sort_index()
    df = df.interpolate(method='index')
    df = df[~df.index.duplicated()]
    df = df.reindex(list(t_range))
    print(df)


    l, = ax.plot(df.index,df.median(axis=1),alpha=0.75)
    qs = np.linspace(0.5,0.95,10)
    qs = [0.5,0.90,0.95]
    for q in qs:
        alpha = (1-q)/2.
        print(alpha,1-alpha)
        ax.fill_between(df.index,df.quantile(alpha,axis=1),df.quantile(1-alpha,axis=1),color=l.get_color(),lw=0,alpha=0.2)
    for y in [2199,2299]:
        x = df_last.loc[y]
        ci_lo = np.percentile(x,2.5)
        ci_hi = np.percentile(x,97.5)
        ci_me = np.percentile(x,50)
        ax.plot([df.index[-1]]*2,[ci_lo,ci_hi],lw=3,alpha=0.75,color=l.get_color())
ax.set_xlabel("Global Warming Level (in K)")
ax.set_ylabel("Sea-level rise (in m)")
plt.show()