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

fnms = sys.argv[1:-1]
fig, ax = plt.subplots(1,1)

divider = make_axes_locatable(ax)
# below height and pad are in inches
ax2 = divider.append_axes("right", 1, pad=0.1)
#ax2.spines['right'].set_visible(False)
#ax2.spines['right'].set_visible(True)
#ax2.spines['top'].set_visible(False)
#ax2.spines['left'].set_visible(False)
#ax2.yaxis.set_ticks([])
#ax2.yaxis.set_ticklabels([])
ax2.yaxis.tick_right()
#ax2.spines['bottom'].set_visible(False)
years = [2099,2199,2299]
#ax2.xaxis.set_ticks([y+1 for y in years])
ax2.xaxis.set_ticks([y+1 for y in years])
ticklabels = [y+1 for y in years]
ax2.xaxis.set_ticklabels(ticklabels,rotation=45)
ax2.xaxis.set_ticks_position('bottom')
jiggle = np.linspace(-15,15,len(fnms))

scenarios = {
        "rcp26": "RCP2.6",
        "rcp45": "RCP4.5",
        "rcp60": "RCP6.0",
        "rcp85": "RCP8.5"}

for i in np.arange(1,6,0.25):
    scenarios["%.2fK"%i] = "%.2fK"%i

for i in np.arange(1,6,0.5):
    scenarios["%.2fK"%i] = "%.1fK"%i

for i in np.arange(2020,2101,20):
    scenarios["2K-%d"%i] = "%d"%i

print(scenarios)

for c,fnm in enumerate(fnms):
    label_id = fnm[:-4].split("_")[-1]
    label = scenarios[label_id]
    df = pd.read_csv(fnm,index_col=1)
    df_last = df.set_index("year")
    idx = np.argmin(np.diff(df.index))
    last_year  = df_last.index[idx]
    df = df[np.logical_and(df["year"]<=last_year,df["year"]>2000)].drop("year",axis=1)
    #df = df[df["year"]<last_year].drop("year",axis=1)
    df.index -= mean
    print(df)

    t_range = np.linspace(np.around(min(df.index),1),np.around(max(df.index),1),10)

    new_idx = list(df.index) + list(t_range)
    df = df.reindex(new_idx).sort_index()
    df = df.interpolate(method='index')
    #df = df[~df.index.duplicated()]
    df = df.reindex(list(t_range))
    #df = df.sort_index()
    print(df)


    ax.plot(df.index,df.median(axis=1),color="C%d"%c,alpha=0.75,label=label)
    qs = np.linspace(0.5,0.95,10)
    qs = [0.5,0.90,0.95]
    for q in qs:
        alpha = (1-q)/2.
        print(alpha,1-alpha)
        ax.fill_between(df.index,df.quantile(alpha,axis=1),df.quantile(1-alpha,axis=1),color="C%d"%c,lw=0,alpha=0.2)
    for y in years:
        x = df_last.loc[y]
        ci_lo = np.percentile(x,2.5)
        ci_hi = np.percentile(x,97.5)
        ci_me = np.percentile(x,50)
        ax2.errorbar(y+jiggle[c],ci_me,yerr=[[ci_me-ci_lo],[ci_hi-ci_me]],fmt='.',capsize=2,alpha=0.75,color="C%d"%c)
ax.set_xlabel("Global Warming Level (in K)")
ax.legend(loc=2)
ax.set_ylabel("Sea-level rise (in m)")
#ax.grid()
ax2.set_xlim(ax2.get_xlim()[0]-20,ax2.get_xlim()[1]+20)
#ax2.set_ylim(ax.get_ylim()[0],ax2.get_ylim()[1])
#ax.set_ylim(ax.get_ylim()[0],ax2.get_ylim()[1])
plt.show()
fnm_out = sys.argv[-1]
fig.savefig(fnm_out,dpi=300, bbox_inches='tight', pad_inches = 0.01)
