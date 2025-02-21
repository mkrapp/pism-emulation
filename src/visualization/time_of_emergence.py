import argparse
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gamma, norm, ttest_ind
#from pyam.plotting import PYAM_COLORS as rcp_colors
import sys

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
            prog='time_of_emergence',
            description='Plot scenarios.')
    parser.add_argument('--model_output', type=str, required=True, help="Output from regression model (to pickle from)")
    parser.add_argument('--model', type=str, required=True, help="Model type", choices=["mlp","gp","rf"])
    parser.add_argument('--rcp26', type=str, required=True, help="GMT time series of RCP2.6")
    parser.add_argument('--rcp85', type=str, required=True, help="GMT time series of RCP8.5")
    args = parser.parse_args()
    fnm1 = args.rcp26
    df1 = pd.read_csv(fnm1,index_col=0).drop("GMT",axis=1)
    fnm2 = args.rcp85
    df2 = pd.read_csv(fnm2,index_col=0).drop("GMT",axis=1)

    colors = {
            "rcp26": rcp_colors["AR6-RCP-2.6"],
            "rcp45": rcp_colors["AR6-RCP-4.5"],
            "rcp60": rcp_colors["AR6-RCP-6.0"],
            "rcp85": rcp_colors["AR6-RCP-8.5"]
            }

    start_year = 1970

    df1 = df1.loc[start_year:]
    df2 = df2.loc[start_year:]

    # likely ToE
    q1 = 0.68
    alpha = (1-q1)/2.
    df1_ci_lo = df1.quantile(alpha,axis=1)
    df2_ci_lo = df2.quantile(alpha,axis=1)
    alpha = 1 - alpha
    df1_ci_hi = df1.quantile(alpha,axis=1)
    df2_ci_hi = df2.quantile(alpha,axis=1)
    df1_ci68 = df1_ci_hi-df1_ci_lo

    q2 = 0.95
    alpha = (1-q2)/2.
    df1_ci_lo = df1.quantile(alpha,axis=1)
    df2_ci_lo = df2.quantile(alpha,axis=1)
    alpha = 1 - alpha
    df1_ci_hi = df1.quantile(alpha,axis=1)
    df2_ci_hi = df2.quantile(alpha,axis=1)

    df1_ci95 = df1_ci_hi-df1_ci_lo
    df2_ci95 = df2_ci_hi-df2_ci_lo

    fig, ax = plt.subplots(1,1,figsize=(7, 3.5))
    divider = make_axes_locatable(ax)
    ax2 = divider.append_axes("right", 1.3, pad=0.1)
    ax2.yaxis.tick_right()
    # previous SLR estimates for 2300
    #DeConto & Pollard 2016	-0.63 to 1.62	-1.78 to 5.89		4.88 to 11.67
    #Edwards et al 2019 MICI	0.09 to 1.44	0.93 to 4.79		7.11 to 10.77
    #Edwards et al 2019 no MICI	-0.09 to 0.5	0.5 to 1.25		6.86 to 7.09
    #Golledge et al 2015	0.14 to 0.23	0.61 to 0.95	0.9 to 1.36	1.61 to 2.96
    #Bulthuis et al 2019	-0.14 to 0.31	-0.09 to 0.58	-0.04 to 0.96	0.17 to 2.01
    #Bamber et al 2019	-0.11 to 1.56			0.03 to 3.05
    alpha=1.0
    lw=2
    # DeConto & Pollard 2016
    #ax2.plot([0.7]*2,[-0.63,1.62],lw=lw,alpha=alpha,color=colors["rcp26"])
    #ax2.plot([1.1]*2,[1.78,5.89],lw=lw,alpha=alpha,color=colors["rcp45"])
    #ax2.plot([1.3]*2,[4.88,11.67],lw=lw,alpha=alpha,color=colors["rcp85"])
    # Edwards et al 2016 no MICI
    ax2.plot([2.7+8]*2,[-0.09,0.5],lw=lw,alpha=alpha,color=colors["rcp26"])
    ax2.plot([3.1+8]*2,[0.5,1.25],lw=lw,alpha=alpha,color=colors["rcp45"])
    ax2.plot([3.3+8]*2,[6.86,7.09],lw=lw,alpha=alpha,color=colors["rcp85"])
    # Golledge et al 2015
    ax2.plot([4.7]*2,[0.14,0.23],lw=lw,alpha=alpha,color=colors["rcp26"],label="RCP2.6")
    ax2.plot([4.9]*2,[0.61,0.95],lw=lw,alpha=alpha,color=colors["rcp45"],label="RCP4.5")
    ax2.plot([5.1]*2,[0.9,1.36],lw=lw,alpha=alpha,color=colors["rcp60"],label="RCP6.0")
    ax2.plot([5.3]*2,[1.61,2.96],lw=lw,alpha=alpha,color=colors["rcp85"],label="RCP8.5")
    # Bulthuis et al 2019
    ax2.plot([6.7]*2,[-0.14,0.31],lw=lw,alpha=alpha,color=colors["rcp26"])
    ax2.plot([6.9]*2,[-0.09,0.58],lw=lw,alpha=alpha,color=colors["rcp45"])
    ax2.plot([7.1]*2,[-0.04,0.96],lw=lw,alpha=alpha,color=colors["rcp60"])
    ax2.plot([7.3]*2,[0.17,2.01],lw=lw,alpha=alpha,color=colors["rcp85"])
    # Bamber et al 2019
    ax2.plot([8.7]*2,[-0.11,1.56],lw=lw,alpha=alpha,color=colors["rcp26"])
    ax2.plot([9.3]*2,[0.03,3.05],lw=lw,alpha=alpha,color=colors["rcp85"])
    # This study
    df_rcp26 = df1.loc[2300]
    df_rcp85 = df2.loc[2300]
    df_rcp45 = pd.read_csv(f"data/processed/{args.model}_emulator_runs_rcp45.csv",index_col=0).loc[2300]
    df_rcp60 = pd.read_csv(f"data/processed/{args.model}_emulator_runs_rcp60.csv",index_col=0).loc[2300]
    ax2.plot([10.7-8]*2,df_rcp26.quantile([0.025,0.975]),lw=lw,alpha=alpha,color=colors["rcp26"])
    ax2.plot([11.3-8]*2,df_rcp85.quantile([0.025,0.975]),lw=lw,alpha=alpha,color=colors["rcp85"])
    ax2.plot([10.9-8]*2,df_rcp45.quantile([0.025,0.975]),lw=lw,alpha=alpha,color=colors["rcp45"])
    ax2.plot([11.1-8]*2,df_rcp60.quantile([0.025,0.975]),lw=lw,alpha=alpha,color=colors["rcp60"])
    #labels = {1: "DP16", 3: "EDW19", 5: "GOL15", 7: "BUl19", 9: "BAM19"}
    labels = {3+8: "EDW19", 5: "GOL15", 7: "BUl19", 9: "BAM19", 11-8: "LOW21"}
    ax2.xaxis.set_ticks([y for y in labels.keys()])
    ticklabels = [v for k,v in labels.items()]
    ax2.xaxis.set_ticklabels(ticklabels,rotation=45)
    ax2.legend(ncol=1,loc=2)
    ax2.set_title("SLR in 2300",fontsize=10,va="top")

    l, = ax.plot(df1.index,df1.median(axis=1),color=colors["rcp26"],lw=2,label='RCP2.6')
    ax.fill_between(df1.index,df1_ci_lo,df1_ci_hi,lw=0,alpha=0.4,color=l.get_color())
    #ax.fill_between(df1.index,df1.min(axis=1),df1.max(axis=1),lw=0,alpha=0.2,color=l.get_color())
    #ax.plot(df1.index,df1_ci_lo,ls='-',lw=1,alpha=0.75,color=l.get_color())
    #ax.plot(df1.index,df1_ci_hi,ls='-',lw=1,alpha=0.75,color=l.get_color())
    l, = ax.plot(df2.index,df2.median(axis=1),color=colors["rcp85"],lw=2,label='RCP8.5')
    ax.fill_between(df2.index,df2_ci_lo,df2_ci_hi,lw=0,alpha=0.4,color=l.get_color())
    #ax.fill_between(df2.index,df2.min(axis=1),df2.max(axis=1),lw=0,alpha=0.2,color=l.get_color())
    #ax.plot(df2.index,df2_ci_lo,ls='-',lw=1,alpha=0.75,color=l.get_color())
    #ax.plot(df2.index,df2_ci_hi,ls='-',lw=1,alpha=0.75,color=l.get_color())
    p_values = []
    different68 = []
    different95 = []
    for y in df2.index:
        #res = ttest_ind(df1.loc[y],df2.loc[y])
        diff95 = np.abs(df2.loc[y].median()-df1.loc[y].median())>df1_ci95.loc[y]
        different95.append(diff95)
        diff68 = np.abs(df2.loc[y].median()-df1.loc[y].median())>df1_ci68.loc[y]
        different68.append(diff68)
        #p_values.append(res.pvalue)
    #ax2 = ax.twinx()
    #ax2.plot(df1.index,p_values,'k-',zorder=0,label="p-values (t-test)")
    #ax2.axhline(0.05,ls='--',lw=1,color="k",alpha=0.25,label='p=0.05')
    #ax3 = ax.twinx()
    idx68 = np.argmax(np.diff(np.asarray(different68)))
    year68 = df1.index[idx68]
    idx95 = np.argmax(np.diff(np.asarray(different95)))
    year95 = df1.index[idx95]
    #ax2.axvline(year,ls='--',lw=1,color="k",alpha=0.25,label='ToE = year %d'%year)
    #ax2.fill_between(df1.index,different,lw=0,color='k',alpha=0.25,label=r'CI$_{%d}$ (yr=%d)'%(int(q*100),year))
    ax.axvline(year68,lw=1,ls='--',color="dimgray",alpha=0.75)#,label='ToE (%d%% CI) = %d'%(int(q1*100),year68))
    ax.axvline(year95,lw=1,ls='--',color="dimgray",alpha=0.75)#,label='ToE (%d%% CI) = %d'%(int(q2*100),year95))
    ax.text(year68-15,1.7,"%d (likely)"%year68,rotation=90,color="dimgray")
    ax.text(year95-15,3,"%d (very likely)"%year95,rotation=90,color="dimgray")
    #ax2.legend(loc=1)

    # PISM ranges
    fnm_in = args.model_output
    with open(fnm_in, "rb") as f:
        [_,_,time_train,_,_,_,ys,_,_] = pickle.load(f)
    #time_train = time_train[time_train<=time[-1]]
    n = int(len(ys)/2)
    y_rcp26 = ys[:n,:len(time_train)]
    y_rcp85 = ys[n:,:len(time_train)]
    for pct in [0]:#[2.5,5,10,25]:
        ax.fill_between(time_train,np.percentile(y_rcp26,pct,axis=0),np.percentile(y_rcp26,100-pct,axis=0),lw=0,color='grey',alpha=0.25,zorder=0,label='Unconstrained')
        ax.fill_between(time_train,np.percentile(y_rcp85,pct,axis=0),np.percentile(y_rcp85,100-pct,axis=0),lw=0,color='grey',alpha=0.25,zorder=0)

    ax.legend(loc=2)
    ax.set_ylabel('Sea level rise (in m)')
    ax.set_xlabel('Year')
    # set axis limits
    ymin = min(ax.get_ylim()[0],ax2.get_ylim()[0])
    ymax = max(ax.get_ylim()[1],ax2.get_ylim()[1])
    ax2.set_ylim(ymin,ymax)
    ax.set_ylim(ymin,ymax)
    fig.savefig(f"reports/figures/{args.model}_toe.png",dpi=300, bbox_inches='tight', pad_inches = 0.01)
    fig.savefig(f"reports/figures/{args.model}_toe.pdf",dpi=300, bbox_inches='tight', pad_inches = 0.01)

    plt.show()


if __name__ == "__main__":
    main()

