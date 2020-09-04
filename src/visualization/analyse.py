import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import pickle

start_idx = 0
end_idx = -200
np.set_printoptions(precision=3,linewidth=180)

def main():
    # here goes the main part

    fnm = sys.argv[1]
    subset = fnm.split(".")[0].split("_")[1]
    print(subset)
    with open(fnm, "rb") as f:
        [df,df_timeseries,df_forcings] = pickle.load(f)
    print(df)
    time = df_timeseries.iloc[start_idx:end_idx].index
    df_timeseries = df_timeseries.iloc[start_idx:end_idx]
    #df_timeseries -= df_timeseries.iloc[0]
    n_expid = df_timeseries.columns.get_level_values(0).unique().values
    #n_expid = n_expid
    dependent_variables = df_timeseries.columns.get_level_values(1).unique().values
    print(dependent_variables)
    parameters = list(df.columns.get_level_values(0).unique().values)
    parameters.remove("scenario")

    scenarios = df_forcings.columns.get_level_values(0).unique().values
    variables = df_forcings.columns.get_level_values(1).unique().values

    print(time)
    df_forcings = df_forcings.loc[time]# - 273.15

    forcings = list(df_forcings.columns.get_level_values(1).unique().values)
    print(forcings)

    nt = len(time)


    y_name = dependent_variables[0]
    y_name2 = dependent_variables[6]

    dt = time[1]-time[0]

    fig, axes = plt.subplots(9,9,sharex=True,sharey=True,figsize=(13,8))
    axes = axes.flatten()

    n_params = len(parameters)
    nt = len(time)
    n = int(len(n_expid)/2)
    for i in range(n):
        # RCP2.6
        this_forc = df_forcings[df["scenario"].loc[i]]["global_mean_temperature"]#.values.flatten()
        #this_forc -= this_forc.iloc[0]
        this_y = df_timeseries[(i, y_name)].iloc[:].values
        #this_y -= this_y[0] # set start value to zero
        #this_y *= -1.
        x = time
        x = df_timeseries[(i, y_name2)].cumsum().iloc[:].values
        #x = this_forc.cumsum()
        #y_name2 = "global_mean_temperature"
        l, = axes[i].plot(x,this_y)
        # RCP8.5
        this_forc = df_forcings[df["scenario"].loc[i+n]]["global_mean_temperature"]#.values.flatten()
        #this_forc -= this_forc.iloc[0]
        this_y = df_timeseries[(i+n, y_name)].iloc[:].values
        #this_y -= this_y[0] # set start value to zero
        #this_y *= -1.
        x = time
        x = df_timeseries[(i+n, y_name2)].cumsum().iloc[:].values
        #x = this_forc.cumsum()
        l, = axes[i].plot(x,this_y)
        axes[i].set_title(",".join([str(p) for p in df.loc[i]][1:]),fontsize=6,va='top')
    fig.suptitle("%s ~\n"%y_name+r"$\int$%s"%y_name2+"\n%d-%d"%(time[0],time[-1]),y=0.98)

    plt.tight_layout()
    plt.show()
    fnm_out = 'reports/figures/panel_%s-%s_%d-%d.png'%(y_name,y_name2,time[0],time[-1])
    fig.savefig(fnm_out,dpi=300, bbox_inches='tight', pad_inches = 0.01)

if __name__ == "__main__":
    main()
