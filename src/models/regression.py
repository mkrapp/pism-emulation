#import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import r2_score
from gaussian_process_emulator import GaussianProcessEmulator, ExponentialDecay
import pickle

start_idx = 2#13
end_idx = -1#13#53#135#85#134
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
    n_expid_train = n_expid
    n_expid_train = np.delete(n_expid, [5,86,32,113,59,140])
    print(len(n_expid))
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

    dt = time[1]-time[0]


    n_params = len(parameters)
    nt = len(time)
    ys = []
    forcs = []
    X = np.zeros((len(n_expid)*nt,n_params+3)).astype(float)
    y = np.zeros((len(n_expid)*nt)).astype(float)
    Xtrain = np.zeros((len(n_expid_train)*nt,n_params+3)).astype(float)
    ytrain = np.zeros((len(n_expid_train)*nt)).astype(float)
    m = 0
    for i,n in enumerate(n_expid):
        this_forc = df_forcings[df["scenario"].loc[n]]["global_mean_temperature"]#.values.flatten()
        x1 = this_forc
        x2 = this_forc.cumsum()
        #x2 -= x2.iloc[0]
        x3 = (this_forc.groupby((this_forc != this_forc.shift(1)).cumsum()).cumcount()+1)*dt # years since last temperature change
        forcs.append(this_forc.values.flatten())
        this_y = df_timeseries[(n, y_name)].iloc[:].values
        if y_name == dependent_variables[3]:
            this_y *= 1e-18
        this_y -= this_y[0] # set start value to zero
        this_y *= -1.
        #x1 = np.roll(this_y,1)
        #x1[0] = np.nan
        ys.append(this_y)
        this_params = []
        for p in parameters:
            this_params.append(df.loc[n][p])
        for t in range(nt):
            X[i*nt+t,:n_params] = this_params
            #X[i*nt+t,n_params:] = [this_forc.iloc[t],x1.iloc[t],x2[t]]
            #X[i*nt+t,n_params:] = [this_forc.iloc[t],time[t]-time[0]]
            #X[i*nt+t,n_params:] = [x1.iloc[t],x2.iloc[t]]#,time[t]-time[0]]
            #X[i*nt+t,n_params:] = [x1.iloc[t],x2.iloc[t],x3.iloc[t]]
            X[i*nt+t,n_params:] = [x1.iloc[t],x2.iloc[t],x3.iloc[t]]
            y[i*nt+t] = this_y[t]
            if i in n_expid_train:
                Xtrain[m*nt+t,:n_params] = this_params
                Xtrain[m*nt+t,n_params:] = [x1.iloc[t],x2.iloc[t],x3.iloc[t]]
                ytrain[m*nt+t] = this_y[t]
        if i in n_expid_train:
            m += 1


    ys = np.array(ys)
    forcs = np.array(forcs)


    gpe = GaussianProcessEmulator()
    ytrain = np.reshape(ytrain,(-1,1))
    #user_kernels = "White() + SquaredExponential(active_dims=[0,1,2,3]) + SquaredExponential(active_dims=[4]) * ExponentialDecay(active_dims=[5])"
    user_kernels = "Constant() + SquaredExponential(active_dims=[0,1,2,3])"# + SquaredExponential(active_dims=[4]) + SquaredExponential(active_dims=[5,6])"
    user_kernels = "SquaredExponential(active_dims=[0,1,2,3])*SquaredExponential(active_dims=[4,5])"# + SquaredExponential(active_dims=[4]) + SquaredExponential(active_dims=[5,6])"
    user_kernels = "SquaredExponential(active_dims=[0,1,2,3,4,5])"# + SquaredExponential(active_dims=[4]) + SquaredExponential(active_dims=[5,6])"
    gpe.initialize(Xtrain,ytrain,method="user=%s"%user_kernels,maxiter=10000,learning_rate=1e-2,scale_X=True,multiple_kernel_dims=False,test_size=0.01)
    gpe.training()
    gpe.summary()
    #gpe.load("./models/")
    #gpe.summary()
    mu, stdv = gpe.predict(X)
    stdv = stdv**0.5
    mu = mu.squeeze()
    stdv = stdv.squeeze()
    print(mu.shape)
    print(stdv.shape)

    # PLOTTING
    fig, axes = plt.subplots(9,9,sharex=True,sharey=True,figsize=(13,8))
    axes = axes.flatten()

    n = int(len(n_expid)/2)
    for i in range(n):
        # RCP2.6
        l, = axes[i].plot(time,ys[i])
        y_pred = mu[i*nt:(i+1)*nt]
        y_pred_std = stdv[i*nt:(i+1)*nt]
        r2 = r2_score(ys[i],y_pred)
        axes[i].plot(time,y_pred,ls='--',c=l.get_color())
        axes[i].fill_between(time,y_pred-2*y_pred_std,y_pred+2*y_pred,lw=0,alpha=0.25,color=l.get_color())
        fw = "normal"
        if r2 < 0.3:
            fw = "bold"
        axes[i].text(0.05,0.6,"%.2f"%r2,fontsize=6,fontweight=fw,color=l.get_color(),transform=axes[i].transAxes)
        # RCP8.5
        l, = axes[i].plot(time,ys[i+n])
        y_pred = mu[(i+n)*nt:(i+n+1)*nt]
        y_pred_std = stdv[(i+n)*nt:(i+n+1)*nt]
        r2 = r2_score(ys[i+n],y_pred)
        axes[i].plot(time,y_pred,ls='--',c=l.get_color())
        axes[i].fill_between(time,y_pred-2*y_pred_std,y_pred+2*y_pred_std,lw=0,alpha=0.25,color=l.get_color())
        fw = "normal"
        if r2 < 0.3:
            fw = "bold"
        axes[i].text(0.05,0.8,"%.2f"%r2,fontsize=6,fontweight=fw,color=l.get_color(),transform=axes[i].transAxes)
        axes[i].set_title(",".join([str(p) for p in df.loc[i]][1:]),fontsize=6,va='top')
    plt.show()
    # save forcing and coefficients for emulator
    fnm_out = 'data/processed/gp_%s.pkl'%(y_name)
    pickle_this = input("Pickle this? (y/[n])") or "n"
    if pickle_this=="y":
        print('Pickle data into "%s".'%fnm_out)
        gpe.save("./models/")
        with open(fnm_out, "wb") as f:
            pickle.dump([forcs,parameters,time,y_name,ys.min(),ys.max(),ys,X,df],f)
        fig.tight_layout()
        fnm_out = 'reports/figures/gp_%s_panel.png'%(y_name)
        fig.savefig(fnm_out,dpi=300, bbox_inches='tight', pad_inches = 0.01)
    sys.exit()


if __name__ == "__main__":
    main()

