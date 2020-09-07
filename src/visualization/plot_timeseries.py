import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("src/models/")
from sklearn.metrics import r2_score
from gaussian_process_emulator import GaussianProcessEmulator, ExponentialDecay
import pickle

def main():
    # here goes the main part
    fnm_in = sys.argv[1]
    with open(fnm_in, "rb") as f:
        [forcs,parameters,time,y_name,miny,maxy,ys,X,df] = pickle.load(f)

    fnm_in = sys.argv[2]
    with open(fnm_in, "rb") as f:
        [df,df_timeseries,scenarios] = pickle.load(f)

    gpe = GaussianProcessEmulator()
    gpe.load("./models/")
    #mu, stdv = gpe.predict(X)
    #stdv = stdv**0.5
    #mu = mu.squeeze()
    #stdv = stdv.squeeze()
    #print(mu.shape)
    #print(stdv.shape)
    # PLOTTING

    fig, axes = plt.subplots(9,9,sharex=True,sharey=True,figsize=(13,8))
    axes = axes.flatten()

    nt = len(time)
    n_expid = df_timeseries.columns.get_level_values(0).unique().values
    n_expid = np.delete(n_expid, [5,86,32,113,59,140])
    n = int(len(n_expid)/2)
    for i in range(n):
        # RCP2.6
        this_X = X[i*nt:(i+1)*nt,:]
        y_pred, y_pred_std = gpe.predict(this_X)
        y_pred = y_pred[:,0]
        y_pred_std = y_pred_std[:,0]
        r2 = r2_score(ys[i],y_pred)
        x = time
        #x = this_X[:,-2]
        l, = axes[i].plot(x,ys[i])
        axes[i].plot(x,y_pred,ls='--',c=l.get_color())
        axes[i].fill_between(x,y_pred-y_pred_std,y_pred+y_pred,lw=0,alpha=0.25,color=l.get_color())
        fw = "normal"
        if r2 < 0.3:
            fw = "bold"
        axes[i].text(0.05,0.6,"%.2f"%r2,fontsize=6,fontweight=fw,color=l.get_color(),transform=axes[i].transAxes)
        # RCP8.5
        this_X = X[(i+n)*nt:(i+n+1)*nt,:]
        y_pred, y_pred_std = gpe.predict(this_X)
        y_pred = y_pred[:,0]
        y_pred_std = y_pred_std[:,0]
        r2 = r2_score(ys[i+n],y_pred)
        x = time
        #x = this_X[:,-2]
        l, = axes[i].plot(x,ys[i+n])
        axes[i].plot(x,y_pred,ls='--',c=l.get_color())
        axes[i].fill_between(x,y_pred-y_pred_std,y_pred+y_pred_std,lw=0,alpha=0.25,color=l.get_color())
        fw = "normal"
        if r2 < 0.3:
            fw = "bold"
        axes[i].text(0.05,0.8,"%.2f"%r2,fontsize=6,fontweight=fw,color=l.get_color(),transform=axes[i].transAxes)
        axes[i].set_title(",".join([str(p) for p in df.loc[i]][1:]),fontsize=6,va='top')
    plt.show()
    fig.tight_layout()
    fnm_out = 'reports/figures/gp_%s_panel.png'%(y_name)
    fig.savefig(fnm_out,dpi=300, bbox_inches='tight', pad_inches = 0.01)

if __name__ == "__main__":
    main()
