import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("src/models/")
from sklearn.metrics import r2_score
#from gaussian_process_emulator import GaussianProcessEmulator, ExponentialDecay
import pickle
import itertools

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
            prog='ranking',
            description='Ranking samples.')
    parser.add_argument('--model_output', type=str, required=True, help="Output from regression model (to pickle from)")
    parser.add_argument('--input', type=str, required=True, help="Input file with scenarios (to pickle from)")
    parser.add_argument('--model', type=str, required=True, help="Model type", choices=["mlp","gp","rf"])
    #parser.add_argument('--nrandom', type=int, required=True, help="Number of random samples")
    args = parser.parse_args()
    fnm_model_output = args.model_output
    with open(fnm_model_output, "rb") as f:
        [forcs,parameters,time,y_name,miny,maxy,ys,X,df] = pickle.load(f)

    fnm_in = args.input
    with open(fnm_in, "rb") as f:
        [df,df_timeseries,scenarios] = pickle.load(f)

    fnm_model = f"./models/{args.model}.pkl"
    with open(fnm_model, "rb") as f:
        model = pickle.load(f)

    #gpe = GaussianProcessEmulator()
    #gpe.load("./models/")
    #mu, stdv = gpe.predict(X)
    #stdv = stdv**0.5
    #mu = mu.squeeze()
    #stdv = stdv.squeeze()
    #print(mu.shape)
    #print(stdv.shape)
    # PLOTTING

    #constrained_expid = [2,4,5,11,13,14,16,20,22,23,25,29,31,32,38,40,41,47,49,50,56,59,60,65,67,68,70,74,76,79]
    #constrained_expid = 29 79 74 25 70 56 38
    df_hist_matched = pd.read_csv(f"data/processed/{args.model}_emulator_runs_pism_matched.csv",index_col=0).T
    constrained_expid = df_hist_matched.index.values.astype(int)

    sia_values = [1.2,2.4,4.8]
    ssa_values = [0.425,0.6,0.8]
    q_values   = [0.25,0.5,0.75]
    phi_values = [5,10,15]
    combinations = set(itertools.product(*[sia_values,ssa_values,q_values,phi_values]))
    combinations = {tuple(c) for c in df[df["scenario"]=="rcp26"].iloc[constrained_expid-1][parameters].values}
    print(combinations)
    #for c in combinations:
    #    sia,ssa,q,phi = c
    #    if phi == 15:
    #        combinations = combinations.difference(set([c]))
    #    if (q in [0.25]) and (phi == 5):
    #        combinations = combinations.difference(set([c]))
    #    if (q in [0.75]) and (phi == 10):
    #        combinations = combinations.difference(set([c]))
    #    if (q == 0.75) and (ssa in [0.425,0.6]):
    #        combinations = combinations.difference(set([c]))
    #    if (q == 0.75) and (sia in [1.2,2.4]):
    #        combinations = combinations.difference(set([c]))
    #print(combinations)
    #print(len(combinations))
    ##valid_expid = {"sia":[],"ssa":[],"q":[],"phi":[]}
    #this_df = df[["sia","ssa","q","phi"]]
    #valid_expid = []
    #for c in combinations:
    #    sia,ssa,q,phi = c
    #    valid_expid.append((this_df[this_df == c].dropna().index+1).values)
    ##    valid_expid["sia"].append(sia)
    ##    valid_expid["ssa"].append(ssa)
    ##    valid_expid["q"].append(q)
    ##    valid_expid["phi"].append(phi)
    ##df_valid_expid = pd.DataFrame(valid_expid)
    ##print(df)
    #valid_expid = np.sort(np.array(valid_expid),axis=0)
    #print(valid_expid[:,0])
    #constrained_expid = valid_expid.flatten()
    #sys.exit()

    fig, axes = plt.subplots(9,9,sharex=True,sharey=True,figsize=(14,8))
    axes = axes.flatten()

    nt = len(time)
    n_expid = df_timeseries.columns.get_level_values(0).unique().values
    #n_expid = np.delete(n_expid, [5,86,32,113,59,140])
    n = int(len(n_expid)/2)
    slr26_constrained_emu = []
    slr85_constrained_emu = []
    slr26_constrained = []
    slr85_constrained = []
    for i in range(n):
        label1 = None
        label2 = None
        label1 = "RCP2.6"
        label2 = "RCP8.5"
        # RCP2.6
        this_X = X[i*nt:(i+1)*nt,:]
        #y_pred, y_pred_std = gpe.predict(this_X, return_std=True)
        y_pred = model.predict(this_X)
        y_pred = y_pred.flatten()
        #y_pred_std = y_pred_std.flatten()
        #y_pred = y_pred[:,0]
        #y_pred_std = y_pred_std[:,0]
        r2 = r2_score(ys[i],y_pred)
        x = time
        #x = this_X[:,-2]
        slr26 = y_pred
        l, = axes[i].plot(x,y_pred-ys[i],color=rcp_colors["AR6-RCP-2.6"],alpha=0.75,lw=2,label=label1)
        #l, = axes[i].plot(x,ys[i])
        #axes[i].plot(x,y_pred,ls='--',c=l.get_color())
        #axes[i].fill_between(x,y_pred-y_pred_std,y_pred+y_pred,lw=0,alpha=0.25,color=l.get_color())
        fw = "normal"
        if r2 < 0.3:
            fw = "bold"
        #axes[i].text(0.05,0.6,"%.2f"%r2,fontsize=6,fontweight=fw,color=l.get_color(),transform=axes[i].transAxes)
        # RCP8.5
        this_X = X[(i+n)*nt:(i+n+1)*nt,:]
        #y_pred, y_pred_std = model.predict(this_X, return_std=True)
        y_pred = model.predict(this_X)
        y_pred = y_pred.flatten()
        #y_pred_std = y_pred_std.flatten()
        slr85 = y_pred
        #y_pred = y_pred[:,0]
        #y_pred_std = y_pred_std[:,0]
        r2 = r2_score(ys[i+n],y_pred)
        x = time
        #x = this_X[:,-2]
        l, = axes[i].plot(x,y_pred-ys[i+n],color=rcp_colors["AR6-RCP-8.5"],alpha=0.75,lw=2,label=label2)
        axes[i].axhline(0,ls='-',color='gray',lw=1,zorder=-10)
        #l, = axes[i].plot(x,ys[i+n])
        #axes[i].plot(x,y_pred,ls='--',c=l.get_color())
        #axes[i].fill_between(x,y_pred-y_pred_std,y_pred+y_pred_std,lw=0,alpha=0.25,color=l.get_color())
        fw = "normal"
        if r2 < 0.3:
            fw = "bold"
        #axes[i].text(0.05,0.8,"%.2f"%r2,fontsize=6,fontweight=fw,color=l.get_color(),transform=axes[i].transAxes)
        fw = "normal"
        if (n_expid[i]+1) in constrained_expid:
            fw = "bold"
            slr26_constrained.append(ys[i])
            slr85_constrained.append(ys[i+n])
            slr26_constrained_emu.append(slr26)
            slr85_constrained_emu.append(slr85)

            #axes[i].text(0.05,0.95,"*",fontsize=10,ha='center',va='center',fontweight="bold",color="k",transform=axes[i].transAxes)
            axes[i].plot(0.1,0.8,ls='',marker="*",markersize=10,color="k",mew=0,transform=axes[i].transAxes)
        axes[i].set_title(",".join([str(p) for p in df.loc[i]][1:]),fontsize=6,fontweight=fw,va='top',loc='right')
        axes[i].set_title(i+1,fontsize=6,fontweight=fw,va='top',loc='left')
        axes[i].set_xlim(2000,2315)
        axes[i].set_xticks([2000,2100,2200,2300])
        axes[i].set_xticklabels([2000,2100,2200,2300],rotation=45)
        #axes[i].set_yscale('symlog',linthresh=0.01)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5,-0.01), ncol= 2)

    slr26_constrained = np.array(slr26_constrained)
    slr85_constrained = np.array(slr85_constrained)
    slr26_constrained_emu = np.array(slr26_constrained_emu)
    slr85_constrained_emu = np.array(slr85_constrained_emu)

    #pd.set_option("display.precision", 2)
    YS = np.array(ys[:n])
    print(YS.shape)
    df_rcp26_all = pd.DataFrame({"min": np.min(YS,axis=0),"median": np.median(YS,axis=0),"max": np.max(YS,axis=0)},index=x)
    df_rcp26_all -= df_rcp26_all.loc[2020]
    df_rcp26 = pd.DataFrame({"min": np.min(slr26_constrained,axis=0),"median": np.median(slr26_constrained,axis=0),"max": np.max(slr26_constrained,axis=0)},index=x)
    df_rcp26 -= df_rcp26.loc[2020]
    df_rcp85 = pd.DataFrame({"min": np.min(slr85_constrained,axis=0),"median": np.median(slr85_constrained,axis=0),"max": np.max(slr85_constrained,axis=0)},index=x)
    df_rcp85 -= df_rcp85.loc[2020]
    YS = np.array(ys[n:])
    print(YS.shape)
    df_rcp85_all = pd.DataFrame({"min": np.min(YS,axis=0),"median": np.median(YS,axis=0),"max": np.max(YS,axis=0)},index=x)
    df_rcp85_all -= df_rcp85_all.loc[2020]
    df_rcp26_emu = pd.DataFrame({"min": np.min(slr26_constrained_emu,axis=0),"median": np.median(slr26_constrained_emu,axis=0),"max": np.max(slr26_constrained_emu,axis=0)},index=x)
    df_rcp26_emu -= df_rcp26_emu.loc[2020]
    df_rcp85_emu = pd.DataFrame({"min": np.min(slr85_constrained_emu,axis=0),"median": np.median(slr85_constrained_emu,axis=0),"max": np.max(slr85_constrained_emu,axis=0)},index=x)
    df_rcp85_emu -= df_rcp85_emu.loc[2020]

    with pd.option_context('display.float_format', '{:0.2f}'.format):
        print("ALL PISM RCP2.6")
        print(df_rcp26_all.loc[[2050,2100,2150,2200,2250,2300]].T)
        print("HC PISM RCP2.6")
        print(df_rcp26.loc[[2050,2100,2150,2200,2250,2300]].T)
        print("HC Emulator RCP2.6")
        print(df_rcp26_emu.loc[[2050,2100,2150,2200,2250,2300]].T)
        print("ALL PISM RCP8.5")
        print(df_rcp85_all.loc[[2050,2100,2150,2200,2250,2300]].T)
        print("HC PISM RCP8.5")
        print(df_rcp85.loc[[2050,2100,2150,2200,2250,2300]].T)
        print("HC Emulator RCP8.5")
        print(df_rcp85_emu.loc[[2050,2100,2150,2200,2250,2300]].T)
    fig.text(-0.01,0.5, "Prediction error: Emulator - PISM (SLR in m)", fontsize=16, ha="center", va="center", rotation=90)
    fig.tight_layout()
    plt.show()
    fnm_out = f'reports/figures/{args.model}_{y_name}_panel.png'
    print(fnm_out)
    fig.savefig(fnm_out,dpi=300, bbox_inches='tight', pad_inches = 0.01)

if __name__ == "__main__":
    main()
