import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gamma, norm, ttest_ind
from gaussian_process_emulator import GaussianProcessEmulator
import pickle
import sys
from tqdm import tqdm

rng = np.random.default_rng(seed=123)

def main():
    # here goes the main part
    fnm_in = sys.argv[1]
    with open(fnm_in, "rb") as f:
        [forcs,parameters,time,y_name,miny,maxy,ys,X,df] = pickle.load(f)
    fnm_in = sys.argv[2]
    with open(fnm_in, "rb") as f:
        _,_,scenarios = pickle.load(f)


    rcp26 = scenarios['rcp26']
    rcp85 = scenarios['rcp85']
    nparams = len(df.T)
    nt = len(rcp26)
    nruns = int(len(df)/2)
    n_exps = len(df)
    print(nruns)

    nt = len(time)
    n_params = len(parameters)

    fig, ax = plt.subplots(2,1,sharex=True,sharey=True)
    #fig3, ax3 = plt.subplots(2,1)#,sharex=True)
    fig2, ax2 = plt.subplots(1,1,figsize=(8, 6))
    # plot PISM time series
    for i in tqdm(range(n_exps)):
        this_ax = ax[0]
        if i > n_exps/2:
            this_ax = ax[1]
        this_ax.plot(time,ys[i],ls='-',c="r",lw=1,alpha=0.1)

    # create random combinations for PISM parameters
    n_combinations = int(sys.argv[3])
    combinations = rng.uniform(low=0,high=1,size=(n_combinations,nparams-1))
    params = np.array([df[df.columns[1:]].min(axis=0).values,df[df.columns[1:]].max(axis=0).values]).T
    #params[-1,0] = 10
    print(params)
    for i,(pmin,pmax) in enumerate(params):
        combinations[:,i] = pmin + (pmax-pmin) * combinations[:,i]
    print(combinations[0])

    print(df)
    # original combinations
    n_orig_combinations = int(len(df)/2)
    orig_combinations = df.values[:n_combinations,1:]
    print(n_orig_combinations)
    print(orig_combinations[0])

    dt = time[1]-time[0]

    combinations_predict = combinations.tolist()
    allY = []
    origY = []
    #use_decades = range(2017,2301)#[2020,2050,2100,2300]
    use_decades = range(2017,2060)#,2301,10)#[2020,2050,2100,2300]
    use_decades = [2100,2300]
    use_decades = range(2040,2300,20)
    #use_decades = range(2030,2301,20)
    Y_decades = {'RCP2.6': {d: [] for d in use_decades}, 'RCP8.5': {d: [] for d in use_decades}}
    gpe = GaussianProcessEmulator()
    gpe.load("./models/")
    for scen in [rcp26,rcp85]:
        newY = []
        this_forc = scen["global_mean_temperature"]
        x1 = this_forc
        x2 = this_forc.cumsum()
        x2 -= x2.iloc[0]
        x3 = (this_forc.groupby((this_forc != this_forc.shift(1)).cumsum()).cumcount()+1)*dt # years since last temperature change
        for c in tqdm(orig_combinations):
            X = np.zeros((nt,n_params+3))
            for t in range(nt):
                X[t,:n_params] = c
                X[t,n_params:] = [x1.iloc[t],x2.iloc[t],x3.iloc[t]]
            y_pred, _ = gpe.predict(X)
            y_pred = y_pred[:,0]
            origY.append(y_pred)
        for c in tqdm(combinations_predict):
            X = np.zeros((nt,n_params+3))
            for t in range(nt):
                X[t,:n_params] = c
                X[t,n_params:] = [x1.iloc[t],x2.iloc[t],x3.iloc[t]]
            y_pred, _ = gpe.predict(X)
            y_pred = y_pred[:,0]
            for d in use_decades:
                idx = [t for t in range(nt) if time[t] == d][0]-1
                if scen is rcp26:
                    Y_decades['RCP2.6'][d].append(y_pred[idx])
                else:
                    Y_decades['RCP8.5'][d].append(y_pred[idx])
            this_ax = ax[0]
            if scen is rcp85:
                this_ax = ax[1]
            newY.append(y_pred)
            allY.append(y_pred)

        newY = np.array(newY)
        #newY = np.array(origY)
        s = 0
        scen_color = "C2"
        scen_label = "RCP2.6"
        if scen is rcp85:
            scen_color = "C4"
            scen_label = "RCP8.5"
        f_alpha = 0.3
        for p in [50,95]:
            alpha = (100-p)/2
            y1 = np.percentile(newY,alpha,axis=0)
            y2 = np.percentile(newY,100-alpha,axis=0)
            this_ax.fill_between(time,y1,y2,lw=0,color='k',alpha=f_alpha)
            #ax3[0].fill_between(time,y1,y2,color=scen_color,lw=0,alpha=f_alpha)
        this_ax.plot(time,np.percentile(newY,50,axis=0),'k-',lw=2,label="Emulator")
        #ax3[0].plot(time,np.percentile(newY,50,axis=0),ls='-',color=scen_color,lw=2,label=scen_label)
    # PISM ensembles
    Y_decades_pism = {'RCP2.6': {d: [] for d in use_decades}, 'RCP8.5': {d: [] for d in use_decades}}
    for i_rcp26 in tqdm(range(nruns)):
        for d in use_decades:
            #idx = [t for t in this_df.index if time[t] == d][0]
            idx = [t for t,_ in enumerate(time) if time[t] == d][0]
            Y_decades_pism['RCP2.6'][d].append(ys[i_rcp26][idx])
    for i_rcp85 in range(nruns,2*nruns):
        for d in use_decades:
            #idx = [t for t in this_df.index if time[t] == d][0]
            idx = [t for t,_ in enumerate(time) if time[t] == d][0]
            Y_decades_pism['RCP8.5'][d].append(ys[i_rcp85][idx])

    sep = 0.25
    #sl_bins = np.arange(min(Y_decades["RCP8.5"][use_decades[-1]])-sep/2,max(Y_decades["RCP2.6"][use_decades[0]])+sep/2,0.1)
    sl_bins_center = np.arange(miny+sep/2,maxy+sep/2,sep)
    sl_bins = np.arange(miny-sep/2,maxy+sep/2,sep)
    x = np.linspace(miny,maxy,1000)
    lines = {'RCP2.6': [], 'RCP8.5': []}
    plot_empirical_hist = False
    for i,d in enumerate(use_decades[::-1]):
        xscale = 1 - i / (len(use_decades)*1.6)
        print(xscale)
        lw = 1.5 - i / (len(use_decades)*2)
        print(d,ttest_ind(Y_decades["RCP2.6"][d],Y_decades["RCP8.5"][d]))
        for scenario in ["RCP2.6","RCP8.5"]:
            scen_color = "C2"
            if scenario =="RCP8.5":
                scen_color = "C4"
            y = Y_decades_pism[scenario][d]
            #ax3[1].hist(y,bins=sl_bins,density=True,alpha=0.5,label=scenario,color=scen_color)
            loc, scale = norm.fit(y)
            #ax3[1].plot(x,norm.pdf(x,loc,scale),color=scen_color)
            this_label = None
            if i==len(use_decades)-1:
                this_label = scenario
            this_x = x.mean()+xscale*(x-x.mean())
            this_y = i+norm.pdf(x,loc,scale)
            idx = np.argmax(this_y)
            ax2.fill_between(this_x,i,this_y,color=scen_color,lw=0,alpha=0.5,label=this_label)
            #ax2.plot(this_x[idx],this_y[idx],ls='',marker='.',color=scen_color)
            #ax2.plot([this_x[idx],this_x[idx]],[i,this_y[idx]],ls='-',lw=1,alpha=0.5,color=scen_coloR)
            if plot_empirical_hist:
                hist, _ = np.histogram(y, bins=sl_bins, density=True)
                this_hist = i+hist
                this_bin = sl_bins_center.mean()+xscale*(sl_bins_center-sl_bins_center.mean())
                ax2.plot(this_bin,this_hist,color=scen_color,lw=lw,drawstyle="steps",alpha=0.5)
            lines[scenario].append(this_x[idx])
            if d%20==0:
                ax2.text(x.mean()+xscale*(x[0]-x.mean()),i,d,ha='right',va='center',fontsize=8)
        if i==len(use_decades)-1:
            ax2.legend(loc=5)
    y_lines = range(len(use_decades))
    ax2.plot(lines['RCP2.6'],y_lines,lw=1,color="C2",alpha=0.5,zorder=0)
    ax2.plot(lines['RCP8.5'],y_lines,lw=1,color="C4",alpha=0.5,zorder=0)
    x0 = 0
    for x0 in [0,1,2,3,4]:
        ax2.plot([x0,x.mean()+xscale*(x0-x.mean())],[0,i],'k-',lw=0.5,alpha=0.75,zorder=0)
    ax2.set_ylim(-0.05,None)
    #ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    fig2.savefig('reports/figures/gp_%s_emergence.png'%(y_name),dpi=300, bbox_inches='tight', pad_inches = 0.01)


    # draw from both RCP2.6 and 8.5
    do_this = False
    if do_this:
        t0 = time[0]+2
        sl_thres = 0.1
        time_emerge = []
        time_emerge_pism = []
        for i_rcp26 in tqdm(range(n_combinations)):
            for i_rcp85 in range(n_combinations,2*n_combinations):
                idx = np.argmin(np.abs((allY[i_rcp26]-allY[i_rcp85]-sl_thres))[::-1])
                if time[-idx]>t0:
                    time_emerge.append(time[-idx])
        for i_rcp26 in tqdm(range(nruns)):
            for i_rcp85 in range(nruns,2*nruns):
                idx = np.argmin(np.abs((Y[i_rcp26]-Y[i_rcp85]-sl_thres))[::-1])
                if time[-idx]>t0:
                    time_emerge_pism.append(time[-idx])
        time_bins = list(time+0.5)
        ax3[1].hist(time_emerge,bins=time_bins,density=True,alpha=0.5,label='Emulator')
        ax3[1].hist(time_emerge_pism,bins=time_bins,density=True,alpha=0.5,label='PISM')

        a, loc, scale = gamma.fit(time_emerge)
        print(a, loc, scale)
        print("10%",gamma.ppf(0.10,a,loc,scale))
        print("50%",gamma.ppf(0.50,a,loc,scale))
        print("75%",gamma.ppf(0.75,a,loc,scale))
        print("90%",gamma.ppf(0.90,a,loc,scale))
        print("95%",gamma.ppf(0.95,a,loc,scale))
        color="black"
        mode = (a-1)*scale+loc
        mode_val = gamma.pdf(mode,a,loc,scale)
        print(mode,mode_val)
        ax3[1].plot([mode,mode],[0,mode_val],ls='--',c=color,label='mode')
        #median = gamma.ppf(0.500,a,loc,scale)
        #median_val = gamma.pdf(median,a,loc,scale)
        #ax3[1].plot([median,median],[0,median_val],ls='-',c=color,label='median')
        ax3[1].fill_between(time,gamma.pdf(time,a,loc,scale),y2=0,lw=0,color=color,alpha=f_alpha,label='Gamma\n(%.1f,%.1f,%.1f)'%(a,loc,scale))
        time1 = np.linspace(time[0],gamma.ppf(0.95,a,loc,scale),100)
        ax3[1].fill_between(time1,gamma.pdf(time1,a,loc,scale),y2=0,lw=0,color=color,alpha=f_alpha)
        time2 = np.linspace(time[0],gamma.ppf(0.50,a,loc,scale),100)
        ax3[1].fill_between(time2,gamma.pdf(time2,a,loc,scale),y2=0,lw=0,color=color,alpha=f_alpha)

        # plotting stuff
        ax3[1].set_xlabel("year")
        ax3[1].set_ylabel('Pr(RCP8.5>RCP2.6)')
        ax3[1].legend(loc=1)

        fig3.savefig('reports/figures/gp_%s_emergence.png'%(y_name),dpi=300, bbox_inches='tight', pad_inches = 0.01)

    #ax3[0].set_ylabel(y_name)
    #ax3[0].legend(loc=3)

    ax[0].plot(time,np.percentile(ys[:nruns],50,axis=0),'r-',lw=2,label="PISM")
    ax[1].plot(time,np.percentile(ys[nruns:],50,axis=0),'r-',lw=2,label="PISM")
    ax[0].set_title("RCP2.6")
    ax[1].set_title("RCP8.5")
    ax[0].legend(loc=2)
    ax[0].text(-0.1, 0.0, y_name, ha='center', rotation="90", va='center', transform=ax[0].transAxes)
    fig4, ax4 = plt.subplots(1,1,figsize=(8, 6))
    for i in tqdm(range(n_exps)):
        color = "C0"
        if i > n_exps/2:
            color = "C1"
        ax4.plot(ys[i],origY[i],ls='',marker='.',mew=0,c=color,alpha=0.1)
    plt.show()
    fig.savefig('reports/figures/gp_%s.png'%(y_name),dpi=300, bbox_inches='tight', pad_inches = 0.01)

if __name__ == "__main__":
    main()

