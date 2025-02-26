import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import sys
sys.path.append("src/models/")
#from gaussian_process_emulator import GaussianProcessEmulator
from tqdm import tqdm

def read_slater2020():
    fnm = "data/external/41558_2020_893_MOESM2_ESM.xlsx"
    xlsx = pd.ExcelFile(fnm)
    imbie = pd.read_excel(xlsx, 'IMBIE ', index_col=0)*1.e-3 # mm -> m
    ar5mid = pd.read_excel(xlsx, 'AR5 Mid', skiprows=1, index_col=0)
    ar5up  = pd.read_excel(xlsx, 'AR5 Upper', skiprows=1, index_col=0)
    ar5lo = pd.read_excel(xlsx, 'AR5 Lower', skiprows=1, index_col=0)
    return {
            'IMBIE': {
                'time': imbie.index,
                'mean': imbie['Cumulative sea-level contribution (mm)'],
                'stdv': imbie['Cumulative sea-level contribution error (mm)']
                },
            'RCP2.6': {
                'time': ar5up.index,
                'mid': ar5mid['RCP 2.6'],
                'lo': ar5lo['RCP 2.6'],
                'up': ar5up['RCP 2.6']
                },
            'RCP8.5': {
                'time': ar5up.index,
                'mid': ar5mid['RCP 8.5'],
                'lo': ar5lo['RCP 8.5'],
                'up': ar5up['RCP 8.5']
                }
            }

def main():
    # here goes the main part
    fnm_in = sys.argv[1]
    with open(fnm_in, "rb") as f:
        [_,parameters,time_train,y_name,miny,maxy,ys,_,_] = pickle.load(f)

    t0_train = time_train[0]

    #miny = -0.05
    #maxy = 0.7

    #print(gpr.kernel.get_params())
    fnm_in = sys.argv[2]
    with open(fnm_in, "rb") as f:
        _,_,scenarios = pickle.load(f)
    start_year = 1970
    end_year   = 2299
    time0 = 1992
    print(time_train)
    nrandom = int(sys.argv[3])
    time = scenarios['rcp26'].loc[start_year:end_year].index
    print(time)
    rcp26 = scenarios['rcp26'].loc[start_year:end_year]
    rcp85 = scenarios['rcp85'].loc[start_year:end_year]

    dt = time[1]-time[0]


    nt = len(rcp26)
    n_params = len(parameters)

    # widgets
    print(y_name)
    with open("./models/gp_exact.pkl", "rb") as f:
        gpe = pickle.load(f)
    #gpe = GaussianProcessEmulator()
    #gpe.load("./models/")
    def model_update(c,scen):
        X = np.zeros((nt,n_params+3))
        this_forc = scen["global_mean_temperature"]
        x1 = this_forc
        x2 = this_forc.cumsum()
        x2 -= x2.loc[t0_train] # cumulative warming starts with first years of training data
        x3 = (this_forc.groupby((this_forc != this_forc.shift(1)).cumsum()).cumcount()+1)*dt # years since last temperature change
        for i,t in enumerate(time):
            X[i,:n_params] = c
            X[i,n_params:] = [x1.loc[t],x2.loc[t],x3.loc[t]]
        y_pred = gpe.predict(X)
        idx_diff = time0-start_year
        y_pred = y_pred - y_pred[idx_diff]

        return np.array(y_pred)

    # Antarctica only; https://www.nature.com/articles/s41586-020-2591-3/tables/1
    t_start = time==1993
    t_end = time==2018
    dslr_obs = 0.44e-3
    # https://www.pnas.org/content/116/4/1095#T2
    t_start = time==1979
    t_end = time==2017
    dslr_obs = 15.9e-3
    # IMBIE team; abstract; https://www.nature.com/articles/s41586-018-0179-y
    t_start = time==1992
    t_end = time==2017
    dslr_obs_mean = 7.6*1e-3
    uncert_factor = 2.
    dslr_obs_max  = (7.6+uncert_factor*3.9)*1e-3
    dslr_obs_min  = (7.6-uncert_factor*3.9)*1e-3
    ddslr_obs_mean = (109)/360.e3 # Gt/a -> m/a
    ddslr_obs_min  = (109 - uncert_factor*56)/360.e3 # Gt/a -> m/a
    ddslr_obs_max  = (109 + uncert_factor*56)/360.e3 # Gt/a -> m/a


    fig1, ax1 = plt.subplots(1,1)
    fig1r, ax1r = plt.subplots(1,1)
    #plt.subplots_adjust(bottom=0.35)
    limits = {}
    sia_val = 2.4
    limits["sia"] = (1.2,4.8)
    ssa_val = 0.6
    limits["ssa"] = (0.42,0.8)
    q_val   = 0.5
    limits["q"] = (0.25,0.75)
    phi_val = 10
    limits["phi"] = (5,15)

    #print(len(rcp26))
    #rcp0 = [rcp26[0]]*len(rcp26)
    y1 = model_update([sia_val,ssa_val,q_val,phi_val],rcp26)
    y2 = model_update([sia_val,ssa_val,q_val,phi_val],rcp85)
    l1, = ax1.plot(time,y1,lw=2,alpha=0.75,label='RCP2.6')
    l2, = ax1.plot(time,y2,lw=2,alpha=0.75,label='RCP8.5')
    l4, = ax1.plot([time[t_start],time[t_end]],[0.0,dslr_obs_mean],'k-',lw=1)
    l4r, = ax1r.plot([time[t_start],time[t_end]],[0.0,ddslr_obs_mean],'k-',lw=1)
    x = [time[t_start],time[t_end],time[t_end],time[t_start],time[t_start]]
    y = [y1[t_start],y1[t_start],y1[t_end],y1[t_end],y1[t_start]]
    #l5, = ax1.plot(x,y,'r-',lw=1)
    l1r, = ax1r.plot(time,np.gradient(y1),lw=2,alpha=0.75,label='RCP2.6')
    l2r, = ax1r.plot(time,np.gradient(y2),lw=2,alpha=0.75,label='RCP8.5')
    ax1.set_ylim(miny,maxy)

    # plot median and 2.5-95% CI from ensembles
    n = int(len(ys)/2)
    time_train = time_train[time_train<=time[-1]]
    ys = np.array(ys)
    print(ys.shape)
    y_rcp26 = ys[:n,:len(time_train)]
    y_rcp85 = ys[n:,:len(time_train)]
    ax1.plot(time_train,np.percentile(y_rcp26,50,axis=0),ls='--',c='k',alpha=0.75,lw=1,zorder=10,label='PISM (median)')
    ax1r.plot(time_train,np.percentile(np.gradient(y_rcp26,axis=1),50,axis=0),ls='--',c='k',alpha=0.75,lw=1,zorder=10,label='PISM (median)')
    #ax1.fill_between(time_train,np.percentile(y_rcp26,2.5,axis=0),np.percentile(y_rcp26,97.5,axis=0),lw=0,alpha=0.1,color=l1.get_color(),zorder=-1)
    ax1.plot(time_train,np.percentile(y_rcp85,50,axis=0),ls='--',c='k',alpha=0.75,lw=1,zorder=10)
    ax1r.plot(time_train,np.percentile(np.gradient(y_rcp85,axis=1),50,axis=0),ls='--',c='k',alpha=0.75,lw=1,zorder=10)
    #ax1.fill_between(time_train,np.percentile(y_rcp85,2.5,axis=0),np.percentile(y_rcp85,97.5,axis=0),lw=0,alpha=0.1,color=l2.get_color(),zorder=-1)
    ax1.fill_between(time_train,np.percentile(y_rcp26,2.5,axis=0),np.percentile(y_rcp26,97.5,axis=0),lw=0,color='grey',alpha=0.2,zorder=0)
    ax1.fill_between(time_train,np.percentile(y_rcp85,2.5,axis=0),np.percentile(y_rcp85,97.5,axis=0),lw=0,color='grey',alpha=0.2,zorder=0)
    #ax1.fill_between(time_train,y_rcp26.percentile(axis=0),y_rcp26.max(axis=0),lw=0,color='grey',alpha=0.1,zorder=0)
    #ax1.fill_between(time_train,y_rcp85.min(axis=0),y_rcp85.max(axis=0),lw=0,color='grey',alpha=0.1,zorder=0)
    ax1r.fill_between(time_train,np.percentile(np.gradient(y_rcp26,axis=1),2.5,axis=0),np.percentile(np.gradient(y_rcp26,axis=1),97.5,axis=0),lw=0,color='grey',alpha=0.2,zorder=0)
    ax1r.fill_between(time_train,np.percentile(np.gradient(y_rcp85,axis=1),2.5,axis=0),np.percentile(np.gradient(y_rcp85,axis=1),97.5,axis=0),lw=0,color='grey',alpha=0.2,zorder=0)
    ax1.legend(loc=2)
    ax1r.legend(loc=2)

    # plot imbie
    #xlsx = read_slater2020()
    #df_imbie = xlsx['IMBIE']
    #l, = ax1.plot(df_imbie['time'],df_imbie['mean'],c='k',zorder=5)
    #ax1.fill_between(df_imbie['time'],df_imbie['mean']-df_imbie['stdv'],df_imbie['mean']+df_imbie['stdv'],lw=0,alpha=0.25,color=l.get_color())
    #df_rcp26 = xlsx['RCP2.6']
    #df_rcp85 = xlsx['RCP8.5']
    #l, = ax1.plot(df_rcp26['time'],df_rcp26['mid'],ls='--',c=l1.get_color(),zorder=2)
    #ax1.fill_between(df_rcp26['time'],df_rcp26['lo'],df_rcp26['up'],lw=0,alpha=0.25,color=l.get_color())
    #l, = ax1.plot(df_rcp85['time'],df_rcp85['mid'],ls='--',c=l2.get_color(),zorder=2)
    #ax1.fill_between(df_rcp85['time'],df_rcp85['lo'],df_rcp85['up'],lw=0,alpha=0.25,color=l.get_color())

    fig3 = plt.figure(figsize=(5,3))
    axsia = fig3.add_axes([0.1, 0.2, 0.8, 0.1])
    axssa = fig3.add_axes([0.1, 0.4, 0.8, 0.1])
    axq   = fig3.add_axes([0.1, 0.6, 0.8, 0.1])
    axphi = fig3.add_axes([0.1, 0.8, 0.8, 0.1])
    sl_sia = Slider(axsia, 'sia', limits["sia"][0], limits["sia"][1], valinit=sia_val, valstep=0.01)
    sl_ssa = Slider(axssa, 'ssa', limits["ssa"][0], limits["ssa"][1], valinit=ssa_val, valstep=0.01)
    sl_q   = Slider(axq, 'q', limits["q"][0], limits["q"][1], valinit=q_val, valstep=0.01)
    sl_phi = Slider(axphi, 'phi', limits["phi"][0], limits["phi"][1], valinit=phi_val, valstep=0.1)

    #hist_matching = ((dslr_obs_min <= y1[t_end]-y1[t_start] <= dslr_obs_max) and
    #               (dslr_obs_min <= y2[t_end]-y2[t_start] <= dslr_obs_max) and
    #               (ddslr_obs_min <= np.gradient(y1)[t_end]-np.gradient(y1)[t_start] <= ddslr_obs_max) and
    #               (ddslr_obs_min <= np.gradient(y2)[t_end]-np.gradient(y2)[t_start] <= ddslr_obs_max))
    #hist_matching = ((dslr_obs_min <= y1[t_end]-y1[t_start] <= dslr_obs_max) and
    #               (dslr_obs_min <= y2[t_end]-y2[t_start] <= dslr_obs_max))
    #if hist_matching:
    #    label = 1
    #else:
    #    label = 0
    d = {"sia": sia_val, "ssa": ssa_val, "q": q_val, "phi": phi_val}#, "label": label}
    fig2, ax2 = plt.subplots(3,3)
    columns = ["sia","ssa","q","phi"]
    colors = {0: "#7bc043", 1: "#ee4035"}
    for i,y in enumerate(columns[1:]):
        for j,x in enumerate(columns[:-1]):
            if i>=j:
                #ax2[i,j].plot(d[x],d[y],ls='',marker='.',color=colors[d["label"]],alpha=0.5,mew=0)
                ax2[i,j].set_xlim(limits[x])
                ax2[i,j].set_ylim(limits[y])
            else:
                ax2[i,j].remove()
            if i==2:
                ax2[i,j].set_xlabel(x)
            if j==0:
                ax2[i,j].set_ylabel(y)
    fig2.tight_layout()

    y1_other  = []
    y2_other  = []

    def update(val):
        sia = sl_sia.val
        ssa = sl_ssa.val
        q   = sl_q.val
        phi = sl_phi.val
        y1 = model_update([sia,ssa,q,phi],rcp26)
        y2 = model_update([sia,ssa,q,phi],rcp85)
        #l5.set_ydata([y1[t_start],y1[t_start],y1[t_end],y1[t_end],y1[t_start]])
        #hist_matching = ((dslr_obs_min <= y1[t_end]-y1[t_start] <= dslr_obs_max) and
        #               (dslr_obs_min <= y2[t_end]-y2[t_start] <= dslr_obs_max) and
        #               (ddslr_obs_min <= np.gradient(y1)[t_end]-np.gradient(y1)[t_start] <= ddslr_obs_max) and
        #               (ddslr_obs_min <= np.gradient(y2)[t_end]-np.gradient(y2)[t_start] <= ddslr_obs_max))
        hist_matching = ((dslr_obs_min <= y1[t_end]-y1[t_start] <= dslr_obs_max) and
                       (dslr_obs_min <= y2[t_end]-y2[t_start] <= dslr_obs_max))
        if hist_matching:
            ax1.plot(time,y1,lw=1,color='C0',alpha=0.25,zorder=1)
            ax1.plot(time,y2,lw=1,color='C1',alpha=0.25,zorder=1)
            ax1r.plot(time,np.gradient(y1),lw=1,color='C0',alpha=0.25,zorder=1)
            ax1r.plot(time,np.gradient(y2),lw=1,color='C1',alpha=0.25,zorder=1)
            label = 1
            d = {"sia": sia, "ssa": ssa, "q": q, "phi": phi, "label": label}
            for i,x in enumerate(columns[1:]):
                for j,y in enumerate(columns[:-1]):
                    if i>=j:
                        ax2[i,j].plot(d[y],d[x],ls='',marker='.',color=colors[d["label"]],alpha=0.5,mew=0)
        #else:
        #    y1_other.append(y1)
        #    y2_other.append(y2)
        #    ax1.collections.clear()
        #    ax1.fill_between(time,np.array(y1_other).min(axis=0),np.array(y1_other).max(axis=0),color='grey',lw=0,alpha=0.1,zorder=0)
        #    ax1.fill_between(time,np.array(y2_other).min(axis=0),np.array(y2_other).max(axis=0),color='grey',lw=0,alpha=0.1,zorder=0)
        #    #ax1.plot(time,y1,lw=0.1,color='grey',alpha=0.01,zorder=0)
        #    #ax1.plot(time,y2,lw=0.1,color='grey',alpha=0.01,zorder=0)
        #    #ax1r.plot(time,np.gradient(y1),lw=0.1,color='grey',alpha=0.01,zorder=0)
        #    #ax1r.plot(time,np.gradient(y2),lw=0.1,color='grey',alpha=0.01,zorder=0)
        #    label = 0
        l1.set_ydata(y1)
        l2.set_ydata(y2)
        l1r.set_ydata(np.gradient(y1))
        l2r.set_ydata(np.gradient(y2))
        fig1.canvas.draw_idle()
        fig2.canvas.draw_idle()
        fig1r.canvas.draw_idle()

    sl_sia.on_changed(update)
    sl_ssa.on_changed(update)
    sl_q.on_changed(update)
    sl_phi.on_changed(update)

    resetax = fig3.add_axes([0.25, 0.025, 0.2, 0.1])
    button = Button(resetax, 'Reset', hovercolor='0.975')
    randax = fig3.add_axes([0.5, 0.025, 0.2, 0.1])
    button_random = Button(randax, 'rand (x%d)'%nrandom, hovercolor='0.975')
    hideax = fig3.add_axes([0.75, 0.025, 0.2, 0.1])
    button_hide = Button(hideax, 'Hide')

    def random(event):
        button_random.set_active(False)
        for _ in tqdm(range(nrandom)):
            for s in [sl_sia,sl_ssa,sl_q,sl_phi]:
                x = np.random.uniform(low=s.valmin,high=s.valmax)
                s.set_val(x)
        button_random.set_active(True)

    def reset(event):
        sl_sia.reset()
        sl_ssa.reset()
        sl_q.reset()
        sl_phi.reset()

    def hide(event):
        if button_hide.label.get_text() == "Hide":
            button_hide.label.set_text("Show")
            l1.set_ydata(None)
            l2.set_ydata(None)
            l4.set_xdata(None)
            l4.set_ydata(None)
            #l5.set_xdata(None)
            #l5.set_ydata(None)
        elif button_hide.label.get_text() == "Show":
            button_hide.label.set_text("Hide")
            update(0)

    button.on_clicked(reset)
    button_random.on_clicked(random)
    button_hide.on_clicked(hide)
    plt.show()
    # save plot
    l1.set_ydata(None)
    l2.set_ydata(None)
    l1r.set_ydata(None)
    l2r.set_ydata(None)
    l4.set_xdata(None)
    l4.set_ydata(None)
    l4r.set_xdata(None)
    l4r.set_ydata(None)
    #l5.set_xdata(None)
    #l5.set_ydata(None)
    fig1.tight_layout()
    #fig1.savefig("reports/figures/gp_constrain_slr.png",dpi=300, bbox_inches='tight', pad_inches = 0.01)
    #fig1r.savefig("reports/figures/gp_constrain_dslr.png",dpi=300, bbox_inches='tight', pad_inches = 0.01)
    #fig2.savefig("reports/figures/gp_constrain_parameter.png",dpi=300, bbox_inches='tight', pad_inches = 0.01)

if __name__ == "__main__":
    main()

