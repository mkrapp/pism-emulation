import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import sys
sys.path.append("src/models/")
from gaussian_process_emulator import GaussianProcessEmulator
from tqdm import tqdm

def main():
    # here goes the main part
    fnm_in = sys.argv[1]
    with open(fnm_in, "rb") as f:
        [_,parameters,time,y_name,miny,maxy,_,_,_] = pickle.load(f)

    miny = -0.1
    maxy = 0.5

    #print(gpr.kernel.get_params())
    fnm_in = sys.argv[2]
    with open(fnm_in, "rb") as f:
        _,_,scenarios = pickle.load(f)
    #scenarios -= 273.15
    start_year = 1970#2019#1951
    end_year   = 2100#100#180#50
    print(time)
    #maxy = 4.0
    #miny = -0.2
    nrandom = int(sys.argv[3])#1000
    time = scenarios['rcp26'].loc[start_year:end_year].index
    print(time)
    rcp26 = scenarios['rcp26'].loc[start_year:end_year]
    rcp85 = scenarios['rcp85'].loc[start_year:end_year]
    rcp50 = (rcp26+rcp85)/2.
    #rcp50 = rcp26*0. + rcp26.iloc[0]

    #plt.plot(time,rcp26["global_mean_temperature"])
    #plt.plot(time,rcp85["global_mean_temperature"])
    #plt.plot(time,rcp50["global_mean_temperature"])
    #plt.show()
    #sys.exit()
    dt = time[1]-time[0]


    nt = len(rcp26)
    n_params = len(parameters)

    # widgets
    print(y_name)
    gpe = GaussianProcessEmulator()
    gpe.load("./models/")
    def model_update(c,scen):
        X = np.zeros((nt,n_params+3))
        this_forc = scen["global_mean_temperature"]#.values
        x1 = this_forc
        x2 = this_forc.cumsum()
        x2 -= x2.iloc[0]
        x3 = (this_forc.groupby((this_forc != this_forc.shift(1)).cumsum()).cumcount()+1)*dt # years since last temperature change
        for i,t in enumerate(time):
            X[i,:n_params] = c
            X[i,n_params:] = [x1.loc[t],x2.loc[t],x3.loc[t]]
        y_pred, _ = gpe.predict(X)
        y_pred = y_pred[:,0] - y_pred[0,0]

        return np.array(y_pred)

    t_start = time==1976
    t_end = time==2016
    dslr_obs = 0.0139
    #t_start = time==1992
    #t_end = time==2016
    #dslr_obs = 0.02157
    #t_start = time==2007
    #t_end = time==2013
    #dslr_obs = 0.0146

    fig1, ax1 = plt.subplots(1,1)
    plt.subplots_adjust(bottom=0.35)
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
    y3 = model_update([sia_val,ssa_val,q_val,phi_val],rcp50)
    l1, = ax1.plot(time,y1,lw=2,alpha=0.75,label='RCP2.6')
    l2, = ax1.plot(time,y2,lw=2,alpha=0.75,label='RCP8.5')
    #l3, = ax1.plot(time,y3,lw=2,c="C4",alpha=0.75,label='RCP5.0')
    ax1.plot([time[t_start],time[t_end]],[0.0,dslr_obs],'k-',lw=1)
    x = [time[t_start],time[t_end],time[t_end],time[t_start],time[t_start]]
    y = [y1[t_start],y1[t_start],y1[t_end],y1[t_end],y1[t_start]]
    l5, = ax1.plot(x,y,'r-',lw=1)
    ax1.set_ylim(miny,maxy)
    ax1.legend(loc=2)

    axsia = fig1.add_axes([0.1, 0.1, 0.8, 0.03])
    axssa = fig1.add_axes([0.1, 0.15, 0.8, 0.03])
    axq   = fig1.add_axes([0.1, 0.2, 0.8, 0.03])
    axphi = fig1.add_axes([0.1, 0.25, 0.8, 0.03])
    sl_sia = Slider(axsia, 'sia', limits["sia"][0], limits["sia"][1], valinit=sia_val, valstep=0.01)
    sl_ssa = Slider(axssa, 'ssa', limits["ssa"][0], limits["ssa"][1], valinit=ssa_val, valstep=0.01)
    sl_q   = Slider(axq, 'q', limits["q"][0], limits["q"][1], valinit=q_val, valstep=0.01)
    sl_phi = Slider(axphi, 'phi', limits["phi"][0], limits["phi"][1], valinit=phi_val, valstep=0.1)

    if (dslr_obs-0.002 <= y1[t_end]-y1[t_start] <= dslr_obs+0.002):
        label = 1
    else:
        label = 0
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

    def update(val):
        sia = sl_sia.val
        ssa = sl_ssa.val
        q   = sl_q.val
        phi = sl_phi.val
        y1 = model_update([sia,ssa,q,phi],rcp26)
        y2 = model_update([sia,ssa,q,phi],rcp85)
        y3 = model_update([sia,ssa,q,phi],rcp50)
        l5.set_ydata([y1[t_start],y1[t_start],y1[t_end],y1[t_end],y1[t_start]])
        #ax1.plot(time,y1,lw=1,color='grey',alpha=0.01,zorder=0)
        #ax1.plot(time,y2,lw=1,color='grey',alpha=0.01,zorder=0)
        #ax1.plot(time,y3,lw=2,color='C4',alpha=0.01,zorder=0)
        if (dslr_obs-0.002 <= y1[t_end]-y1[t_start] <= dslr_obs+0.002):
            ax1.plot(time,y1,lw=2,color='C0',alpha=0.75,zorder=1)
            ax1.plot(time,y2,lw=2,color='C1',alpha=0.75,zorder=1)
            label = 1
            d = {"sia": sia, "ssa": ssa, "q": q, "phi": phi, "label": label}
            for i,x in enumerate(columns[1:]):
                for j,y in enumerate(columns[:-1]):
                    if i>=j:
                        ax2[i,j].plot(d[y],d[x],ls='',marker='.',color=colors[d["label"]],alpha=0.75,mew=0)
        else:
            ax1.plot(time,y1,lw=1,color='grey',alpha=0.01,zorder=0)
            ax1.plot(time,y2,lw=1,color='grey',alpha=0.01,zorder=0)
            label = 0
        l1.set_ydata(y1)
        l2.set_ydata(y2)
        #l3.set_ydata(y3)
        fig1.canvas.draw_idle()
        fig2.canvas.draw_idle()

    sl_sia.on_changed(update)
    sl_ssa.on_changed(update)
    sl_q.on_changed(update)
    sl_phi.on_changed(update)

    resetax = fig1.add_axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')
    randax = fig1.add_axes([0.60, 0.025, 0.15, 0.04])
    button_random = Button(randax, 'rand (x%d)'%nrandom, hovercolor='0.975')
    hideax = fig1.add_axes([0.45, 0.025, 0.1, 0.04])
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
            #l3.set_ydata(None)
            l5.set_xdata(None)
            l5.set_ydata(None)
        elif button_hide.label.get_text() == "Show":
            button_hide.label.set_text("Hide")
            update(0)

    button.on_clicked(reset)
    button_random.on_clicked(random)
    button_hide.on_clicked(hide)
    plt.show()
    # save plot
    axssa.remove()
    axsia.remove()
    axphi.remove()
    axq.remove()
    resetax.remove()
    randax.remove()
    hideax.remove()
    l1.set_ydata(None)
    l2.set_ydata(None)
    #l3.set_ydata(None)
    l5.set_xdata(None)
    l5.set_ydata(None)
    fig1.tight_layout()
    #fig1.savefig("reports/figures/gp_constrain_slr.png",dpi=300, bbox_inches='tight', pad_inches = 0.01)
    #fig2.savefig("reports/figures/gp_constrain_parameter.png",dpi=300, bbox_inches='tight', pad_inches = 0.01)

if __name__ == "__main__":
    main()

