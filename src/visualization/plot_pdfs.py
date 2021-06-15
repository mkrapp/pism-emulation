import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import gamma, norm, ttest_ind
import pickle
import sys
from tqdm import tqdm

rng = np.random.default_rng(seed=123)

def main():
    # here goes the main part
    fig2, ax2 = plt.subplots(1,1,figsize=(8, 6))

    # read emulator runs
    fnms = sys.argv[1:]
    dfs = {}
    for fnm in fnms:
        df = pd.read_csv(fnm,index_col=0).drop("GMT",axis=1)
        name = fnm.split(".")[0].split("_")[-1]
        dfs[name] = df
        nruns = len(df)
        time = df.index

    th = 0.01

    colors = {"rcp26": "C0", "rcp85": "C1", "rcp45": "C2", "rcp60": "C3"}
    colors.update({"1K": "C0", "2K": "C1", "3K": "C2", "4K": "C3", "5K": "C4"})

    allY = []
    use_decades = range(2000,2301,1)
    Y_decades = {k: {d: [] for d in use_decades} for k in dfs.keys()}
    # PISM ensembles
    for i in tqdm(range(nruns)):
        for d in use_decades:
            #idx = [t for t in this_df.index if time[t] == d][0]
            idx = [t for t,_ in enumerate(time) if time[t] == d][0]
            for k,df in dfs.items():
                Y_decades[k][d].append(df.iloc[idx,i])

    miny = -0.5
    maxy = 4.5
    print(miny,maxy)
    x = np.linspace(miny,maxy,1001)
    lines = {k: [] for k in dfs.keys()}
    z     = {k: [] for k in dfs.keys()}
    for i,d in enumerate(use_decades[::-1]):
        xscale = 1 - i / (len(use_decades)*1.6)
        #print(xscale)
        lw = 1.5 - i / (len(use_decades)*2)
        for scenario in dfs.keys():
            scen_color = colors[scenario]
            y = Y_decades[scenario][d]
            loc, scale = norm.fit(y)
            this_label = None
            if i==len(use_decades)-1:
                this_label = scenario
            this_x = x.mean()+xscale*(x-x.mean())
            this_y = norm.pdf(x,loc,scale)
            z[scenario].append(this_y)
            idx = np.argmax(this_y)
            ax2.fill_between(this_x[this_y>th],i,i+this_y[this_y>th]*xscale,color=scen_color,lw=0,alpha=0.5,label=this_label)
            lines[scenario].append(this_x[idx])
            if d%20==0:
                ax2.text(x.mean()+xscale*(x[0]-x.mean()),i,d,ha='right',va='center',fontsize=8)
        if i==len(use_decades)-1:
            ax2.legend(loc=5)
    y_lines = range(len(use_decades))

    for scenario in dfs.keys():
        color = colors[scenario]
        ax2.plot(lines[scenario],y_lines,lw=1,color=color,alpha=0.5,zorder=0)
    x0 = 0
    for x0 in [0,1,2,3,4]:
        ax2.plot([x0,x.mean()+xscale*(x0-x.mean())],[0,i],color='gray',lw=0.5,alpha=0.75,zorder=0)
    ax2.set_ylim(-0.05,None)
    #ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    fig2.savefig('reports/figures/slr_emergence.png',dpi=300, bbox_inches='tight', pad_inches = 0.01)

    # animation
    n_decades = len(use_decades)
    print(n_decades)

#    def init():
#        ax3.set_ylim(-0.1,5)
#        ax3.set_xlim(miny, maxy)
#        del xdata[:]
#        del ydata[:]
#        line1.set_data(xdata, ydata)
#        line2.set_data(xdata, ydata)
#        return line1, line2,

    fig3, ax3 = plt.subplots()
    line1 = ax3.fill_between([], [], lw=0, alpha=0.5, color="C0",label="RCP2.6")
    line2 = ax3.fill_between([], [], lw=0, alpha=0.5, color="C1", label="RCP8.5")
    text1 = ax3.text(0.9,0.3,use_decades[-1],fontsize=16,ha='center',va='center',transform=ax3.transAxes)
#    ax3.grid()
    ax3.set_xlim(miny, maxy)
    ax3.set_ylim(-0.05,3)
#    ax3.set_ylim(-0.05,None)
    ax3.set_yticks([])
    ax3.spines['top'].set_visible(False)
    #ax3.spines['bottom'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    ax3.legend(loc=5)
    ax3.set_xlabel("Sea level contribution (in m)")


    def run(i):
        # update the data
        y1 = z["rcp26"][n_decades-i-1]
        y2 = z["rcp85"][n_decades-i-1]
        ax3.set_xlim(x[0],x[-1])
        ax3.set_xlim(miny, maxy)
        ax3.set_ylim(-0.05,3)

        ax3.collections.clear()
        line1 = ax3.fill_between(x[y1>th], y1[y1>th], lw=0, alpha=0.5, color="C0")
        line2 = ax3.fill_between(x[y2>th], y2[y2>th], lw=0, alpha=0.5, color="C1")
        text1.set_text(use_decades[i])

        return line1, line2, text1

    ani = animation.FuncAnimation(fig3, run, n_decades, interval=50, repeat_delay=1000)#, init_func=init)
    fig3.tight_layout()
    #ani = animation.ArtistAnimation(fig3, ims, interval=50, blit=True, repeat_delay=1000)
    ani.save("pdfs.gif")

    ##th = 0.0
    #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #X, Y = np.meshgrid(x,np.array(use_decades[::-1]))
    #Z1 = np.array(z["RCP2.6"])
    #Z2 = np.array(z["RCP8.5"])
    ##cset = ax.contour(X, Y, Z1, zdir='z', levels=np.arange(0.25,1.1,0.25), offset=0, colors="C0",alpha=0.5)
    ##cset = ax.contour(X, Y, Z2, zdir='z', levels=np.arange(0.25,1.1,0.25), offset=0, colors="C1",alpha=0.5)
    #Z1 = np.where(Z1>=th,Z1,np.nan)
    #surf = ax.plot_surface(X, Y, Z1, color="C0", alpha=0.25, linewidth=0, antialiased=False)
    #Z2 = np.where(Z2>=th,Z2,np.nan)
    #surf = ax.plot_surface(X, Y, Z2, color="C1", alpha=0.25, linewidth=0, antialiased=False)
    #ax.set_box_aspect((3,3,1))#(np.ptp(X), np.ptp(Y), np.ptp(Z1)))
    ## make the panes transparent
    #ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    #ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    #ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ## make the grid lines transparent
    #ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    #ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    #ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    #fig.savefig('reports/figures/slr_emergence3d.png',dpi=300, bbox_inches='tight', pad_inches = 0.01)
    plt.show()


if __name__ == "__main__":
    main()

