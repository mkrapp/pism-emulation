import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import gamma, norm, ttest_ind
import pickle
import sys
#from pyam.plotting import PYAM_COLORS as rcp_colors
from tqdm import tqdm

plt.rcParams.update({
    "pdf.fonttype" : 42
    })

rcp_colors = {
        'AR6-RCP-8.5': '#980002',
        'AR6-RCP-6.0': '#c37900',
        'AR6-RCP-4.5': '#709fcc',
        'AR6-RCP-2.6': '#003466'}

rng = np.random.default_rng(seed=123)

def main():
    # here goes the main part
    fig, ax = plt.subplots(1,1,figsize=(7, 3.5))

    # read emulator runs
    fnms = sys.argv[1:]
    dfs = {}
    for fnm in fnms:
        df = pd.read_csv(fnm,index_col=0).drop("GMT",axis=1)
        name = fnm.split(".")[0].split("_")[-1]
        dfs[name] = df
        nruns = len(df)
        time = df.index

    th = 0.05

    colors = rcp_colors#{"rcp26": "C0", "rcp85": "C1", "rcp45": "C2", "rcp60": "C3"}
    colors = {
            "rcp26": rcp_colors["AR6-RCP-2.6"],
            "rcp45": rcp_colors["AR6-RCP-4.5"],
            "rcp60": rcp_colors["AR6-RCP-6.0"],
            "rcp85": rcp_colors["AR6-RCP-8.5"],
            }
    labels = {"rcp26": "RCP2.6", "rcp85": "RCP8.5", "rcp45": "RCP4.5", "rcp60": "RCP6.0"}
    colors.update({"1K": "C0", "2K": "C1", "3K": "C2", "4K": "C3", "5K": "C4"})

    allY = []
    use_decades = range(2020,2301,1)
    plot_decades = range(2050,2301,25)
    Y_decades = {k: {d: [] for d in use_decades} for k in dfs.keys()}
    # PISM ensembles
    for i in tqdm(range(nruns)):
        for d in use_decades:
            #idx = [t for t in this_df.index if time[t] == d][0]
            idx = [t for t,_ in enumerate(time) if time[t] == d][0]
            for k,df in dfs.items():
                Y_decades[k][d].append(df.iloc[idx,i])

    golden = (1 + 5 ** 0.5) / 2
    miny = -0.5
    maxy = 4.5
    print(miny,maxy)
    x = np.linspace(miny,maxy,10001)
    lines = {k: [] for k in dfs.keys()}
    z     = {k: [] for k in dfs.keys()}
    props = dict(facecolor='white', alpha=0.5,lw=0,pad=0)
    # for animation
    for i,d in enumerate(use_decades[::-1]):
        for scenario in dfs.keys():
            y = Y_decades[scenario][d]
            params = norm.fit(y)
            this_y = norm.pdf(x,*params)
            z[scenario].append(this_y)
    # for 3-d plot
    ymax = 15
    yinc = 5
    ymaxvalue = -1
    for i,d in enumerate(plot_decades[::-1]):
        xscale = 1 - i / (len(plot_decades)*golden)
        lw = 1.5 - i / (len(use_decades)*2)
        for scenario in dfs.keys():
            scen_color = colors[scenario]
            this_label = None
            if i==len(plot_decades)-1:
                this_label = labels[scenario]
            this_x = x.mean()+xscale*(x-x.mean())
            y = Y_decades[scenario][d]
            params = norm.fit(y)
            this_y = norm.pdf(x,*params)
            idx = np.argmax(this_y)
            if (this_y[idx]*xscale+i) > ymaxvalue:
                ymaxvalue = this_y[idx]*xscale+i
            # normalized for scale
            ax.fill_between(this_x[this_y>th],i,i+this_y[this_y>th]*xscale,color=scen_color,lw=0.5,alpha=0.5,label=this_label)
            # for diagnostics plot histogram for 2300
            #if d == 2300:
            #    ax.hist(y,bins=30,range=(x[0],x[-1]),density=True,color=scen_color,lw=0.5,alpha=0.25)
            # normalized by height
            #ax.fill_between(this_x[this_y>th],i,i+this_y[this_y>th]/(max(this_y)*xscale),color=scen_color,lw=0.5,alpha=0.5,label=this_label)
            lines[scenario].append(this_x[idx])
            if d%25==0:
                ax.text((0.95-xscale)*x.mean(),i,d,ha='right',va='center',fontsize=8,zorder=10,bbox=props)
                ax.plot([(1-xscale)*x.mean()]*2,[i,i+ymax*xscale],color='gray',lw=0.5,alpha=1,zorder=-10,solid_capstyle='round') # pseudo y-lines
                ax.plot([(1-xscale)*x.mean(),(1+xscale)*x.mean()],[i,i],color='gray',lw=0.5,alpha=1,zorder=-10,solid_capstyle='round')
        print(d, xscale,this_y[idx]*xscale,ymaxvalue)
        if i==len(plot_decades)-1:
            ax.legend(loc=10)
    y_lines = range(len(plot_decades))

    for scenario in dfs.keys():
        color = colors[scenario]
        ax.plot(lines[scenario],y_lines,lw=2,color=color,alpha=0.25,zorder=-10)
    x0 = 0
    for x0 in [0,1,2,3,4]:
        ax.plot([x0,x.mean()+xscale*(x0-x.mean())],[0,i],color='gray',lw=0.5,alpha=1,zorder=-10,solid_capstyle='round')
    ax.plot([0,4],[0,0],color='gray',lw=0.5,alpha=1,zorder=-10,solid_capstyle='round')
    ax.plot([x.mean()+xscale*(0-x.mean()),x.mean()+xscale*(4-x.mean())],[i,i],color='gray',lw=0.5,alpha=1,zorder=-10,solid_capstyle='round')
    yrange = range(yinc,ymax+yinc,yinc)
    for ymax in yrange:
        ax.plot([0,x.mean()+xscale*(0-x.mean())],[ymax,i+ymax*xscale],color='gray',lw=0.5,alpha=0.75,zorder=0,solid_capstyle='round')
    ax.set_ylim(-0.05,ymaxvalue+0.05)
    ax.set_xlim(0,None)
    #ax.set_xticks([])
    ax.set_yticks(yrange)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['left'].set_visible(False)
    #ax.grid()
    ax.set_xlabel("Sea level contribution (in m)")
    ax.set_ylabel("Probability distribution function")
    fig.savefig('reports/figures/slr_emergence.png',dpi=300, bbox_inches='tight', pad_inches = 0.01)
    fig.savefig('reports/figures/slr_emergence.pdf',dpi=300, bbox_inches='tight', pad_inches = 0.01)
    #sys.exit()

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

    fig_ani, ax_ani = plt.subplots(figsize=(4,3))
    line = []
    for scenario in dfs.keys():
        ax_ani.fill_between([], [], lw=0, alpha=0.5, color=colors[scenario],label=labels[scenario])
        l, = ax_ani.plot([], [], lw=1, alpha=0.75, color=colors[scenario])
        line.append(l)
    #ax_ani.fill_between([], [], lw=0, alpha=0.5, color=colors["rcp85"], label="RCP8.5")
    #line2, = ax_ani.plot([], [], lw=1, alpha=0.75, color=colors["rcp85"])
    text = ax_ani.text(0.9,0.2,use_decades[-1],fontsize=16,ha='center',va='center',transform=ax_ani.transAxes)
#    ax_ani.grid()
    ax_ani.set_xlim(miny, maxy)
    ax_ani.set_ylim(-0.05,3)
#    ax_ani.set_ylim(-0.05,None)
    ax_ani.set_yticks([])
    ax_ani.spines['top'].set_visible(False)
    #ax_ani.spines['bottom'].set_visible(False)
    ax_ani.spines['right'].set_visible(False)
    ax_ani.spines['left'].set_visible(False)
    ax_ani.legend(loc=5)
    ax_ani.set_xlabel("Sea level contribution (in m)")


    def run(i):
        # update the data
        ax_ani.set_xlim(x[0],x[-1])
        ax_ani.set_xlim(miny, maxy)
        ax_ani.set_ylim(-0.05,3)

        [a.remove()  for a in ax_ani.collections]
        for j,scenario in enumerate(dfs.keys()):
            y = z[scenario][n_decades-i-1]
            ax_ani.fill_between(x[y>th], y[y>th], lw=0, alpha=0.5, color=colors[scenario])
            idx = np.argmax(y)
            line[j].set_data(2*[x[idx]], [0,y[idx]])
            text.set_text(use_decades[i])

        return text

    range_decades = []
    n_rep = 15
    for i,d in enumerate(use_decades):
        range_decades.append(i)
        if d%25==0:
            for _ in range(n_rep):
                range_decades.append(i)
    for _ in range(n_rep):
        range_decades.append(i)

    ani = animation.FuncAnimation(fig_ani, run, range_decades, interval=50, repeat_delay=1000)#, init_func=init)
    fig_ani.tight_layout()
    #ani = animation.ArtistAnimation(fig_ani, ims, interval=50, blit=True, repeat_delay=1000)
    ani.save("reports/figures/pdfs.gif",dpi=150)

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

