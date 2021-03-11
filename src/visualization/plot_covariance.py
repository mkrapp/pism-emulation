import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import pickle

def main():
    # here goes the main part
    with open("./models/gp_exact.pkl", "rb") as f:
        gpe = pickle.load(f)
    idx = np.loadtxt("data/interim/idx_train.txt")
    print(gpe.kernel_.k1)
    fig, axes = plt.subplots(1,2,figsize=(8,4),sharex=True,sharey=True)
    N = 3
    #fig2, ax2 = plt.subplots(N,N,sharex=True,sharey=True)
    #ax2 = ax2.flatten()
    x = gpe.X_train_
    print(x.shape)
    idx_sorted = np.argsort(idx)
    print(idx_sorted)
    x = x[idx_sorted,:]
    nt = 2300-2017
    X = np.load("X.npy")
    x = X[:27*nt,:]
    for i,kernel in enumerate([gpe.kernel,gpe.kernel_]):
        ax = axes[i]
        cov = kernel(x)
        print(cov.shape)
        extent = [0,len(x)-1,len(x)-1,0]
        cmap="plasma"
        im = ax.imshow(cov,extent=extent,origin="upper",cmap=cmap,interpolation="nearest")
        plt.colorbar(im,ax=ax,shrink=0.7)

        zoom = False
        if zoom:
            # inset axes....
            start = 3*nt
            end = 6*nt
            axins = ax.inset_axes([0.03, 0.03, 0.42, 0.42])
            axins.matshow(cov, extent=extent, origin="upper",cmap=cmap,interpolation="nearest")
            # sub region of the original image
            x1, x2, y1, y2 = start,end-1,end-1,start
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            axins.set_xticklabels('')
            axins.set_yticklabels('')

            rec,cl = ax.indicate_inset_zoom(axins,edgecolor='k')
            rec.set_alpha(0.9)
            for l in cl:
                l.set_alpha(0.9)
        #ax.set_title(r"$k(\mathbf{x}_{train},\mathbf{x}'_{train})$")
        if i==0:
            label = "before"
        else:
            label = "after"
        ax.set_title(r"$k(\mathbf{x},\mathbf{x}')$"+f" {label} training")
    fig.savefig("reports/figures/covariance_matrix.png",dpi=150, bbox_inches='tight', pad_inches = 0.01)

    #time = np.arange(2017,2300)
    #for k in range(N*N):
    #    n = np.random.randint(0,81*2)
    #    this_x = X[n*nt:(n+1)*nt,:]
    #    ax2[k].plot(time,gpe.sample_y(this_x,4).squeeze(),lw=1,alpha=0.5)


    plt.show()

if __name__ == "__main__":
    main()

