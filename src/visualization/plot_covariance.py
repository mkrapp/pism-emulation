import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import pickle

def main():
    # here goes the main part
    with open("./models/gp_exact.pkl", "rb") as f:
        gpe = pickle.load(f)
    fig, ax = plt.subplots(1,1)
    x = gpe.X_train_
    print(x.shape)
    #x[:,0] = 0
    #x[:,1] = 0
    #x[:,2] = 0
    #x[:,3] = 0
    #x[:,4] = 0
    #x[:,5] = 0
    #x[:,6] = 0
    cov = gpe.kernel_(x)
    print(cov.shape)
    extent = [0,len(x)-1,len(x)-1,0]
    cmap="Spectral"
    im = ax.imshow(cov,extent=extent,origin="upper",cmap=cmap,interpolation="nearest")
    plt.colorbar(im,ax=ax,shrink=0.9)
    # inset axes....
    start = 200
    end = 251
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
    #im = ax[1].matshow(cov[:20,:20],cmap="Spectral")
    #plt.colorbar(im,ax=ax[1],shrink=0.8)
    ax.set_title(r"$k(\mathbf{x}_{train},\mathbf{x}'_{train})$")
    plt.show()

if __name__ == "__main__":
    main()

