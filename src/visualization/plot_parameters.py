import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

def main():
    # here goes the main part
    limits = {}
    limits["sia"] = (1.2,4.8)
    limits["ssa"] = (0.42,0.8)
    limits["q"] = (0.25,0.75)
    limits["phi"] = (5,15)
    fig, ax = plt.subplots(3,3,figsize=(6,6))
    columns = ["sia","ssa","q","phi"]
    d = pd.read_csv("data/processed/emulator_runs_parameters.csv",index_col=0)
    slr = pd.read_csv("data/processed/emulator_runs_rcp85.csv",index_col=0).T.drop(["GMT"])[2300].T

    for i,x in enumerate(columns[1:]):
        for j,y in enumerate(columns[:-1]):
            if i>=j:
                im = ax[i,j].scatter(d[y],d[x],s=60,c=slr.values,marker='.',linewidths=0,cmap="plasma_r",alpha=0.75)

    cbar = fig.colorbar(im, ax=ax[-2,2:3].ravel().tolist(),label="SLR [m] in 2300\nfor RCP8.5")

    for i,y in enumerate(columns[1:]):
        for j,x in enumerate(columns[:-1]):
            if i>=j:
                ax[i,j].set_xlim(limits[x])
                ax[i,j].set_ylim(limits[y])
            else:
                ax[i,j].remove()
            if i==2:
                ax[i,j].set_xlabel(x)
            if j==0:
                ax[i,j].set_ylabel(y)

    plt.tight_layout()
    fig.savefig("reports/figures/parameters.png",dpi=300, bbox_inches='tight', pad_inches = 0.01)
    plt.show()

if __name__ == "__main__":
    main()

