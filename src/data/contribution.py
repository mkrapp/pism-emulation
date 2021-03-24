import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

def main():
    # here goes the main part
    year = int(sys.argv[-1])
    for fnm in sys.argv[1:-1]:
        df = pd.read_csv(fnm,index_col=0)
        df.drop(columns=["GMT"],inplace=True)

        df_median = df.quantile(0.5,axis=1).loc[year]
        df_lo = df.quantile(0.025,axis=1).loc[year]
        df_hi = df.quantile(0.975,axis=1).loc[year]
        print("%s %d median (low,high) = %.2f (%.2f, %.2f)"%(fnm.split(".")[0].split("_")[-1],year,df_median,df_lo,df_hi))

if __name__ == "__main__":
    main()

