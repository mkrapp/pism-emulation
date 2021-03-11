import pandas as pd

df_rcp26 = pd.read_csv("data/processed/emulator_runs_rcp26.csv",index_col=0)
df_rcp85 = pd.read_csv("data/processed/emulator_runs_rcp85.csv",index_col=0)
df_rcp45 = pd.read_csv("data/processed/emulator_runs_rcp45.csv",index_col=0)
df_rcp60 = pd.read_csv("data/processed/emulator_runs_rcp60.csv",index_col=0)
for endyear in [2100,2300]:
    this_df_rcp26 = df_rcp26.loc[endyear] - df_rcp26.loc[2000]
    print("RCP2.6 (%d) = %.2f - %.2f"%(endyear,this_df_rcp26.quantile(0.025), this_df_rcp26.quantile(0.975)))

    this_df_rcp85 = df_rcp85.loc[endyear] - df_rcp85.loc[2000]
    print("RCP8.5 (%d) = %.2f - %.2f"%(endyear,this_df_rcp85.quantile(0.025), this_df_rcp85.quantile(0.975)))

    this_df_rcp45 = df_rcp45.loc[endyear] - df_rcp45.loc[2000]
    print("RCP4.5 (%d) = %.2f - %.2f"%(endyear,this_df_rcp45.quantile(0.025), this_df_rcp45.quantile(0.975)))

    this_df_rcp60 = df_rcp60.loc[endyear] - df_rcp60.loc[2000]
    print("RCP6.0 (%d) = %.2f - %.2f"%(endyear,this_df_rcp60.quantile(0.025), this_df_rcp60.quantile(0.975)))
