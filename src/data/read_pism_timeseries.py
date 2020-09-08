import numpy as np
import pandas as pd
import netCDF4 as nc4
import sys
import itertools
import os.path
import pickle
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter

def load(fnm):
    df = pd.read_csv(fnm,delim_whitespace=True,comment='#',header=None,index_col=0).mean(axis=1)
    return df

def main():
    # here goes the main part

    dt = int(sys.argv[1])
    # forcings
    forcings = []
    scenarios = ["rcp26","rcp85"]
    variables = ["ice_surface_temp_anomaly", "salinity_ocean", "climatic_mass_balance_anomaly", "theta_ocean"]
    for scenario in scenarios:
        print(scenario)
        for var in variables:
            print("\t"+var)
            fnm = "data/external/PISM/spatialavg/%s-%s-spatialavg.nc"%(scenario,var)
            nc = nc4.Dataset(fnm,"r")
            time = nc.variables["time"][:].squeeze()
            forcings.append(nc.variables[var][:].squeeze().data)
            nc.close()
        var = "global_mean_temperature"
        print("\t"+var)
        fnm = "data/external/gmt/global_tas_Amon_NorESM1-M_%s_r1i1p1.dat"%(scenario)
        this_df = load(fnm)
        this_df.loc[:] = savgol_filter(this_df, 51, 3)
        forcings.append(this_df.loc[time].values)
    iterables = [scenarios,variables+["global_mean_temperature"]]
    multi_index = pd.MultiIndex.from_product(iterables, names=['scenario', 'variable'])
    forcings = np.array(forcings).T
    df_forcings = pd.DataFrame(forcings,index=time.astype(int),columns=multi_index)
    df_forcings.index.name="year"

    #ax = plt.subplot(1,1,1)
    #for scenario in scenarios:
    #    fnm = "data/external/gmt/global_tas_Amon_NorESM1-M_%s_r1i1p1.dat"%(scenario)
    #    x = load(fnm)
    #    time = x.index
    #    l, = ax.plot(time,x)
    #    x = savgol_filter(x, 51, 3)
    #    ax.plot(time,x,ls='--',c=l.get_color())
    #plt.show()
    #sys.exit()

    ## forcings (GMT only)
    #forcings = []
    #scenarios = ["rcp26","rcp85"]
    #variables = ["global_mean_temperature"]
    #for scenario in scenarios:
    #    print(scenario)
    #    for var in variables:
    #        print("\t"+var)
    #        fnm = "gmt/global_tas_Amon_NorESM1-M_%s_r1i1p1.dat"%(scenario)
    #        this_df = load(fnm)
    #        time = this_df.index
    #        forcings.append(this_df.values)
    #iterables = [scenarios,variables]
    #multi_index = pd.MultiIndex.from_product(iterables, names=['scenario', 'variable'])
    #forcings = np.array(forcings).T
    #df_forcings = pd.DataFrame(forcings,index=time.astype(int),columns=multi_index)
    #df_forcings.index.name="year"

    # outputs and parameters
    variables = [
            "sea_level_rise_potential",
            "ice_mass",
            "basal_mass_flux_floating",
            "tendency_of_ice_mass",
            "tendency_of_ice_mass_due_to_discharge",
            "tendency_of_ice_mass_due_to_basal_mass_flux",
            "tendency_of_ice_mass_due_to_surface_mass_flux"]

    sia = ["1.2","2.4","4.8"]
    ssa = ["0.42","0.6","0.8"]
    q   = ["0.25","0.5","0.75"]
    phi = ["5","10","15"]
    #sia = ["1.2","4.8"]
    #ssa = ["0.42","0.8"]
    #q   = ["0.25","0.75"]
    #phi = ["5","15"]

    flags = {
            "sia": "stress_balance.sia.enhancement_factor",
            "ssa": "stress_balance.ssa.enhancement_factor",
            "q": "basal_resistance.pseudo_plastic.q",
            "phi": "basal_yield_stress.mohr_coulomb.topg_to_phi.phi_min"
            }

    combinations = list(itertools.product(*[scenarios,sia,ssa,q,phi]))
    print(len(combinations))
    d = {"scenario": [], "sia": [], "ssa": [], "q": [], "phi": []}
    timeseries = []
    path1,path2= "data/external/PISM/v1/timeser_2100/","data/external/PISM/v1/timeser_2300/"
    for c in combinations:
        fnm1 = path1+"timeser_NorESM1-M-%s-sia%s_ssa%s_q%s_phi%s.nc"%c
        fnm2 = path2+"timeser_NorESM1-M-%s-th2300_sia%s_ssa%s_q%s_phi%s.nc"%c
        #fnm = "data/external/timeser/timeser_NorESM1-M-%s-sia%s_ssa%s_q%s_phi%s.nc"%c
        #fnm1 = path1+"timeser_NorESM1-M-%s-pico_sia%s_ssa%s_q%s_phi%s.nc"%c
        #fnm2 = path2+"timeser_NorESM1-M-%s-pico2300_sia%s_ssa%s_q%s_phi%s.nc"%c
        nc1 = nc4.Dataset(fnm1,"r")
        nc2 = nc4.Dataset(fnm2,"r")
        for fnm,nc in zip([fnm1,fnm2],[nc1,nc2]): # you could just assume this for the first file (fnm1), I guess
            c_n = 1
            c_dict = {}
            pism_byte = nc.variables["pism_config"]
            for c_name,flag in flags.items():
                c_from_file = pism_byte.getncattr(flag)
                #print(fnm,flag,c_from_file)
                #print(c_name,c_from_file,c[c_n])
                if (float(c_from_file) != float(c[c_n])):
                    print("%s: Expected %s, got %s (from file'%s')"%(c_name, c[c_n],c_from_file,fnm))
                c_n += 1
                c_dict[c_name] = float(c_from_file)
        for var in variables:
            x1 = nc1.variables[var][:].squeeze()
            x2 = nc2.variables[var][:].squeeze()
            x = np.concatenate([x1,x2])
            timeseries.append(x)
        time1 = (nc1.variables["time"][:].squeeze()/(86.4e3*365.)).astype(int)
        time2 = (nc2.variables["time"][:].squeeze()/(86.4e3*365.)).astype(int)
        time = np.concatenate([time1,time2])
        nc1.close()
        nc2.close()
        d["scenario"].append(c[0])
        #d["sia"].append(float(c[1]))
        #d["ssa"].append(float(c[2]))
        #d["q"].append(float(c[3]))
        #d["phi"].append(float(c[4]))
        d["sia"].append(c_dict["sia"])
        d["ssa"].append(c_dict["ssa"])
        d["phi"].append(c_dict["phi"])
        d["q"].append(c_dict["q"])


    df = pd.DataFrame(d)
    print(df)
    iterables = [range(len(df)),variables]
    multi_index = pd.MultiIndex.from_product(iterables, names=['expid', 'variable'])
    timeseries = np.array(timeseries).T
    # have to subtract 100, otherwise:
    # pandas._libs.tslibs.np_datetime.OutOfBoundsDatetime: Out of bounds nanosecond timestamp: 2300-12-31 00:00:00
    index = pd.date_range('1/1/%d'%(time[0]-100), periods=len(time), freq='Y')
    df_timeseries = pd.DataFrame(timeseries,index=index,columns=multi_index)
    df_timeseries.index.name="year"
    index_forcing = pd.date_range('1/1/%d'%(df_forcings.index[0]-100), periods=len(df_forcings), freq='Y')
    df_forcings.index = index_forcing
    #print(df_timeseries.index)
    #print(df_forcings.index)
    last = df_forcings.index[-1]
    for t in df_timeseries.index[df_timeseries.index>last]:
        df_forcings.loc[t] = df_forcings.loc[last]
    offset = (df_timeseries.index[0].year-df_forcings.index[0].year)%dt
    df_timeseries = df_timeseries.resample("%dY"%dt).mean()
    df_forcings = df_forcings.iloc[offset:].resample("%dY"%dt).mean()
    time  = df_timeseries.index.year + 100
    df_forcings.index = df_forcings.index.year + 100
    df_timeseries.index = time
    #print(df_timeseries.index)
    #print(df_forcings.index)
    print(len(time))
    fnm = sys.argv[2][:-4]+"-%dyr.pkl"%dt
    print("Datasets 'df' and 'df_timeseries' and 'df_forcings' pickled under '%s'"%fnm)
    with open(fnm, "wb") as f:
        pickle.dump([df,df_timeseries,df_forcings], f)

    #df_timeseries.plot()
    # quick inspection plot
    n_expid = df_timeseries.columns.get_level_values(0).unique().values
    variables = df_timeseries.columns.get_level_values(1).unique().values
    nrows=2
    ncols = int(np.ceil(len(variables)/nrows))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True)
    axes = axes.flatten()
    colors = ["C0","C1"]
    for i in n_expid:
        if (df["scenario"].loc[i] == "rcp26"):
                c = colors[0]
        else:
                c = colors[1]
        for n,var in enumerate(variables):
            if "tendency" in var or "flux" in var:
                axes[n].plot(time[3:],df_timeseries[(i, var)].values[3:]/360000.,ls='-',c=c,alpha=0.1,lw=1)
            else:
                axes[n].plot(time[3:],df_timeseries[(i, var)].values[3:],ls='-',c=c,alpha=0.1,lw=1)

    for n,var in enumerate(variables):
        axes[n].set_ylabel(var)
        axes[n].grid()
    axes[-1].set_xlabel("year")
    custom_lines = [Line2D([0], [0], color="C0", lw=1),
                    Line2D([0], [0], color="C1", lw=1)]
    axes[0].legend(custom_lines, ['RCP2.6', 'RCP8.5'], loc=3,ncol=2)
    fig.savefig('output_variables_2300.png',dpi=150, bbox_inches='tight', pad_inches = 0.01)


    plt.show()


if __name__ == "__main__":
    main()

