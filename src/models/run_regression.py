#import itertools
import argparse
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import sys
#from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, LeaveOneOut, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import StandardScaler, FunctionTransformer, MaxAbsScaler, MinMaxScaler
#from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import pickle
from tqdm import tqdm

start_year = 2018
end_year = 2300
TRAIN_SIZE = 0.8
np.set_printoptions(precision=3,linewidth=180)

def gp_model(X_train, y_train):
    k1 = RBF(length_scale=[1]*X_train.shape[1])
    kernel = 1**2*k1 + WhiteKernel()
    gp = GaussianProcessRegressor(kernel=kernel, random_state=0)
    gp_model = Pipeline([ ('scaler', StandardScaler()), ("gp", gp) ])
    gp_model.fit(X_train, y_train)
    print("\nLearned kernel: %s" % gp.kernel_)
    print("Log-marginal-likelihood: %.3f"
          % gp_model.named_steps["gp"].log_marginal_likelihood(gp.kernel_.theta))
    return gp_model

def rf_model(X_train, y_train):
    ## Random Forest Regressor
    rf = RandomForestRegressor(random_state=42)
    rf_model = Pipeline([ ('scaler', StandardScaler()), ("rf", rf) ])
    rf_model.fit(X_train, y_train.ravel())
    # Get the feature importances
    importances = rf_model.named_steps["rf"].feature_importances_
    # Create a DataFrame for feature names and their corresponding importances
    features = X_train.columns if isinstance(X_train, pd.DataFrame) else [f'Feature {i}' for i in range(X_train.shape[1])]
    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
        })

    # Sort the feature importances in descending order
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Plot the feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.title('Feature Importance from RandomForestRegressor')
    plt.gca().invert_yaxis()  # To display the most important feature at the top
    plt.show()
    return rf_model

def mlp_model(X_train, y_train):
    ## Define the parameter grid
    #param_grid = {
    #    'hidden_layer_sizes': [(64, 32), (32, 16), (32, 32), (64, 64)],  # Different architectures
    #    'batch_size': [32, 64, 128, 256],  # Various batch sizes
    #    'alpha': [0.0001, 0.001, 0.01, 0.1],  # Regularization strength
    #    'learning_rate_init': [0.0001, 0.001, 0.01],  # Learning rates
    #    'max_iter': [500],  # Keep iterations fixed for simplicity
    #    'early_stopping': [True],  # Use early stopping
    #    'validation_fraction': [0.1]  # Use 10% of training data for validation
    #}

    ## Initialize LOO cross-validation
    ##loo = LeaveOneOut()
    ## Use k-fold instead of LOO (5-fold is a good tradeoff)
    #kf = KFold(n_splits=5, shuffle=True, random_state=42)

    ## Initialize the MLPRegressor
    #model = MLPRegressor(random_state=42)

    ## Set up the GridSearchCV (you can replace with RandomizedSearchCV for faster results)
    #grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=kf, n_jobs=-1, verbose=2)
    ##grid_search = RandomizedSearchCV(estimator=model,
    ##                                 param_distributions=param_grid, n_iter=20, cv=3, n_jobs=-1, verbose=2, random_state=42)

    ## Fit the grid search on your training data
    #grid_search.fit(X_train, y_train.ravel())

    ## Print the best parameters found by GridSearchCV
    #print(f"Best hyperparameters: {grid_search.best_params_}")

    ## Evaluate the best model
    #best_model = grid_search.best_estimator_
    #y_pred = best_model.predict(X_test)
    #from sklearn.metrics import mean_squared_error
    #mse = mean_squared_error(y_test, y_pred)
    #print(f"Best Model Validation MSE: {mse:.4f}")
    #sys.exit()
    hidden_layer_sizes=(64,64)
    print(hidden_layer_sizes)
    #model = MLPRegressor(random_state=1,
    #                     hidden_layer_sizes=hidden_layer_sizes,)
    #                     max_iter=1, warm_start=True)
    mlp = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='adam',
            alpha=1e-3,
            learning_rate='adaptive',
            learning_rate_init=1e-4,
            tol=1e-9,
            max_iter=500,
            batch_size=64,
            validation_fraction=0.2,
            early_stopping=True
            )
    # Use a pipeline to apply batch normalization (as sklearn MLP doesn't support it natively)
    mlp_model = Pipeline([ ('scaler', StandardScaler()), ("mlp", mlp) ])
    mlp_model.fit(X_train, y_train.ravel())
    this_mlp_model = mlp_model.named_steps["mlp"]
    print(f"Loss: {this_mlp_model.loss_:.6f}")  # Separate print
    # Plot the training loss and validation loss
    if hasattr(this_mlp_model, 'loss_curve_'):
        fig, ax = plt.subplots(2,1,sharex=True)#figsize=(10, 6))
        ax[0].plot(this_mlp_model.loss_curve_[2:], label='Training Loss')
        ax[0].set_yscale("log")
        if hasattr(this_mlp_model , 'validation_scores_'):
            ax[1].plot(this_mlp_model.validation_scores_[2:], label='Validation Loss')
        fig.tight_layout()
        plt.show()
    return mlp_model


def main():
    # here goes the main part
    parser = argparse.ArgumentParser(
            prog='run_regression',
            description='Run regression model.')
    parser.add_argument('--input', type=str, required=True, help="Input file (to pickle from)")
    parser.add_argument('--model', type=str, required=True, help="Model type", choices=["mlp","gp","rf"])
    args = parser.parse_args()
    model_type = args.model
    if model_type == "gp":
        train_size = 0.05
        print(f"Changing trainings sample size fraction to {train_size} for model '{model_type}'!")
    else:
        train_size = TRAIN_SIZE

    fnm = args.input
    subset = fnm.split(".")[0].split("_")[1]
    print(subset)
    with open(fnm, "rb") as f:
        [df,df_timeseries,df_forcings] = pickle.load(f)
    print(df)
    print(df_timeseries)
    df_timeseries = df_timeseries.loc[start_year:end_year]
    time = df_timeseries.index
    #df_timeseries -= df_timeseries.iloc[0]
    n_expid = df_timeseries.columns.get_level_values(0).unique().values
    print(len(n_expid))
    dependent_variables = df_timeseries.columns.get_level_values(1).unique().values
    print(dependent_variables)
    parameters = list(df.columns.get_level_values(0).unique().values)
    parameters.remove("scenario")

    scenarios = df_forcings.columns.get_level_values(0).unique().values
    variables = df_forcings.columns.get_level_values(1).unique().values

    print(time)
    df_forcings = df_forcings.loc[time]# - 273.15

    forcings = list(df_forcings.columns.get_level_values(1).unique().values)
    print(forcings)

    nt = len(time)

    y_name = dependent_variables[0]

    dt = time[1]-time[0]


    n_params = len(parameters)
    nt = len(time)
    ys = []
    forcs = []
    X = np.zeros((len(n_expid)*nt,n_params+3)).astype(float)
    y = np.zeros((len(n_expid)*nt)).astype(float)
    m = 0
    for i,n in enumerate(n_expid):
        this_forc = df_forcings[df["scenario"].loc[n]]["global_mean_temperature"]#.values.flatten()
        x1 = this_forc
        x2 = this_forc.cumsum()
        #x2 -= x2.iloc[0]
        x3 = (this_forc.groupby((this_forc != this_forc.shift(1)).cumsum()).cumcount()+1)*dt # years since last temperature change
        forcs.append(this_forc.values.flatten())
        this_y = df_timeseries[(n, y_name)].iloc[:].values
        #if y_name == dependent_variables[3]:
        #    this_y *= 1e-18
        this_y -= this_y[0] # set start value to zero
        this_y *= -1.
        #x1 = np.roll(this_y,1)
        #x1[0] = np.nan
        ys.append(this_y)
        this_params = []
        for p in parameters:
            this_params.append(df.loc[n][p])
        for t in range(nt):
            X[i*nt+t,:n_params] = this_params
            #X[i*nt+t,n_params:] = [this_forc.iloc[t],x1.iloc[t],x2[t]]
            #X[i*nt+t,n_params:] = [this_forc.iloc[t],time[t]-time[0]]
            #X[i*nt+t,n_params:] = [x1.iloc[t],x2.iloc[t]]#,time[t]-time[0]]
            X[i*nt+t,n_params:] = [x1.iloc[t],x2.iloc[t],x3.iloc[t]]
            y[i*nt+t] = this_y[t]

    ys = np.array(ys)
    print(f"{ys.shape = }")
    forcs = np.array(forcs)

    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    print(f"{y.shape = }")

    idx = np.arange(X.shape[0]).astype(int)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X, y, idx, train_size=train_size, random_state=0)
    print(f"{y_train.shape = }")
    np.savetxt("data/interim/idx_train.txt",idx_train,fmt="%d")
    np.save("X.npy",X)

    if model_type == "mlp":
        model_f = mlp_model
    elif model_type == "gp":
        model_f = gp_model
    elif model_type == "rf":
        model_f = rf_model

    model = model_f(X_train, y_train)

    # PLOTTING
    fig, axes = plt.subplots(9,9,sharex=True,sharey=True,figsize=(13,8))
    axes = axes.flatten()

    n = int(len(n_expid)/2)
    for i in range(n):
        # RCP2.6
        l, = axes[i].plot(time,ys[i,:])

        X_ = X[i*nt:(i+1)*nt,:]
        y_pred = model.predict(X_)
        r2 = r2_score(ys[i,:],y_pred)
        axes[i].plot(time,y_pred,ls='--',c=l.get_color())
        fw = "normal"
        if r2 < 0.3:
            fw = "bold"
        axes[i].text(0.05,0.6,"%.2f"%r2,fontsize=6,fontweight=fw,color=l.get_color(),transform=axes[i].transAxes)
        # RCP8.5
        l, = axes[i].plot(time,ys[i+n,:])
        X_ = X[(i+n)*nt:(i+n+1)*nt,:]
        y_pred = model.predict(X_)
        r2 = r2_score(ys[i+n,:],y_pred)
        axes[i].plot(time,y_pred,ls='--',c=l.get_color())
        fw = "normal"
        if r2 < 0.3:
            fw = "bold"
        axes[i].text(0.05,0.8,"%.2f"%r2,fontsize=6,fontweight=fw,color=l.get_color(),transform=axes[i].transAxes)
        axes[i].set_title(",".join([str(p) for p in df.loc[i]][1:]),fontsize=6,va='top')
    fig.tight_layout()
    plt.show()

    # save forcing and coefficients for emulator
    fnm_model = f"./models/{model_type}.pkl"
    print('Pickle model into "%s".'%fnm_model)
    with open(fnm_model, "wb") as f:
        pickle.dump(model,f)

    fnm_out = f"data/processed/{model_type}_{y_name}.pkl"
    print('Pickle data into "%s".'%fnm_out)
    with open(fnm_out, "wb") as f:
        pickle.dump([forcs,parameters,time,y_name,ys.min(),ys.max(),ys,X,df],f)
    #fig.tight_layout()
    #fnm_out = f'reports/figures/{model_type}_{y_name}_panel.png'
    #fig.savefig(fnm_out,dpi=300, bbox_inches='tight', pad_inches = 0.01)


if __name__ == "__main__":
    main()

