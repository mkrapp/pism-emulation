import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("src/models/")
from gaussian_process_emulator import GaussianProcessEmulator, ExponentialDecay
import gpflow
import gpflow.utilities
from gpflow.kernels import White, Constant, Linear, SquaredExponential, Exponential, ChangePoints
import tensorflow as tf
import pickle

def plotkernelsample(k, ax, xmin=-10, xmax=10, n_samples=8):
    xx = np.linspace(xmin, xmax, xmax*20)[:, None]
    K = k(xx)
    ax.plot(xx, np.random.multivariate_normal(np.zeros(xmax*20), K, n_samples).T)
    ax.set_title("Samples " + k.__class__.__name__)


def plotkernelfunction(k, ax, xmin=-10, xmax=10, other=0):
    xx = np.linspace(xmin, xmax, xmax*20)[:, None]
    ax.plot(xx, k(xx, np.zeros((1, 1)) + other))
    ax.set_title(k.__class__.__name__ + " k(x, %.1f)" % other)

fnm = sys.argv[1]
with open(fnm, "rb") as f:
    [df,df_timeseries,df_forcings] = pickle.load(f)



f, axes = plt.subplots(2, 2, figsize=(6, 6))
axes = axes.flatten()

start = 2#84
end = -1#00
time = df_timeseries.iloc[start:end].index
nt = len(time)
dt = time[1]-time[0]
df_forcings = df_forcings.loc[time]
print(df_forcings)
dependent_variables = df_timeseries.columns.get_level_values(1).unique().values
parameters = list(df.columns.get_level_values(0).unique().values)
parameters.remove("scenario")
y_name = dependent_variables[0]
print(y_name)
idx = np.random.randint(low=0,high=162,size=64)
n_params = len(parameters)
X = np.zeros((len(idx)*nt,n_params+2)).astype(float)
y = np.zeros((len(idx)*nt)).astype(float)
colors = []
forcs = []
for n,i in enumerate(idx):
    this_forc = df_forcings[df["scenario"].loc[i]]["global_mean_temperature"]
    this_forc -= this_forc.iloc[0]
    this_forc = this_forc.cumsum()
    #this_forc = this_forc.ewm(halflife=10).mean()
    #x1 = (this_forc.groupby((this_forc != this_forc.shift(1)).cumsum()).cumcount()+1)*dt # years since last temperature change
    #x2 = np.where(np.gradient(this_forc)==0,-1,1)
    this_df = df_timeseries[(i,y_name)].iloc[start:end]
    this_df -= this_df.iloc[0]
    this_df *= -1.
    #y = np.gradient(this_df)
    this_y = this_df.values
    x = this_df.index
    #axes[0].plot(this_forc.values,this_y,alpha=0.75)
    #l, = axes[0].plot(x,this_y,alpha=0.75)
    #l, = axes[0].plot(x,this_forc,alpha=0.75)
    l, = axes[0].plot(this_forc,this_y,alpha=0.75)
    colors.append(l.get_color())
    this_params = []
    for p in parameters:
        this_params.append(df.loc[i][p])
    for t in range(nt):
        X[n*nt+t,:n_params] = this_params
        X[n*nt+t,n_params:] = [this_forc.iloc[t],time[t]-time[0]]
        y[n*nt+t] = this_y[t]
    forcs.append(this_forc)
axes[0].set_title(y_name)

#plt.show()
#sys.exit()

#fig.savefig("slr_gradient.png",dpi=150)
print(this_forc)

#k = Constant() + SquaredExponential()#ChangePoints([Linear(),ExponentialDecay(alpha=0.1)],[5])
#k = ChangePoints([Linear(),Constant()],[5],steepness=1) * ExponentialDecay()#ChangePoints([Linear(),ExponentialDecay(alpha=0.1)],[5])
#k = ChangePoints([SquaredExponential(lengthscales=0.5),SquaredExponential(lengthscales=5)],[0],steepness=1)*SquaredExponential()
#k = Constant()
#variances = [1]*6
#lengthscales = [11.726051408601933,2.7550378917398,1.191595125455758,2.1875601704509093,4.425140145275285,3.268006271318101]
#for i in range(6):
#    k += SquaredExponential(active_dims=[i],variance=variances[i],lengthscales=lengthscales[i])
#print(X.shape)
#plotkernelfunction(k, axes[1], other=0.0)
#plotkernelsample(k, axes[2])
#xx = np.linspace(-10, 10, 100)[:, None]
#im = axes[3].matshow(k(xx,xx))
#plt.colorbar(im,ax=axes[3],shrink=0.8)
#plt.show()
#sys.exit()

print(X.shape)
y = np.reshape(y,(-1,1))
gpe = GaussianProcessEmulator()
#user_kernels = "ChangePoints([Linear()* ExponentialDecay(),Constant()* ExponentialDecay()],[290.394225],steepness=1)"
#user_kernels = "Constant() * SquaredExponential(active_dims=[0]) * SquaredExponential(active_dims=[1]) * SquaredExponential(active_dims=[2]) * SquaredExponential(active_dims=[3]) * SquaredExponential(active_dims=[4]) * ExponentialDecay(active_dims=[5])" # * Linear(active_dims=[5])"
#user_kernels = "Constant() + SquaredExponential(active_dims=[0,1,2,3]) * Linear(active_dims=[4]) * ExponentialDecay(active_dims=[5])"
#user_kernels = "Linear(active_dims=[6])*(Constant(active_dims=[0,1,2,3,4])*SquaredExponential(active_dims=[0,1,2,3])*ExponentialDecay(active_dims=[4]) + Constant(active_dims=[0,1,2,3,4])*SquaredExponential(active_dims=[0,1,2,3,4]))"
user_kernels = "Constant() + SquaredExponential(active_dims=[0,1,2,3]) * SquaredExponential(active_dims=[4,5])"
gpe.initialize(X,y,method="user=%s"%user_kernels,maxiter=1000,learning_rate=1e-2,scale_X=True,multiple_kernel_dims=False)
##gpe.initialize(X,y,method="non-linear",maxiter=10000,learning_rate=1e-2,scale_X=True,multiple_kernel_dims=True)
gpe.training()
gpe.summary()
#gpe.save("./models/")
#gpe.load("./models/")
im = axes[3].matshow(gpe.kernel(X,X))
plt.colorbar(im,ax=axes[3],shrink=0.8)
y_pred, y_std = gpe.predict(X)
print(y_pred.shape)
y_pred = y_pred
y_std  = y_std
y_std = y_std**0.5
for n,i in enumerate(idx):
    t = range(n*nt,(n+1)*nt)
    x = forcs[n]
    axes[0].plot(x,y_pred[t,0],ls='--',c=colors[n],alpha=0.75)
    axes[0].fill_between(x,y_pred[t,0]-2*y_std[t,0],y_pred[t,0]+2*y_std[t,0],lw=0,alpha=0.25,color=colors[n])
plt.show()
