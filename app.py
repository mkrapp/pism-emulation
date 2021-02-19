import plotly.graph_objects as go # or plotly.express as px
from plotly.subplots import make_subplots
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import pickle
#from IPython.display import display
#import ipywidgets as widgets
import plotly.graph_objects as go
import sys
sys.path.append("./src/models/")

fnm_in = "./data/processed/gp_sea_level_rise_potential.pkl"
with open(fnm_in, "rb") as f:
    [_,parameters,time_train,y_name,miny,maxy,ys,_,_] = pickle.load(f)

t0_train = time_train[0]

fnm_in = "./data/interim/runs_v1-1yr.pkl"
with open(fnm_in, "rb") as f:
    _,_,scenarios = pickle.load(f)
start_year = 1970
end_year   = 2299
time0 = 1992
#print(time_train)
time = scenarios['rcp26'].loc[start_year:end_year].index
#print(time)
rcp26 = scenarios['rcp26'].loc[start_year:end_year]
rcp85 = scenarios['rcp85'].loc[start_year:end_year]

dt = time[1]-time[0]

nt = len(rcp26)
n_params = len(parameters)

print(y_name)
with open("./models/gp_exact.pkl", "rb") as f:
    gpe = pickle.load(f)


def model_update(c,scen):
    X = np.zeros((nt,n_params+3))
    this_forc = scen["global_mean_temperature"]
    x1 = this_forc
    x2 = this_forc.cumsum()
    x2 -= x2.loc[t0_train] # cumulative warming starts with first years of training data
    x3 = (this_forc.groupby((this_forc != this_forc.shift(1)).cumsum()).cumcount()+1)*dt # years since last temperature change
    for i,t in enumerate(time):
        X[i,:n_params] = c
        X[i,n_params:] = [x1.loc[t],x2.loc[t],x3.loc[t]]
    y_pred = gpe.predict(X)
    idx_diff = time0-start_year
    y_pred = y_pred - y_pred[idx_diff]

    return np.array(y_pred)

# Antarctica only; https://www.nature.com/articles/s41586-020-2591-3/tables/1
t_start = time==1993
t_end = time==2018
dslr_obs = 0.44e-3
# https://www.pnas.org/content/116/4/1095#T2
t_start = time==1979
t_end = time==2017
dslr_obs = 15.9e-3
# IMBIE team; abstract; https://www.nature.com/articles/s41586-018-0179-y
t_start = time==1992
t_end = time==2017
dslr_obs_mean = 7.6*1e-3
uncert_factor = 2.
dslr_obs_max  = (7.6+uncert_factor*3.9)*1e-3
dslr_obs_min  = (7.6-uncert_factor*3.9)*1e-3
ddslr_obs_mean = (109)/360.e3 # Gt/a -> m/a
ddslr_obs_min  = (109 - uncert_factor*56)/360.e3 # Gt/a -> m/a
ddslr_obs_max  = (109 + uncert_factor*56)/360.e3 # Gt/a -> m/a  

# slider default values
sia_val = 2.4
ssa_val = 0.6
q_val   = 0.5
phi_val = 10

y1 = model_update([sia_val,ssa_val,q_val,phi_val],rcp26)
y2 = model_update([sia_val,ssa_val,q_val,phi_val],rcp85)
[sia_val,ssa_val,q_val,phi_val]

# Create traces
trace1 = go.Scatter(x=time, y=y1, mode='lines', line_color="dodgerblue", name='RCP2.6')
trace2 = go.Scatter(x=time, y=y2, mode='lines', line_color="orange", name='RCP8.5')
gtrace1 = go.Scatter(x=time, y=np.gradient(y1), mode='lines', line_color="dodgerblue", showlegend=False, name='RCP2.6')
gtrace2 = go.Scatter(x=time, y=np.gradient(y2), mode='lines', line_color="orange", showlegend=False, name='RCP8.5')

# plot median and 2.5-95% CI from ensembles
n = int(len(ys)/2)
time_train = time_train[time_train<=time[-1]]
ys = np.array(ys)
print(ys.shape)
y_rcp26 = ys[:n,:len(time_train)]
y_rcp85 = ys[n:,:len(time_train)]

median1 = go.Scatter(x=time_train,
                     y=np.percentile(y_rcp26,50,axis=0),
                     mode='lines',
                     opacity=0.5,
                     line_color="black", 
                     name='PISM RCP2.6 (median)')
median2 = go.Scatter(x=time_train,
                     y=np.percentile(y_rcp85,50,axis=0),
                     mode='lines',
                     opacity=0.5,
                     line_color="black",
                     name='PISM RCP8.5 (median)')
gmedian1 = go.Scatter(x=time_train,
                     y=np.percentile(np.gradient(y_rcp26,axis=1),50,axis=0),
                     mode='lines',
                     opacity=0.5,
                     line_color="black",
                     showlegend=False,
                     name='PISM RCP2.6 (median)')
gmedian2 = go.Scatter(x=time_train,
                     y=np.percentile(np.gradient(y_rcp85,axis=1),50,axis=0),
                     mode='lines',
                     opacity=0.5,
                     line_color="black",
                     showlegend=False,
                     name='PISM RCP8.5 (median)')

lower1 = go.Scatter(x=time_train,
                    y=np.percentile(y_rcp26,2.5,axis=0),
                    mode='lines',
                    line_color='rgba(10,10,10,0.0)',
                    showlegend=False,
                    name='PISM RCP2.6 (2.5%)')
upper1 = go.Scatter(x=time_train,
                    y=np.percentile(y_rcp26,97.5,axis=0),
                    mode='lines',
                    line_color='rgba(10,10,10,0.0)',
                    fillcolor='rgba(10,10,10,0.1)',
                    fill='tonexty',
                    showlegend=False,
                    name='PISM RCP2.6 (97.5%)')
glower1 = go.Scatter(x=time_train,
                    y=np.percentile(np.gradient(y_rcp26,axis=1),2.5,axis=0),
                    mode='lines',
                    line_color='rgba(10,10,10,0.0)',
                    showlegend=False,
                    name='PISM RCP2.6 (2.5%)')
gupper1 = go.Scatter(x=time_train,
                    y=np.percentile(np.gradient(y_rcp26,axis=1),97.5,axis=0),
                    mode='lines',
                    line_color='rgba(10,10,10,0.0)',
                    fillcolor='rgba(10,10,10,0.1)',
                    fill='tonexty',
                    showlegend=False,
                    name='PISM RCP2.6 (97.5%)')

lower2 = go.Scatter(x=time_train,
                    y=np.percentile(y_rcp85,2.5,axis=0),
                    mode='lines',
                    line_color='rgba(10,10,10,0.0)',
                    showlegend=False,
                    name='PISM RCP8.5 (2.5%)')
upper2 = go.Scatter(x=time_train,
                    y=np.percentile(y_rcp85,97.5,axis=0),
                    mode='lines',
                    line_color='rgba(10,10,10,0.0)',
                    fillcolor='rgba(10,10,10,0.1)',
                    fill='tonexty',
                    showlegend=False,
                    name='PISM RCP8.5 (97.5%)')
glower2 = go.Scatter(x=time_train,
                    y=np.percentile(np.gradient(y_rcp85,axis=1),2.5,axis=0),
                    mode='lines',
                    line_color='rgba(10,10,10,0.0)',
                    showlegend=False,
                    name='PISM RCP8.5 (2.5%)')
gupper2 = go.Scatter(x=time_train,
                    y=np.percentile(np.gradient(y_rcp85,axis=1),97.5,axis=0),
                    mode='lines',
                    line_color='rgba(10,10,10,0.0)',
                    fillcolor='rgba(10,10,10,0.1)',
                    fill='tonexty',
                    showlegend=False,
                    name='PISM RCP8.5 (97.5%)')

fig = make_subplots(rows=2, cols=1, subplot_titles=("Sea Level Rise", "Rate of Sea Level Rise"))
fig.add_trace(trace1,row=1, col=1)
fig.add_trace(trace2,row=1, col=1)
fig.add_trace(gtrace1,row=2, col=1)
fig.add_trace(gtrace2,row=2, col=1)
fig.add_trace(median1,row=1, col=1)
fig.add_trace(median2,row=1, col=1)
fig.add_trace(gmedian1,row=2, col=1)
fig.add_trace(gmedian2,row=2, col=1)
fig.add_trace(lower1,row=1, col=1)
fig.add_trace(upper1,row=1, col=1)
fig.add_trace(lower2,row=1, col=1)
fig.add_trace(upper2,row=1, col=1)
fig.add_trace(glower1,row=2, col=1)
fig.add_trace(gupper1,row=2, col=1)
fig.add_trace(glower2,row=2, col=1)
fig.add_trace(gupper2,row=2, col=1)

fig.update_layout(height=800)


def update(change):
    y1 = model_update([sia.value,ssa.value,q.value,phi.value],rcp26)
    y2 = model_update([sia.value,ssa.value,q.value,phi.value],rcp85)

app = dash.Dash()

app.layout = html.Div(children=[
       html.Div([ dcc.Graph(id="fig", figure=fig)]),
       html.Div([
           html.Div([
           html.Label('SIA'),
           dcc.Slider(
               id='slider-sia',
               min=1.2,
               max=4.8,
               step=0.1,
               value=sia_val,
               marks={i: "%.1f"%i for i in np.arange(1.2,4.9,0.4)}
           )],style={"margin": "20px 50px"}),
           html.Div([
           html.Label('SSA'),
           dcc.Slider(
               id='slider-ssa',
               min=0.42,
               max=0.8,
               step=0.01,
               value=ssa_val,
               marks={i: "%.2f"%i for i in np.arange(0.42,0.8,0.04)}
           )],style={"margin": "20px 50px"}),
           html.Div([
           html.Label('q'),
           dcc.Slider(
               id='slider-q',
               min=0.25,
               max=0.75,
               step=0.02,
               value=q_val,
               marks={i: "%.1f"%i for i in np.arange(0.3,0.8,0.1)}
           )],style={"margin": "20px 50px"}),
           html.Div([
           html.Label('phi'),
           dcc.Slider(
               id='slider-phi',
               min=5,
               max=15,
               step=0.2,
               value=phi_val,
               marks={i: str(i) for i in range(5,16)}
           )],style={"margin": "20px 50px"}),
       ])
])

@app.callback(Output('fig', 'figure'),
              Input('slider-sia', 'value'),
              Input('slider-ssa', 'value'),
              Input('slider-q', 'value'),
              Input('slider-phi', 'value'),
              State('fig', 'figure'))
def update_fig(sia,ssa,q,phi,figure):
    y1 = model_update([sia,ssa,q,phi],rcp26)
    y2 = model_update([sia,ssa,q,phi],rcp85)
    figure['data'][0]['y'] = y1
    figure['data'][1]['y'] = y2
    figure['data'][2]['y'] = np.gradient(y1)
    figure['data'][3]['y'] = np.gradient(y2)
    return figure


app.run_server(debug=True, use_reloader=False, host="192.168.1.155")  # Turn off reloader if inside Jupyte
