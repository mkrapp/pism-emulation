
path_pism      = "data/external/PISM/"
path_forcing   = "data/external/gmt/"
path_interim   = "data/interim/"
model = "mlp" # ["mlp", "gp", "rf"]
path_processed = f"data/processed/{model}_"
path_figures   = f"reports/figures/{model}_"

N = 10_000

SCEN   = ["rcp26","rcp45","rcp60","rcp85"]
DECADE = [2020,2040,2060,2080,2100]
LEVEL  = [1,2,3,4,5]

SIA = ["1.2","2.4","4.8"]
SSA = ["0.42","0.6","0.8"]
Q   = ["0.25","0.5","0.75"]
PHI = ["5","10","15"]

y     = "sea_level_rise_potential"

rule run_all:
	input:
		path_figures+"timeseries_linear_scenarios_1.png",
		path_figures+"timeseries_linear_scenarios_2.png",
		path_figures+"timeseries_scenarios.png",
		path_figures+"constrain_slr_pism_ranking.png",
		path_figures+"toe.png",
		path_figures+y+"_panel.png",
		path_figures+"gwl_rcps.png",
		path_figures+"gwl_different_decades.png",
		path_figures+"gwl_different_warming.png",
		path_figures+"timeseries_scenarios_filtered.png",
		path_figures+"covariance_matrix.png",
		path_figures+"pdfs.gif",
		path_figures+"slr_emergence.png",
		path_figures+"slr_emergence.pdf",
		path_figures+"parameters.png"

rule download_PISM:
	output:
		path_pism+"pism_inputs.tar.gz"
	shell:
		"wget https://osf.io/exuca/download -O {output} && tar xfvz {output}"

rule unzip_PISM:
	input:
		path_pism+"pism_inputs.tar.gz"
	output:
		path_pism+"timeser_NorESM1-M-{scen}-sia{sia}_ssa{ssa}_q{q}_phi{phi}-2016-2300.nc"
	shell:
		"gunzip -c compress/timeser_NorESM1-M-{wildcards.scen}-sia{wildcards.sia}_ssa{wildcards.ssa}_q{wildcards.q}_phi{wildcards.phi}-2016-2300.nc.gz > {output}"

rule download_rcps:
	output:
		path_forcing+"global_tas_Amon_NorESM1-M_{scen}_r1i1p1.dat"
	shell:
		"wget http://climexp.knmi.nl/CMIP5/Tglobal/global_tas_Amon_NorESM1-M_{wildcards.scen}_r1i1p1.dat -O {output}"

rule prepare_inputs:
	input:
		pism   = expand(path_pism+"timeser_NorESM1-M-{scen}-sia{sia}_ssa{ssa}_q{q}_phi{phi}-2016-2300.nc",scen=["rcp26","rcp85"],sia=SIA,ssa=SSA,q=Q,phi=PHI),
		gmt    = expand(path_forcing+"global_tas_Amon_NorESM1-M_{scen}_r1i1p1.dat",scen=[SCEN[i] for i in [0,3]]),
		script = "src/data/read_inputs.py"
	output:
		path_interim+"runs_2300-1yr.pkl"
	params:
		dt = 1
	shell:
		"python {input.script} --dt {params.dt} --output {output} --rcps {input.gmt}"

rule regression:
	input:
		runs   = path_interim+"runs_2300-1yr.pkl",
		script = "src/models/run_regression.py"
	output:
		model_out = "models/"+model+".pkl",
		idx_train = path_interim+"idx_train.txt",
		aux_data  = path_processed+y+".pkl"
	shell:
		"python {input.script} --input {input.runs} --model {model}"

rule history_matching:
	input:
		aux_data  = path_processed+y+".pkl",
		runs      = path_interim+"runs_2300-1yr.pkl",
		model_out = "models/"+model+".pkl",
		other     = path_interim+"other_scenarios.pkl",
		script    = "src/visualization/history_matching.py"
	params:
		n = N
	output:
		emu_params = path_processed+"emulator_runs_parameters.csv",
		matched    = path_processed+"emulator_runs_pism_matched.csv",
		rcps       = expand(path_processed+"emulator_runs_{scen}.csv",scen=SCEN),
		decades    = expand(path_processed+"emulator_runs_2K-{decade}.csv",decade=DECADE),
		levels     = expand(path_processed+"emulator_runs_{level}K.csv",level=LEVEL),
		slr        = path_figures+"constrain_slr.png",
		dslr       = path_figures+"constrain_dslr.png",
		parameters = path_figures+"constrain_parameter.png",
	shell:
		"python {input.script} --model_output {input.aux_data} --input {input.runs} --nrandom {params.n} --model {model}"

rule ranking:
	input:
		aux_data = path_processed+y+".pkl",
		runs     = path_interim+"runs_2300-1yr.pkl",
		matched  = path_processed+"emulator_runs_pism_matched.csv",
		script   = "src/visualization/ranking.py"
	output:
		slr      = path_figures+"constrain_slr_pism_ranking.png",
	shell:
		"python {input.script} --model_output {input.aux_data} --input {input.runs} --model {model}"

rule time_of_emergence:
	input:
		aux_data = path_processed+y+".pkl",
		rcp26    = path_processed+"emulator_runs_rcp26.csv",
		rcp85    = path_processed+"emulator_runs_rcp85.csv",
		script   = "src/visualization/time_of_emergence.py"
	output:
		path_figures+"toe.png"
	shell:
		"python {input.script} --rcp26 {input.rcp26} --rcp85 {input.rcp85} --model_output {input.aux_data} --model {model}"

rule plot_scenarios:
	input:
		aux_data = path_processed+y+".pkl",
		runs     = path_interim+"runs_2300-1yr.pkl",
		gmt      = expand(path_forcing+"global_tas_Amon_NorESM1-M_{scen}_r1i1p1.dat",scen=[SCEN[i] for i in [1,2]]),
		script   = "src/visualization/plot_scenarios.py"
	output:
		scen1 = path_figures+"timeseries_linear_scenarios_1.png",
		scen2 = path_figures+"timeseries_linear_scenarios_2.png",
		rcps  = path_figures+"timeseries_scenarios.png",
		other = path_interim+"other_scenarios.pkl"
	shell:
		"python {input.script} --model_output {input.aux_data} --input {input.runs} --rcps {input.gmt} --output {output.other} --model {model}"

rule plot_timeseries:
	input:
		aux_data = path_processed+y+".pkl",
		runs     = path_interim+"runs_2300-1yr.pkl",
		matched  = path_processed+"emulator_runs_pism_matched.csv",
		script   = "src/visualization/plot_timeseries.py"
	output:
		path_figures+y+"_panel.png"
	shell:
		"python {input.script} --model_output {input.aux_data} --input {input.runs} --model {model}"

rule plot_filtered_timeseries:
	input:
		gmt    = expand(path_forcing+"global_tas_Amon_NorESM1-M_{scen}_r1i1p1.dat",scen=SCEN),
		script = "src/visualization/plot_filtered_timeseries.py"
	output:
		path_figures+"timeseries_scenarios_filtered.png"
	shell:
		"python {input.script}"

rule plot_covariance:
	input:
		idx_train = path_interim+"idx_train.txt",
		script    = "src/visualization/plot_covariance.py"
	output:
		path_figures+"covariance_matrix.png"
	shell:
		"python {input.script}"

rule plot_pdfs:
	input:
		rcp26  = path_processed+"emulator_runs_rcp26.csv",
		rcp85  = path_processed+"emulator_runs_rcp85.csv",
		script = "src/visualization/plot_pdfs.py"
	output:
		path_figures+"pdfs.gif",
		path_figures+"slr_emergence.png",
		path_figures+"slr_emergence.pdf",
	shell:
		"python {input.script} {input.rcp26} {input.rcp85}"

rule plot_parameters:
	input:
		emu_params = path_processed+"emulator_runs_parameters.csv",
		rcp85      = path_processed+"emulator_runs_rcp85.csv",
		script     = "src/visualization/plot_parameters.py"
	output:
		path_figures+"parameters.png"
	shell:
		"python {input.script}"

rule plot_warming_levels:
	input:
		rcps    = expand(path_processed+"emulator_runs_{scen}.csv",scen=SCEN),
		decades = expand(path_processed+"emulator_runs_2K-{decade}.csv",decade=DECADE),
		levels  = expand(path_processed+"emulator_runs_{level}K.csv",level=LEVEL),
		script  = "src/visualization/warming_levels.py"
	output:
		rcps    = path_figures+"gwl_rcps.png",
		decades = path_figures+"gwl_different_decades.png",
		levels  = path_figures+"gwl_different_warming.png"
	shell:
		"python {input.script} {input.rcps} {output.rcps} &&"
		"python {input.script} {input.decades} {output.decades} &&"
		"python {input.script} {input.levels} {output.levels}"
