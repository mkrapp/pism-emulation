
timeser_files = "data/external/PISM/v2/timeser/"
path_interim   = "data/interim/"
path_processed = "data/processed/"
path_figures = "reports/figures/"

N = 10000

#rule all:
#	input:

rule prepare_inputs:
	output:
		path_interim+"runs_2300-1yr.pkl"
	params:
		inc = 1
	shell:
		"python src/data/read_pism_timeseries.py {params.inc} {output}"

rule regression:
	input:
		runs = path_interim+"runs_2300-1yr.pkl",
		script = "src/models/exact_regression.py"
	output:
		model    = "models/gp_exact.pkl",
		aux_data = path_processed+"gp_sea_level_rise_potential.pkl"
	shell:
		"python {input.script} {input.runs} y"

rule history_matching:
	input:
		aux_data = path_processed+"gp_sea_level_rise_potential.pkl",
		runs     = path_interim+"runs_2300-1yr.pkl",
		model    = "models/gp_exact.pkl",
		script   = "src/visualization/history_matching.py"
	params:
		n = N
	output:
		csv_params = path_processed+"emulator_runs_parameters.csv",
		rcp26      = path_processed+"emulator_runs_rcp26.csv",
		rcp85      = path_processed+"emulator_runs_rcp85.csv",
		rcp45      = path_processed+"emulator_runs_rcp45.csv",
		rcp60      = path_processed+"emulator_runs_rcp60.csv",
		slr        = path_figures+"gp_constrain_slr.png",
		dslr       = path_figures+"gp_constrain_dslr.png",
		parameters = path_figures+"gp_constrain_parameter.png",
#		scenarios  = path_figures+"timeseries_scenarios.png",
#		scenario1  = path_figures+"timeseries_linear_scenarios_1.png",
#		scenario2  = path_figures+"timeseries_linear_scenarios_2.png"
	shell:
		"python {input.script} {input.aux_data} {input.runs} {params.n}"

rule ranking:
	input:
		aux_data = path_processed+"gp_sea_level_rise_potential.pkl",
		runs     = path_interim+"runs_2300-1yr.pkl",
		script   = "src/visualization/ranking.py"
	output:
		slr        = path_figures+"gp_constrain_slr_pism_ranking.png",
	shell:
		"python {input.script} {input.aux_data} {input.runs}"

rule time_of_emergence:
	input:
		aux_data = path_processed+"gp_sea_level_rise_potential.pkl",
		rcp26    = path_processed+"emulator_runs_rcp26.csv",
		rcp85    = path_processed+"emulator_runs_rcp85.csv",
		script   = "src/visualization/time_of_emergence.py"
	output:
		path_figures+"toe.png"
	shell:
		"python {input.script} {input.rcp26} {input.rcp85} {input.aux_data}"

rule plot_timeseries:
	input:
		aux_data = path_processed+"gp_sea_level_rise_potential.pkl",
		runs     = path_interim+"runs_2300-1yr.pkl",
		script   = "src/visualization/plot_timeseries.py"
	output:
		path_figures+"gp_sea_level_rise_potential_panel.png"
	shell:
		"python {input.script} {input.aux_data} {input.runs}"

rule plot_warming_levels:
	input:
		rcps    = expand(path_processed+"emulator_runs_{rcp}.csv",rcp=["rcp26","rcp45","rcp60","rcp85"]),
		decades = expand(path_processed+"emulator_runs_2K-{decade}.csv",decade=[2020,2040,2060,2080,2100]),
		levels  = expand(path_processed+"emulator_runs_{level}K.csv",level=[1,2,3,4,5]),
		script  = "src/visualization/warming_levels.py"
	output:
		rcps    = path_figures+"gwl_rcps.png",
		decades = path_figures+"gwl_different_decades.png",
		levels  = path_figures+"gwl_different_warming.png"
	shell:
		"python {input.script} {input.rcps} {output.rcps} &&"
		"python {input.script} {input.decades} {output.decades} &&"
		"python {input.script} {input.levels} {output.levels}"

#rule composite_fig5:
#	shell:
#		"composite -size 60 label:'a)' reports/figures/timeseries_scenarios.png reports/figures/fig_5a.png &&"
#		"composite -size 60 label:'b)' reports/figures/gwl_rcps.png reports/figures/fig_5b.png &&"
#		"composite -size 60 label:'c)' reports/figures/timeseries_linear_scenarios_1.png reports/figures/fig_5c.png &&"
#		"composite -size 60 label:'d)' reports/figures/gwl_different_decades.png reports/figures/fig_5d.png &&"
#		"composite -size 60 label:'e)' reports/figures/timeseries_linear_scenarios_2.png reports/figures/fig_5e.png &&"
#		"composite -size 60 label:'f)' reports/figures/gwl_different_warming.png reports/figures/fig_5f.png &&"
#		"montage -mode concatenate -tile 2x3 -geometry +50+20 reports/figures/fig_5{{a,b,c,d,e,f}}.png reports/figures/ToE-fig5.png"

