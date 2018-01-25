.PHONY: all australia_0_true australia_1_false australia_1_true australia_2_false
all: australia_0_true australia_1_false australia_1_true australia_2_false

australia_0_true : australia_apw_path.py australia_0_true_speeds.pdf australia_0_true_paths.pdf

australia_1_false : australia_apw_path.py australia_1_false_speeds.pdf australia_1_false_paths.pdf

australia_1_true : australia_apw_path.py australia_1_true_speeds.pdf australia_1_true_paths.pdf

australia_2_false : australia_apw_path.py australia_2_false_speeds.pdf australia_2_false_paths.pdf


australia_0_true_speeds.pdf: australia_apw_path.py
	python australia_apw_path.py 0 true O,180,-19
australia_0_true_paths.pdf: australia_apw_path.py
	python australia_apw_path.py 0 true O,180,-19

australia_1_false_speeds.pdf: australia_apw_path.py
	python australia_apw_path.py 1 false O,180,-19
australia_1_false_paths.pdf: australia_apw_path.py
	python australia_apw_path.py 1 false O,180,-19

australia_1_true_speeds.pdf: australia_apw_path.py
	python australia_apw_path.py 1 true O,180,-19
australia_1_true_paths.pdf: australia_apw_path.py
	python australia_apw_path.py 1 true O,180,-19

australia_2_false_speeds.pdf: australia_apw_path.py
	python australia_apw_path.py 2 false O,180,-19
australia_2_false_paths.pdf: australia_apw_path.py
	python australia_apw_path.py 2 false O,180,-19
