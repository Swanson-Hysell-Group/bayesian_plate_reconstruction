.PHONY: all australia_1 australia_2
all: australia_1 australia_2

australia_1: australia_apw_path.py australia_speeds_1.pdf australia_poles_1.pdf australia_paths_1.pdf

australia_2: australia_apw_path.py australia_speeds_2.pdf australia_poles_2.pdf australia_paths_2.pdf


australia_speeds_1.pdf: australia_apw_path.py
	python australia_apw_path.py 1 O,180,-19
australia_poles_1.pdf: australia_apw_path.py
	python australia_apw_path.py 1 O,180,-19
australia_paths_1.pdf: australia_apw_path.py
	python australia_apw_path.py 1 O,180,-19

australia_speeds_2.pdf: australia_apw_path.py
	python australia_apw_path.py 2 O,180,-19
australia_poles_2.pdf: australia_apw_path.py
	python australia_apw_path.py 2 O,180,-19
australia_paths_2.pdf: australia_apw_path.py
	python australia_apw_path.py 2 O,180,-19
