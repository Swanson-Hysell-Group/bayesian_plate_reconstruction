.PHONY: all keweenawan_1 keweenawan_2
all: keweenawan_1 keweenawan_2

keweenawan_1: keweenawan_apw_path.py keweenawan_speeds_1.pdf keweenawan_paths_1.pdf

keweenawan_2: keweenawan_apw_path.py keweenawan_speeds_2.pdf keweenawan_paths_2.pdf

keweenawan_speeds_1.pdf: keweenawan_apw_path.py
	python keweenawan_apw_path.py 1 O,240,10
keweenawan_paths_1.pdf: keweenawan_apw_path.py
	python keweenawan_apw_path.py 1 O,240,10

keweenawan_speeds_2.pdf: keweenawan_apw_path.py
	python keweenawan_apw_path.py 2 O,240,10
keweenawan_paths_2.pdf: keweenawan_apw_path.py
	python keweenawan_apw_path.py 2 O,240,10
