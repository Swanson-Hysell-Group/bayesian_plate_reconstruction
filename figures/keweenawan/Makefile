.PHONY: all keweenawan_0_true keweenawan_1_true clean
all: keweenawan_0_true keweenawan_1_true

keweenawan_0_true: keweenawan_apw_path.py keweenawan_0_true_speeds.pdf keweenawan_0_true_paths.pdf

keweenawan_1_true: keweenawan_apw_path.py keweenawan_1_true_speeds.pdf keweenawan_1_true_paths.pdf

keweenawan_0_true_speeds.pdf: keweenawan_apw_path.py
	python keweenawan_apw_path.py 0 true O,240,10

keweenawan_0_true_paths.pdf: keweenawan_apw_path.py
	python keweenawan_apw_path.py 0 true O,240,10

keweenawan_1_true_speeds.pdf: keweenawan_apw_path.py
	python keweenawan_apw_path.py 1 true O,240,10

keweenawan_1_true_paths.pdf: keweenawan_apw_path.py
	python keweenawan_apw_path.py 1 true O,240,10

clean:
	rm -f *.pickle *.pdf
