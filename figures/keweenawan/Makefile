.PHONY: all keweenawan_0_true keweenawan_1_true keweenawan_1_false keweenawan_2_false keweenawan_2_true clean
all: keweenawan_0_true keweenawan_1_true keweenawan_1_false keweenawan_2_false keweenawan_2_true

keweenawan_0_true: keweenawan_apw_path.py keweenawan_0_true_speeds.pdf keweenawan_0_true_paths.pdf

keweenawan_1_true: keweenawan_apw_path.py keweenawan_1_true_speeds.pdf keweenawan_1_true_paths.pdf

keweenawan_1_false: keweenawan_apw_path.py keweenawan_1_false_speeds.pdf keweenawan_1_false_paths.pdf

keweenawan_2_false: keweenawan_apw_path.py keweenawan_2_false_speeds.pdf keweenawan_2_false_paths.pdf

keweenawan_2_true: keweenawan_apw_path.py keweenawan_2_true_speeds.pdf keweenawan_2_true_paths.pdf

keweenawan_0_true_speeds.pdf: keweenawan_apw_path.py
	python keweenawan_apw_path.py 0 true O,240,10

keweenawan_0_true_paths.pdf: keweenawan_apw_path.py
	python keweenawan_apw_path.py 0 true O,240,10

keweenawan_1_true_speeds.pdf: keweenawan_apw_path.py
	python keweenawan_apw_path.py 1 true O,240,10

keweenawan_1_true_paths.pdf: keweenawan_apw_path.py
	python keweenawan_apw_path.py 1 true O,240,10

keweenawan_1_false_speeds.pdf: keweenawan_apw_path.py
	python keweenawan_apw_path.py 1 false O,240,10

keweenawan_1_false_paths.pdf: keweenawan_apw_path.py
	python keweenawan_apw_path.py 1 false O,240,10

keweenawan_2_false_speeds.pdf: keweenawan_apw_path.py
	python keweenawan_apw_path.py 2 false O,240,10

keweenawan_2_false_paths.pdf: keweenawan_apw_path.py
	python keweenawan_apw_path.py 2 false O,240,10

keweenawan_2_true_paths.pdf: keweenawan_apw_path.py
	python keweenawan_apw_path.py 2 true O,240,10

keweenawan_2_true_speeds.pdf: keweenawan_apw_path.py
	python keweenawan_apw_path.py 2 true O,240,10

clean:
	rm -f *.pickle *.pdf
