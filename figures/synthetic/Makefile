all: one_euler_pole.pdf two_euler_poles.pdf age_uncertainty.pdf

one_euler_pole.pdf: invert_one_euler_pole.py
	python invert_one_euler_pole.py

one_euler_pole_scenario_b.pdf: invert_one_euler_pole_b.py
	python invert_one_euler_pole_b.py

two_euler_poles.pdf: invert_two_euler_poles.py
	python invert_two_euler_poles.py

age_uncertainty.pdf: age_uncertainty.py
	python age_uncertainty.py

.PHONY: clean
clean:
	rm -f *.pdf *.png
