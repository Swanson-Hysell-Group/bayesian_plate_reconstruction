all: one_euler_pole.pdf two_euler_poles.pdf

one_euler_pole.pdf: invert_one_euler_pole.py
	python invert_one_euler_pole.py

two_euler_poles.pdf: invert_two_euler_poles.py
	python invert_two_euler_poles.py

.PHONY: clean
clean:
	rm -f *.pdf *.png
