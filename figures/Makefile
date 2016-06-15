all: uniform.pdf watson.pdf fisher.pdf one_euler_pole.pdf two_euler_poles.pdf

one_euler_pole.pdf: invert_one_euler_pole.py
	python invert_one_euler_pole.py

two_euler_poles.pdf: invert_two_euler_poles.py
	python invert_two_euler_poles.py

uniform.pdf: spherical_distributions.py
	python spherical_distributions.py

fisher.pdf: spherical_distributions.py
	python spherical_distributions.py

watson.pdf: spherical_distributions.py
	python spherical_distributions.py

.PHONY: clean
clean:
	rm -f *.pdf *.png
