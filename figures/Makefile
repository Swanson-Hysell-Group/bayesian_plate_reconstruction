all: uniform.pdf watson.pdf fisher.pdf one_euler_pole.pdf two_euler_poles.pdf

one_euler_pole.pdf: one_euler_pole.py
	python one_euler_pole.py

two_euler_poles.pdf: two_euler_poles.py
	python two_euler_poles.py

uniform.pdf: spherical_distributions.py
	python spherical_distributions.py

fisher.pdf: spherical_distributions.py
	python spherical_distributions.py

watson.pdf: spherical_distributions.py
	python spherical_distributions.py

.PHONY: clean
clean:
	rm -f *.pdf *.png
