all: uniform.pdf watson.pdf fisher.pdf

uniform.pdf: spherical_distributions.py
	python spherical_distributions.py

fisher.pdf: spherical_distributions.py
	python spherical_distributions.py

watson.pdf: spherical_distributions.py
	python spherical_distributions.py

.PHONY: clean
clean:
	rm -f *.pdf *.png
