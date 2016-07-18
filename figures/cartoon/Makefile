all: distributions.pdf paleomagnetic_euler_pole.png

distributions.pdf: spherical_distributions.py
	python spherical_distributions.py

paleomagnetic_euler_pole.png: paleomagnetic_euler_pole.py
	python paleomagnetic_euler_pole.py

.PHONY: clean
clean:
	rm -f *.pdf *.png
