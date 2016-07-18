all: figures tables

.PHONY: figures
figures:
	make -C figures

.PHONY: tables
tables:
	make -C tables

.PHONY: clean
clean:
	make clean -C figures; \
	make clean -C tables; \
	rm -f *.spl *.bbl *.blg *.aux *.log bayesian_plate_reconstruction.pdf
