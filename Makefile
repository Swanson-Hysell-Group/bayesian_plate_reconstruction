bayesian_plate_reconstruction.pdf: bayesian_plate_reconstruction.tex bayesian_plate_reconstruction.bib figures tables
	pdflatex bayesian_plate_reconstruction
	bibtex bayesian_plate_reconstruction
	pdflatex bayesian_plate_reconstruction
	pdflatex bayesian_plate_reconstruction

.PHONY: figures
figures:
	make -C figures

.PHONY: tables
tables:
	make -C tables

.PHONY: clean
clean:
	make clean -C figures; \
	rm -f *.spl *.bbl *.blg *.aux *.log bayesian_plate_reconstruction.pdf
