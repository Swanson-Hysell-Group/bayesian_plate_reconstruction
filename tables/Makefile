all: keweenawan_poles.tex australia_poles.tex

keweenawan_poles.tex: pole_means.csv keweenawan_table.py
	python keweenawan_table.py

australia_poles.tex: GPMDB_Australia_50Ma.csv australia_table.py
	python australia_table.py

clean:
	rm -f keweenawan_poles.tex keweenawan_poles.aux australia_poles.tex australia_poles.aux 
