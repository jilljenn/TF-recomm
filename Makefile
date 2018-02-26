TABLE_TEX=$(wildcard data/*/*-table.tex)
FIGURES_PDF=$(TABLE_TEX:-table.tex=-results.pdf)

all: $(FIGURES_PDF) $(TABLE_TEX)
	# python fm_fraction.py --d 0 --users --skills --attempts
	# python fm_fraction.py --d 0 --users --skills --wins --fails
	# python fm_fraction.py --d 5 --users --skills --wins --fails

assistments:
	python fm_fraction.py --d 0 --users --items --skills --wins --fails
	python fm_fraction.py --d 5 --users --items --skills --wins --fails
	python fm_fraction.py --d 0 --users --skills --wins --fails
	python fm_fraction.py --d 5 --users --skills --wins --fails
	# python fm_fraction.py --d 0 --users --items --attempts
	# python fm_fraction.py --d 0 --users --items --wins --fails
	# python fm_fraction.py --d 5 --users --items --attempts
	# python fm_fraction.py --d 5 --users --items --wins --fails
	# python fm_fraction.py --d 10 --users --items --attempts
	# python fm_fraction.py --d 10 --users --items --wins --fails

berkeley:
	# python fm_fraction.py --dataset berkeley2 --d 0 --users --items
	# python fm_fraction.py --d 0 --users --items --attempts
	# python fm_fraction.py --dataset berkeley2 --d 0 --users --skills --wins --fails
	# python fm_fraction.py --dataset berkeley2 --d 0 --users --items --skills --wins --fails
	python fm_fraction.py --dataset berkeley2 --d 10 --users --items --skills --wins --fails
	# python fm_fraction.py --d 10 --users --items --skills
	# python fm_fraction.py --d 10 --users --items --attempts
	# python fm_fraction.py --d 10 --users --items --wins --fails

fraction:
	# python fm_fraction.py --d 0 --users --items --skills
	# python fm_fraction.py --d 0 --users --items
	# python fm_fraction.py --d 5 --users --skills
	# python fm_fraction.py --d 5 --users --items
	# python fm_fraction.py --d 5 --users --items --skills
	# python fm_fraction.py --d 10 --users --items
	# python fm_fraction.py --d 10 --users --skills
	python fm_fraction.py --d 10 --users --items --skills
	# python fm_fraction.py --d 50 --users --items --skills

data/*/%-results.pdf: data/*/%-table.tex
	python plot.py --dataset $*

clean:
	rm -f $(FIGURES_PDF)
