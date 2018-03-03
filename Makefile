TABLE_TEX=$(wildcard data/*/*-table.tex)
FIGURES_PDF=$(TABLE_TEX:-table.tex=-results.pdf)

all: #$(FIGURES_PDF) $(TABLE_TEX)
	python fm.py --dataset assistments0 --iter 500 --d 0 --users --skills --wins --fails
	python fm.py --dataset assistments0 --iter 500 --d 0 --users --items --skills --wins --fails
	python fm.py --dataset assistments0 --iter 500 --d 1 --users --items --skills --wins --fails
	python fm.py --dataset assistments0 --iter 500 --d 2 --users --items --skills --wins --fails
	python fm.py --dataset assistments0 --iter 500 --d 5 --users --items --skills --wins --fails
	python fm.py --dataset assistments0 --iter 500 --d 0 --users --items --skills --wins --fails --item_wins --item_fails
	python fm.py --dataset assistments0 --iter 500 --d 1 --users --items --skills --wins --fails --item_wins --item_fails
	python fm.py --dataset assistments0 --iter 500 --d 2 --users --items --skills --wins --fails --item_wins --item_fails

berkeley0:
	python fm.py --dataset berkeley0 --iter 300 --d 0 --users --items
	python fm.py --dataset berkeley0 --iter 300 --d 1 --users --items
	# python fm.py --dataset berkeley0 --iter 300 --d 0 --users --skills --wins --fails
	# python fm.py --dataset berkeley0 --iter 300 --d 0 --users --items --skills --wins --fails
	# python fm.py --dataset berkeley0 --iter 300 --d 1 --users --items --skills --wins --fails
	# python fm.py --dataset berkeley0 --iter 300 --d 2 --users --items --skills --wins --fails
	# python fm.py --dataset berkeley0 --iter 300 --d 5 --users --items --skills --wins --fails
	# python fm.py --dataset berkeley0 --iter 300 --d 0 --users --items --skills --wins --fails --item_wins --item_fails
	# python fm.py --dataset berkeley0 --iter 300 --d 1 --users --items --skills --wins --fails --item_wins --item_fails

combine:
	python combine.py --dataset assistments0
	python plot.py --dataset assistments0

dummy:
	# python fm.py --iter 10 --d 0 --users --items --skills --wins --fails
	python diagram.py
	xelatex diagram_pretty

plot:
	python plot.py --dataset fraction0

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
	python fm.py --dataset fraction0 --d 0 --users --items --skills
	python fm.py --dataset fraction0 --d 0 --users --items
	python fm.py --dataset fraction0 --d 1 --users --items --skills
	python fm.py --dataset fraction0 --d 1 --users --items
	python fm.py --dataset fraction0 --d 5 --users --skills
	python fm.py --dataset fraction0 --d 5 --users --items
	python fm.py --dataset fraction0 --d 5 --users --items --skills
	python fm.py --dataset fraction0 --d 10 --users --items
	python fm.py --dataset fraction0 --d 10 --users --skills
	python fm.py --dataset fraction0 --d 10 --users --items --skills
	python fm.py --dataset fraction0 --d 50 --users --items --skills

data/*/%-results.pdf: data/*/%-table.tex
	python plot.py --dataset $*

clean:
	rm -f $(FIGURES_PDF)
