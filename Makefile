TABLE_TEX=$(wildcard data/*/*-table.tex)
FIGURES_PDF=$(TABLE_TEX:-table.tex=-results.pdf)

all: $(FIGURES_PDF) $(TABLE_TEX)
	# Last TODO
	# python fm.py --dataset assistments0 --iter 500 --d 0 --users --items  # IRT

crop:
	for x in data/*/*results.pdf; do pdfcrop $$x $$x; done

sharedtask:
	time python sharedtask.py --dataset duolingo --iter 500 --d 0 --users --items
	time python sharedtask.py --dataset duolingo --iter 500 --d 0 --users --items --item_wins --item_fails

dodo:
	python fm.py --dataset assistments0 --iter 500 --d 1 --users --items
	python fm.py --dataset assistments0 --iter 500 --d 5 --users --items  # MIRT
	python fm.py --dataset assistments0 --iter 500 --d 0 --users --skills --attempts  # AFM
	python fm.py --dataset assistments0 --iter 500 --d 0 --users --items --attempts

	python fm.py --dataset berkeley0 --iter 300 --d 0 --users --skills --attempts  # AFM
	python fm.py --dataset berkeley0 --iter 300 --d 0 --users --items --attempts

	python fm.py --dataset castor6e0 --d 0 --users --items --skills
	python fm.py --dataset castor6e0 --d 0 --users --items
	python fm.py --dataset castor6e0 --d 1 --users --items --skills
	python fm.py --dataset castor6e0 --d 1 --users --items
	python fm.py --dataset castor6e0 --d 5 --users --skills
	python fm.py --dataset castor6e0 --d 5 --users --items
	python fm.py --dataset castor6e0 --d 5 --users --items --skills
	python fm.py --dataset castor6e0 --d 10 --users --items
	python fm.py --dataset castor6e0 --d 10 --users --skills
	python fm.py --dataset castor6e0 --d 10 --users --items --skills
	# python fm.py --dataset castor6e0 --d 50 --users --items --skills



assistments: #$(FIGURES_PDF) $(TABLE_TEX)
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
	python combine.py --dataset ecpe0
	python plot.py --dataset ecpe0

dummy:
	# python fm.py --iter 10 --d 0 --users --items --skills --wins --fails
	python diagram.py
	xelatex diagram_pretty

plot:
	python combine.py --dataset ecpe0
	python plot.py --dataset ecpe0

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
	python fm.py --dataset ecpe0 --d 0 --users --items --skills
	python fm.py --dataset ecpe0 --d 0 --users --items
	python fm.py --dataset ecpe0 --d 1 --users --items --skills
	python fm.py --dataset ecpe0 --d 1 --users --items
	python fm.py --dataset ecpe0 --d 5 --users --skills
	python fm.py --dataset ecpe0 --d 5 --users --items
	python fm.py --dataset ecpe0 --d 5 --users --items --skills
	python fm.py --dataset ecpe0 --d 10 --users --items
	python fm.py --dataset ecpe0 --d 10 --users --skills
	python fm.py --dataset ecpe0 --d 10 --users --items --skills
	python fm.py --dataset ecpe0 --d 50 --users --items --skills

data/*/%-results.pdf: data/*/%-table.tex
	python combine.py --dataset $*
	python plot.py --dataset $*

clean:
	rm -f $(FIGURES_PDF)
