TABLE_TEX=$(wildcard data/*/*-table.tex)
FIGURES_PDF=$(TABLE_TEX:-table.tex=-results.pdf)

all: $(FIGURES_PDF) $(TABLE_TEX)
	# Last TODO
	# python fm.py --dataset assistments0 --iter 500 --d 0 --users --items  # IRT

test:
	# python fm.py --base_dir /Users/jilljenn --libfm code/libfm --dataset assistments42 --d 0 --users --items
	# python fm.py --base_dir /Users/jilljenn --libfm code/libfm --dataset assistments42 --d 0 --users --items --wins --fails
	# python fm.py --base_dir /Users/jilljenn --libfm code/libfm --dataset assistments42 --d 0 --items --skills --wins --fails
	# python fm.py --base_dir /Users/jilljenn --libfm code/libfm --dataset assistments42 --d 0 --skills --wins --fails
	# python fm.py --base_dir /Users/jilljenn --libfm code/libfm --dataset fraction42 --d 0 --users --items --skills
	# python fm.py --base_dir /Users/jilljenn --libfm code/libfm --dataset fraction42 --d 5 --users --skills
	# python fm.py --base_dir /Users/jilljenn --libfm code/libfm --dataset fraction42 --d 5 --items --skills
	python fm.py --base_dir /Users/jilljenn --libfm code/libfm --dataset fraction42 --d 20 --items --skills
	# python fm.py --base_dir /Users/jilljenn --libfm code/libfm --dataset ecpe42 --d 0 --users --items --skills
	# python fm.py --dataset berkeley42 --d 0 --skills --wins --fails
	# python fm.py --dataset berkeley42 --d 0 --items --skills --wins --fails  # PFA + item == Best
	# python fm.py --dataset berkeley42 --d 0 --users --items --skills --wins --fails
	# python fm.py --dataset berkeley42 --d 8 --users --items
	# python fm.py --dataset berkeley42 --d 8 --items --skills --wins --fails
	# python fm.py --dataset berkeley42 --d 0 --items --skills --wins --fails

crop:
	for x in data/*/*results.pdf; do pdfcrop $$x $$x; done

reverse:
	time python sharedtask.py --dataset reverse --iter 500 --d 0 --users --items
	# time python sharedtask.py --dataset reverse --iter 500 --d 5 --users --items

sharedtask:
	# time python sharedtask.py --dataset duolingo --iter 500 --d 0 --users --items
	time python sharedtask.py --dataset duolingo --iter 200 --d 1 --users --items
	time python sharedtask.py --dataset duolingo --iter 200 --d 2 --users --items

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

results:
	python combine.py --datasets assistments42 berkeley42 castor6e42 ecpe42 fraction42 timss200342
	python results.py

dummy:
	# python fm.py --iter 10 --d 0 --users --items --skills --wins --fails
	python diagram.py
	xelatex diagram_pretty

plot:
	python plot.py --dataset assistments42-kiloboss
	# python plot.py --dataset berkeley42
	# python plot.py --dataset castor6e42
	# python plot.py --dataset ecpe42
	# python plot.py --dataset fraction42
	# python plot.py --dataset timss200342

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

bash:
	python makesh.py --datasets assistments42 berkeley42 castor6e42 ecpe42 fraction42 timss200342 --dimensions 0 5 10 20

push:
	rsync -avz --progress --partial data/*42 raiden:ktm/data
	rsync -avz --progress --partial run*sh raiden:ktm/scripts

pull:
	rsync -avz raiden:ktm/data/* data  # --exclude=X.npz --exclude=rlog.csv
