assistments:
	python fm_fraction.py --d 0 --users --items --attempts
	python fm_fraction.py --d 0 --users --items --wins --fails
	python fm_fraction.py --d 5 --users --items --attempts
	python fm_fraction.py --d 5 --users --items --wins --fails
	python fm_fraction.py --d 10 --users --items --attempts
	python fm_fraction.py --d 10 --users --items --wins --fails

berkeley:
	python fm_fraction.py --d 0 --users --items --skills
	python fm_fraction.py --d 0 --users --items --attempts
	python fm_fraction.py --d 0 --users --items --wins --fails
	python fm_fraction.py --d 10 --users --items --skills
	python fm_fraction.py --d 10 --users --items --attempts
	python fm_fraction.py --d 10 --users --items --wins --fails

fraction:
	python fm_fraction.py --d 0 --users --items --skills
	python fm_fraction.py --d 0 --users --items
	python fm_fraction.py --d 5 --users --skills
	python fm_fraction.py --d 5 --users --items
	python fm_fraction.py --d 5 --users --items --skills
	python fm_fraction.py --d 10 --users --items
	python fm_fraction.py --d 10 --users --skills
	python fm_fraction.py --d 10 --users --items --skills
	# python fm_fraction.py --d 50 --users --items --skills

clean:
	rm /tmp/svd/log/*
