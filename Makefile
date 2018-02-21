all:
	# python fm_fraction.py --d 0 --users --items --skills
	# python fm_fraction.py --d 0 --users --items
	# python fm_fraction.py --d 5 --users --skills
	# python fm_fraction.py --d 5 --users --items
	# python fm_fraction.py --d 5 --users --items --skills
	# python fm_fraction.py --d 10 --users --items
	# python fm_fraction.py --d 10 --users --skills
	# python fm_fraction.py --d 10 --users --items --skills
	# python fm_fraction.py --d 50 --users --items --skills

clean:
	rm /tmp/svd/log/*
