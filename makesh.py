from string import Template
import argparse


PATH = '/home/jj/ktm'


parser = argparse.ArgumentParser(description='Make bash scripts')
parser.add_argument('--datasets', type=str, nargs='+')
parser.add_argument('--dimensions', type=int, nargs='+')
options = parser.parse_args()


for dataset in options.datasets:
    for d in options.dimensions:
        prefix = '{:s}-{:d}'.format(dataset, d)
        with open('template.sh') as f:
            t = Template(f.read())
            bash = t.substitute({
                'PATH': PATH,
                'dataset': dataset,
                'd': d,
                'prefix': prefix
            })

        with open('run-ktm-{:s}.sh'.format(prefix), 'w') as f:
            f.write(bash)
