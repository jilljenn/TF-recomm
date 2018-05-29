from config import ARTICLE_FOLDER
from string import Template
import os.path

with open('paper.tex') as f:
    template = Template(f.read())

with open(os.path.join(ARTICLE_FOLDER, 'results.tex'), 'w') as f:
    for dataset in ['assistments0', 'berkeley0', 'ecpe0', 'fraction0', 'timss20030']:
        f.write(template.substitute(dataset=dataset, Dataset=dataset[:-1].title()))
