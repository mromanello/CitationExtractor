"""Config for Pypi."""

import os
from setuptools import setup, find_packages
import citation_extractor

VERSION = citation_extractor.__version__


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='citation_extractor',
    author='Matteo Romanello',
    author_email='matteo.romanello@gmail.com',
    url='http://github.com/mromanello/CitationExtractor/',
    version=VERSION,
    dependency_links=[
        'http://www.antlr3.org/download/Python/antlr_python_runtime-3.1.3.tar.gz',
        'https://github.com/mromanello/treetagger-python/tarball/master#egg=treetagger-1.0.1'
    ],
    packages=find_packages(),
    package_data={
        'citation_extractor': [
            'data/*.*',
            'crfpp_templates/*.*',
            'data/aph_corpus/goldset/ann/*.*',
            'data/aph_corpus/goldset/iob/*.*'
        ]
    },
    entry_points={
        'console_scripts': [
            'citedloci-pipeline = citation_extractor.pipeline:main',
        ]
    },
    long_description=read('README.md'),
    install_requires=[
        'hucitlib',
        'langid',
        'docopt',
        'pandas',
        'scipy',
        'pycas',
        'treetagger',
        'citation_parser>=0.4.1',
        'sklearn-crfsuite',
        'jellyfish>=0.5.6',
        'stop_words>=2015.2.23.1',
        'scikit-learn>=0.16.1'
    ]
)
