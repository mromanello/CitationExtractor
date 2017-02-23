import os
from setuptools import setup, find_packages
import citation_extractor

VERSION = citation_extractor.__version__
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(name='citation_extractor'
	,author='Matteo Romanello'
	,author_email='matteo.romanello@gmail.com'
	,url='http://github.com/mromanello/CitationExtractor/'
    ,version=VERSION
    ,packages=find_packages()
    ,package_data={'citation_extractor': ['data/*.*'
                                          ,'data/aph_corpus/goldset/ann/*.*'
                                          ,'data/aph_corpus/goldset/iob/*.*']}
    ,long_description=read('README.md')
    #,install_requires=[
    #    'citation-extractor-dependencies'
    #    ,'guess_language'
    #    ,'mecab-python'
    #    ,'nltk'
    #    ,'scikit-learn'
    #    ,'treetagger'
    #    ,'pandas'
    #    ]
)
