import os
from setuptools import setup, find_packages
import citation_extractor

VERSION = citation_extractor.__version__
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(name='crex',
	author='Matteo Romanello',
	author_email='matteo.romanello@gmail.com',
	url='http://github.com/mromanello/CRefEx/',
    version=VERSION,
    packages=find_packages(),
    include_package_data=True,
    #package_dir={'crex': 'crex'},
    package_data={'citation_extractor': ['data/*.*']},
    long_description=read('README.md'),
    #install_requires=['partitioner','CRFPP'],
    zip_safe=False,
)
