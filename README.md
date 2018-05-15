# (Canonical) Citation Extractor

## Status

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.35470.svg)](https://doi.org/10.5281/zenodo.35470)
[![Build Status](https://travis-ci.org/mromanello/CitationExtractor.svg?branch=ml-matcher)](https://travis-ci.org/mromanello/CitationExtractor)
[![codecov](https://codecov.io/gh/mromanello/CitationExtractor/branch/ml-matcher/graph/badge.svg)](https://codecov.io/gh/mromanello/CitationExtractor/branch/ml-matcher)

## Installation

This software supports Python version 2.7, and it was tested only on POSIX–compliant operating systems (Linux, Mac OS X, FreeBSD, etc.).

### Installing TreeTagger

The `CitationExtractor` relies on TreeTagger for the PoS tagging of input texts.

There is a handy script to install it.

To run it without having to clone this repo:

```bash
wget -O install_treetagger.sh https://raw.githubusercontent.com/mromanello/CitationExtractor/master/install_treetagger.sh
chmod a+x install_treetagger.sh
./install_treetagger.sh
rm install_treetagger.sh
```

otherwise:

```bash
git clone https://github.com/mromanello/CitationExtractor.git
cd CitationExtractor
chmod a+x install_treetagger.sh
./install_treetagger.sh
rm install_treetagger.sh
```


### With pip

To install the `CitationExtractor` first run:

    $ pip install http://www.antlr3.org/download/Python/antlr_python_runtime-3.1.3.tar.gz#egg=antlr_python_runtime-3.1.3
    $ pip install https://github.com/mromanello/treetagger-python/archive/master.zip#egg=treetagger-1.0.1

followed by:

    $ pip install citation-extractor

**NB:** the installation of all other dependencies is handled by `setup.py` but for some reason
(that I'm still trying to figure out) it does not pick up these two.

### Verify installation

To double check that everything was installed correctly, try running the following lines (it should take ~20s):

```python
from citation_extractor.settings import crfsuite
from citation_extractor.pipeline import get_extractor
extractor = get_extractor(crfsuite)
assert extractor is not None
```

If the code above runs without throwing exceptions means you managed to install the library!

## Documentation

I'm working on it ;-)

For the time being, you can find a concrete example of how to use the library in [this notebook](https://gist.github.com/mromanello/3d29add74a33da6629509742fe738ca1).
