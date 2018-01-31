# (Canonical) Citation Extractor

## Status

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.35470.svg)](https://doi.org/10.5281/zenodo.35470)
[![Build Status](https://travis-ci.org/mromanello/CitationExtractor.svg?branch=master)](https://travis-ci.org/mromanello/CitationExtractor)
[![codecov](https://codecov.io/gh/mromanello/CitationExtractor/branch/master/graph/badge.svg)](https://codecov.io/gh/mromanello/CitationExtractor)

## Installation

This software supports Python version 2.7, and it was tested only on POSIX–compliant operating systems (Linux, Mac OS X, FreeBSD, etc.).

To install the `CitationExtractor` first run:

    $ pip install http://www.antlr3.org/download/Python/antlr_python_runtime-3.1.3.tar.gz#egg=antlr_python_runtime-3.1.3
    $ pip install https://github.com/mromanello/treetagger-python/archive/master.zip#egg=treetagger-1.0.1

followed by:

    $ pip install citation-extractor

**NB:** the installation of all other dependencies is handled by `setup.py` but for some reason
(that I'm still trying to figure out) it does not pick up these two.
