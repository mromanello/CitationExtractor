### 1.7.x

- added library documentation
- MLCitationMatcher by [@mfilippo](http://github.com/mfilippo/)
- started to move away from brat standoff format as the default output

### 31.01.2018 1.6.x @mr56k

- removed the library `CRFPP` as a dependency, and replaced with the `sklearn`- compatible `sklearn-crfsuite`.
- streamlined the installation process: the software is now installable via `pip` (fixed bug [\#18](https://github.com/mromanello/CitationExtractor/issues/18))

### 28.06.2017 1.5.1 @mr56k

- added support for CRFSuite, an implementation of CRF compatible with sklearn
- added test `tests/test_eval_ner.py` to run the evaluation of various ML models for the NER step

### 28.06.2017 1.5.0 @mr56k

- substantial refactoring of `citation_extractor.ned.matchers.CitationMatcher`
- for the record: this was the version of the code used by M. Filipponi ([@mfilippo](http://github.com/mfilippo/))
