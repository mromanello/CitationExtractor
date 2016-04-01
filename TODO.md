* move `crfpp_templates` to the `data` directory
* re-organise the logging
* in `process.preproc_document` replace `guess_language` with `langid` library as it seems way more accurate (!!)

## Code Refactoring

* `CRFSuite` instead of `CRF++`: <http://sklearn-crfsuite.readthedocs.org/en/latest/> (and combine with <http://www.nltk.org/api/nltk.classify.html>)
* move active learning classes to a separate module
~~* remove obsolete bits from module `process`~~
~~* rename `process` -> `pipeline`~~
~~* in the `settings.base_settings` replace absolute paths with use of `pkg_resources`:~~
* ~~include training/test data in the `data` directory~~
~~* to try to make the `crfpp_wrap.CRF_Classifier` pickleable~~

### Refactoring CitationParser

* create a new module `ned.py` and move here:  
    - `CitationMatcher` (now in `citation_parser`)
    - `KnowledgeBase` (now in `citation_parser`)
    - in the longer-term move also the `CitationParser` and the `anltr` grammar files

## Testing

* use py.test [doku](http://pytest.org/latest/pytest.pdf)
* what to test
    * ~~creating and running a citation extractor~~
    * ~~test whether the `citation_extractor` can be pickled~~
    * use of the several classifiers (not only CRF) i.e. scikitlearnadapter
    * test that the ActiveLearner still works


