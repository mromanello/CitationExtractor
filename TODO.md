* move `crfpp_templates` to the `data` directory
* re-organise the logging
* in `process.preproc_document` replace `guess_language` with `langid` library as it seems way more accurate (!!)

## Code Refactoring

~~* remove obsolete bits from module `process`~~
~~* rename `process` -> `pipeline`~~
* move active learning classes to a separate module
~~* in the `settings.base_settings` replace absolute paths with use of `pkg_resources`:~~

    pkg_resources.resource_filename('citation_extractor','data/authors.csv')

* include training/test data in the `data` directory
* `CRFSuite` instead of `CRF++`: <http://sklearn-crfsuite.readthedocs.org/en/latest/> (and combine with <http://www.nltk.org/api/nltk.classify.html>)
~~* to try to make the `crfpp_wrap.CRF_Classifier` pickleable~~

## Testing

* use py.test [doku](http://pytest.org/latest/pytest.pdf)
* what to test
    * ~~creating and running a citation extractor~~
    * ~~test whether the `citation_extractor` can be pickled~~
    * use of the several classifiers (not only CRF) i.e. scikitlearnadapter
    * test that the ActiveLearner still works


