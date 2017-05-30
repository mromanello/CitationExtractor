## Next steps

* create evaluation `py.tests` for NER, RelEX and (as soon as possible) NED
    - k-fold cross evaluation
    - this way evaluations can be ran every time e.g. a feature extraction function is changed/introduced
    - write results to disk so that they can be inspected e.g. via brat
    - for RelEx: compare rule-based and ML-based extraction 
* create some stats about the traning/test corpus
    - number of entities by class
    - number of relations
    - number tokens
    - language distribution of documents

## Code Refactoring

* to streamline installation, try to remove local dependencies:
	* add `pysuffix` to the codebase => `Utils.pysuffix` (or os)

* change the `LookupDictionary` in `Utils.FastDict` so that it gets the data directly from the Knowledge Base instead of the static file (**needs tests**)

    - put author names into a dictionary, assuring that the keys are unique
    - this code uses the new KB, not the one in `citation_extractor.ned`
    
    flat_author_names = {"%s$$n%i"%(author.get_urn(), i+1):name[1] 
            for author in kb.get_authors() 
                        for i,name in enumerate(author.get_names())  
                                            if author.get_urn() is not None}

* `CRFSuite` instead of `CRF++`: <http://sklearn-crfsuite.readthedocs.org/en/latest/> (and combine with <http://www.nltk.org/api/nltk.classify.html>)
* move `crfpp_templates` to the `data` directory
* re-organise the logging

* ~~in `process.preproc_document` replace `guess_language` with `langid` library as it seems way more accurate (!!)~~
* ~~move active learning classes from `Utils.aph_corpus` to a separate module~~
~~* remove obsolete bits from module `process`~~
~~* rename `process` -> `pipeline`~~
~~* in the `settings.base_settings` replace absolute paths with use of `pkg_resources`:~~
* ~~include training/test data in the `data` directory~~
~~* to try to make the `crfpp_wrap.CRF_Classifier` pickleable~~

### Refactoring CitationParser

* ~create a new module `ned.py` and move here:~  
    ~- `CitationMatcher` (now in `citation_parser`)~
    ~- `KnowledgeBase` (now in `citation_parser`)~
    ~- in the longer-term move also the `CitationParser` and the `anltr` grammar files~

## Testing


* write tests for:
    * ~~creating and running a citation extractor~~
    * ~~test whether the `citation_extractor` can be pickled~~
    * use of the several classifiers (not only CRF) i.e. scikitlearnadapter
    * test that the ActiveLearner still works
* ~~use py.test [doku](http://pytest.org/latest/pytest.pdf)~~


