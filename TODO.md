## Up next

- [ ] revise `pipeline` module:
  - rationale: serialize to JSON as default
  - remove dependency with `brat` code (`conll2standoff`)
  - update tests
- [ ] update TreeTagger installation script
  - and provide a version of Mac OS


## integration of ML-Matcher into the codebase

* [x] evaluation of `MLCitationMatcher` (via `tests/test_eval.py`)
* [x] parallelise train/disambiguate/feature extraction with `dask`
* [x] write test `FeatureExtractor.extract_nil` (mr)
* [x] write tests for `ned.candidates.CandidatesGenerator` (mr)
* [x] write documentation for feature functions (mf)
* [x] implement `MLCitationMatcher.train`
* [x] implement `MLCitationMatcher.classify`

## Next steps

* [ ] improve the code quality/style
* [ ] create evaluation `py.tests` for
    - [x] NER
    - [ ] RelEX (compare rule-based and ML-based extraction)
    - [x] NED
* [ ] create some stats about the traning/test corpus (**but not here, on APh corpus repo**)
    - number of entities by class
    - number of relations
    - number tokens
    - language distribution of documents

## Code Refactoring

* [ ] remove obsolete functions from `pipeline`
* to streamline installation, try to remove local dependencies:
	* [ ] add `pysuffix` to the codebase => `Utils.pysuffix` (or so)

* [ ] change the `LookupDictionary` in `Utils.FastDict` so that it gets the data directly from the Knowledge Base instead of the static file (**needs tests**)

    - put author names into a dictionary, assuring that the keys are unique
    - this code uses the new KB, not the one in `citation_extractor.ned`

    flat_author_names = {"%s$$n%i"%(author.get_urn(), i+1):name[1]
            for author in kb.get_authors()
                        for i,name in enumerate(author.get_names())  
                                            if author.get_urn() is not None}

* move `crfpp_templates` to the `data` directory
* re-organise the logging

## Testing

* [ ] rewrite tests for `pipeline` module

* write tests for:
    * [x] creating and running a citation extractor
    * [x] test whether the `citation_extractor` can be pickled
    * [x] use of the several classifiers (not only CRF) i.e. scikitlearnadapter
    * [ ] test that the ActiveLearner still works
