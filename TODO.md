* re-organise the logging
* in `process.preproc_document` replace `guess_language` with `langid` library as it seems way more accurate (!!)

## Code Refactoring

~~* remove obsolete bits from module `process`~~
* rename `process` -> `pipeline`
* move active learning classes to a separate module

## Testing

* add tests
    - test creating and running a citation extractor
* test that the ActiveLearner still works