* for CICLing create a separate branch for a completely standalone citation_extractor, meaning that it should be possible to install the software in this way

	pip install -U https://github.com/mromanello/CitationExtractor/tarball/master
	
* test the installation process using `virtualenv`
* keep track of dependencies
* re-organise the logging
* deal with exceptions in `preproc.tokenize_and_POStag`

## Code Refactoring

* remove obsolete bits from module `process`

## Testing

* test that CRF++ works properly
* test `process.get_extractor(...)` method [+]
* test `process.extract_citations(...)`
