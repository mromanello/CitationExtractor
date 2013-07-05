* for CICLing create a separate branch for a completely standalone citation_extractor, meaning that it should be possible to install the software in this way

	pip install -U https://github.com/mromanello/CitationExtractor/tarball/master
	
* test the installation process using `virtualenv`
* change the debug log for `FeatureExtractor`
* deal with exceptions in `preproc.tokenize_and_POStag`