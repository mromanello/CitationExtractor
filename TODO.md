* for CICLing create a separate branch for a completely standalone citation_extractor, meaning that it should be possible to install the software in this way

	pip install -U https://github.com/mromanello/CitationExtractor/tarball/master
	
* test the installation process using `virtualenv`

* possible *new* layout

	--citation_extractor
		--__init_.py
		--settings
			--default_settings.py
			--base_settings.py
		--tests
	--miguno
		--__init__.py
	--pysuffix
		--__init__.py
	--setup.py
	--LICENSE.txt
	--README.md
	--INSTALL.txt