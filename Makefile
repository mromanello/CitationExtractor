test:
	pytest -vv --cov=citation_extractor tests/
html:
	cd docs && make clean && make html && cd _build/html && python -m http.server  8001
