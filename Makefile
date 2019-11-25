test:
	pytest -vv --cov=citation_extractor tests/ --ignore=tests/test_eval.py
html:
	cd docs && make clean && make html && cd _build/html && python -m http.server  8001
