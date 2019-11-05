test:
	python -m pytest tests/
html:
	cd docs && make clean && make html && cd _build/html && python -m http.server  8001

