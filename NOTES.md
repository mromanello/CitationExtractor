where I left: try to provide the module with minimum data and directory structure necessary to run some tests.

## Tests

* [testing good practices](http://pytest.org/latest/goodpractises.html)

## Distributing the package

* see <http://pythonhosted.org/setuptools/setuptools.html>

## Installation problems:

to install SciPy on Ubuntu one needs:

    sudo apt-get install gfortran libopenblas-dev liblapack-dev

then SciPy, then scikit-learn

## Notes to implement Supervised Relation Detection

* working with many languages makes it more complicated to work with
syntactic features as chunkers do not exist for all the languages we considered ()

* the training set should contain both positive and negative examples; to create a negative example out of a positive relation, e.g. "rel(arg1,arg2)" is enough to invert it, "rel(arg2,arg1)"

    class= (scope_pos | scope_neg)

## `ML CitationMatcher`

cfr [this thread in SO](https://stackoverflow.com/questions/15111408/how-does-sklearn-svm-svcs-function-predict-proba-work-internally)

to output a probability for each classification by SVM pass `parallel=True`
when

```python
self._classifier = svm.SVC(
    kernel='linear',
    C=C,
    cache_size=cache_size
)
```

return the probabilities from `citation_extractor.ned.ml::predict()`
