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

* working with many languages makes it more comlicated to work with syntactic features as chunkers do not exist for all the languages we considered ()

* the training set should contain both positive and negative examples; to create a negative example out of a positive relation, e.g. "rel(arg1,arg2)" is enough to invert it, "rel(arg2,arg1)"

    class= (scope_pos | scope_neg)

## Notes to improve the Named Entity Disambiguation

### Code

* improve the logging
* test that the code can be parallelised

### Logic

* instead of disambiguating relations first and then entities
* try to do that by following the sequence of the document
* get all the annotations for a given document, ordered as they appear...
* ... then proceed to disambiguate each annotation, using the annotation type to call appropriate function/method
* this way, neighbouring entity mentions can be used to help with the disambiguation of relations

