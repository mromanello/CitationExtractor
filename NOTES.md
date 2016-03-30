where I left: try to provide the module with minimum data and directory structure necessary to run some tests. 

## Tests

* [testing good practices](http://pytest.org/latest/goodpractises.html)

## Distributing the package

* see <http://pythonhosted.org/setuptools/setuptools.html>

## Installation problems:

to install SciPy on Ubuntu one needs:
>>>>>>> 50b3d56ee26142cac9842b2a3ad49bd14de4e602

    sudo apt-get install gfortran libopenblas-dev liblapack-dev

then SciPy, then scikit-learn

## Notes to implement Supervised Relation Detection

    import citation_extractor
    import pkg_resources
    from citation_extractor.relex import prepare_for_training
    from citation_extractor.pipeline import read_ann_file_new
    from nltk.classify.scikitlearn import SklearnClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import LinearSVC
    dir = pkg_resources.resource_filename('citation_extractor','data/aph_corpus/goldset/ann/')
    files = [file.replace('-doc-1.ann','') for file in pkg_resources.resource_listdir('citation_extractor','data/aph_corpus/goldset/ann/') if '.ann' in file]
    train_data = reduce(lambda x, y: x+y,[prepare_for_training(file,dir) for file in files])
    #skl_classifier = SklearnClassifier(RandomForestClassifier(verbose=True,n_jobs=7),sparse=False)
    skl_classifier = SklearnClassifier(LinearSVC(verbose=True))
    skl_classifier.train(train_data)
    test_data = [featureset for featureset in prepare_for_training(files[19],dir,output_class_label=False)]
    test_data = reduce(lambda x, y: x+y,[prepare_for_training(file,dir,output_class_label=False) for file in files[:100]])
    [(i,result,test_data[i]) for i,result in enumerate(skl_classifier.classify_many(test_data)) if result=="scope_pos"]

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

