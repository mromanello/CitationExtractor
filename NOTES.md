where I left: try to provide the module with minimum data and directory structure necessary to run some tests. 

## Tests

* use `py.test` to run the tests
* combine standlone and doctests, depending from the context
* [testing good practices](http://pytest.org/latest/goodpractises.html)
* <https://pytest.org/latest/getting-started.html>

## Distributing the package

* see <http://pythonhosted.org/setuptools/setuptools.html>

## Installation problems:

to install SciPy on Ubuntu one needs:

    sudo apt-get install gfortran libopenblas-dev liblapack-dev

then SciPy, then scikit-learn

## Notes to implement Supervised Relation Detection

* working with many languages makes it more complicated to work with syntactic features as chunkers do not exist for all the languages we considered ()

* the training set should contain both positive and negative examples; to create a negative example out of a positive relation, e.g. "rel(arg1,arg2)" is enough to invert it, "rel(arg2,arg1)"

    class= (scope_pos | scope_neg)


class relation_extractor:
    __init__(self,classifier,train_dirs):
        """
        todo
        """
        training = [(file.replace(".ann",""),train_dir) for dir in train_dir 
                        for file in glob.glob("%s*.ann"%dir)]
        training_instances = [prepare_for_training(doc_id,base_dir) 
                                        for doc_id,based_dir in doc_ids]
        self.classifier.train(training_instances)
    extract(self,entities,fulltext):
        """
        todo
        """
        relations = []
        for candidate in itertools.combinations(entites,2):
            arg1 = candidate[0]
            arg2 = candidate[1]
            feature_set = extract_relation_features(arg1,arg2,entities,fulltext)
            label = self.classifier.classify(feature_set)
            if(label=="scope_pos"):
                relations.append((arg1,arg2,label))
        return relations

* when detecting relations it is necessary to compare all pairs of entities
* to find all unique pairs (combinations) in a list with python:

    import itertools
    my_list = [1,2,3,4]
    for p in itertools.combinations(my_list,2):
        print p