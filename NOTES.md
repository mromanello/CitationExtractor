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

* working with many languages makes it more comlicated to work with syntactic features as chunkers do not exist for all the languages we considered ()

* the training set should contain both positive and negative examples; to create a negative example out of a positive relation, e.g. "rel(arg1,arg2)" is enough to invert it, "rel(arg2,arg1)"

    class= (scope_pos | scope_neg)

class relation_extractor:
    __init__(self,classifier,train_dirs):
        """
        todo
        """
        training = [(file.replace(".ann",""),train_dir) for dir in train_dirs 
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

