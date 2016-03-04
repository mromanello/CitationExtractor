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

def prepare_for_training(doc_id, basedir):
    """
    result = [
        [
            [
                "arg1_entity":"AAUTHOR"
                ,"arg2_entity":"REFSCOPE"
                ,"concent":"AAUTHORREFSCOPE"
            ]
            ,'scope_pos'
        ]
        ,[
            [
                "arg1_entity":"REFSCOPE"
                ,"arg2_entity":"AAUTHOR"
                ,"concent":"REFSCOPEAAUTHOR"
            ]
            ,'scope_neg'
        ]
    ]
    """
    instances = []
    entities, relations = read_ann_file(doc_id, basedir)
    for arg1,arg2 in relations:
        instance.append(extract_relation_features(arg1,arg2,entities,fulltext),'scope_pos')
        instance.append(extract_relation_features(arg2,arg1,entities,fulltext),'scope_neg')
    return instances

def extract_relation_features(arg1,arg2,entities,fulltext):
    """
    the following features should be extracted:
        Arg1_entity:AAUTHOR
        Arg2_entity:REFSCOPE
        ConcEnt: AAUTHORREFSCOPE
        WordsBtw:0
        EntBtw:0 
        Thuc.=True (bow_arg1)
        1.8=True (bow_arg2)
        word_before_arg1
        word_after_arg1
        word_before_arg2
        word_after_arg2
    """
    pass

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