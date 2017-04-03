# -*- coding: utf-8 -*-
# author: Matteo Romanello, matteo.romanello@gmail.com

"""
TODO
"""
import sys,logging,re
import os
import glob
from citation_extractor.core import *
from citation_extractor.crfpp_wrap import CRF_classifier
from citation_extractor.Utils.IO import read_ann_file, read_ann_file_new, init_logger
from Utils import IO
from miguno.partitioner import *
from miguno.crossvalidationdataconstructor import *
import pprint

global logger

class SimpleEvaluator(object):
    """
    >>> import settings #doctest: +SKIP
    >>> extractor_1 = citation_extractor(settings) #doctest: +SKIP
    >>> se = SimpleEvaluator([extractor_1,],iob_file="data/75-02637.iob") #doctest: +SKIP
    >>> se = SimpleEvaluator([extractor_1,],["/Users/56k/phd/code/APh/corpus/by_collection/C2/",]) #doctest: +SKIP
    >>> print se.eval() #doctest: +SKIP
    
    TODO: there should be a test also for IOB files w/ POS tag column
    
    """
    def __init__(self,extractors,iob_directories=[],iob_file=None,label_index=-1):
        """
        Args:
            extractors:
                the list of canonical citation extractors to evaluate
            iob_test_file: 
                the file in IOB format to be used for testing and evaluating the extactors
        """
        # read the test instances from a list of directories containing the test data
        import logging
        self.logger = logging.getLogger("CREX.SIMPLEVAL")
        if(iob_file is None):
            self.logger.debug(iob_directories)
            data = []
            for directory in iob_directories:
                data += IO.read_iob_files(directory,".txt")
            self.test_instances = data
        else:
            self.test_instances = IO.file_to_instances(iob_file)
        self.logger.debug("Found %i instances for test"%len(self.test_instances))
        self.extractors = extractors
        self.output = {}
        self.error_matrix = None
        self.label_index = label_index
        return
    
    def eval(self):
        """
        Run the evaluator.
        
        Returns:
            TODO
        """
        extractor_results = {}
        for extractor in self.extractors:
            eng = extractor[1]
            extractor_name = extractor[0]
            input = [[token[0] for token in instance] for instance in self.test_instances if len(instance)>0]
            POS = False
            if(len(self.test_instances[0][0]) > 2):
                self.label_index = 2 # the last one is the label
                legacy_features = [[("z_POS",token[1]) for token in instance] for instance in self.test_instances if len(instance)>0]
                output = eng.extract(input,legacy_features)
                POS = True
            else:
                output = eng.extract(input)
            if(POS):
                to_evaluate = [[tuple([token["token"].decode("utf-8"),legacy_features[i][n][1],token["label"].decode("utf-8")]) for n,token in enumerate(instance)] for i,instance in enumerate(output)]
            else:
                to_evaluate = [[tuple([token["token"].decode("utf-8"),token["label"].decode("utf-8")]) for n,token in enumerate(instance)] for i,instance in enumerate(output)]
            results = self.evaluate(to_evaluate,self.test_instances,label_index = self.label_index)
            self.output[str(eng)] = self.write_result(to_evaluate,self.test_instances,self.label_index)
            eval_results = results[0]
            by_tag_results = results[1]
            eval_results["f-score"] = self.calc_fscore(eval_results)
            eval_results["precision"] = self.calc_precision(eval_results)
            eval_results["recall"] = self.calc_recall(eval_results)
            by_tag_results = self.calc_stats_by_tag(by_tag_results)
            extractor_results[extractor_name] = results
        return extractor_results
    
    @staticmethod
    def write_result(l_tagged_instances,l_test_instances,label_index=1):
        """
        
        """
        temp = [[(l_test_instances[n][i][0],l_test_instances[n][i][label_index],l_tagged_instances[n][i][label_index]) for i,token in enumerate(instance)] for n,instance in enumerate(l_test_instances)]
        return temp
    
    @staticmethod
    def print_stats(d_results):
        """
        Pretty print of the evaluation stats.
        """
        for item in d_results:
            print "%s\n%s\n%s"%("="*len(item),item,"="*len(item))
            print "%10s\t%10s\t%10s\t%5s\t%5s\t%5s\t%5s"%("f-score","precision","recall","tp","fp","tn","fn")
            print "%10f\t%10f\t%10f\t%5i\t%5i\t%5i\t%5i\n"%(d_results[item]["f-sc"],d_results[item]["prec"],d_results[item]["rec"],d_results[item]["true_pos"],d_results[item]["false_pos"],d_results[item]["true_neg"],d_results[item]["false_neg"],)
        return
    
    @staticmethod
    def read_instances(directories):
        result = []
        for d in directories:
            result += IO.read_iob_files(d)
        return result
    
    @staticmethod
    def evaluate(l_tagged_instances,l_test_instances,negative_BIO_tag = u'O',label_index=-1):
        """
        Evaluates a list of tagged instances against a list of test instances (gold standard):
        
        >>> tagged = [[('cf.','O'),('Hom','O'),('Il.','B-REFAUWORK'),('1.1','I-REFAUWORK'),(';','I-REFAUWORK')]] 
        >>> test = [[('cf.','O'),('Hom','B-REFAUWORK'),('Il.','I-REFAUWORK'),('1.1','B-REFSCOPE'),(';','O')]] 
        >>> res = SimpleEvaluator.evaluate(tagged,test) 
        >>> print res[0] 
        {'false_pos': 3, 'true_pos': 1, 'true_neg': 1, 'false_neg': 1}
        
        
        And with tokens having POS information
        >>> tagged = [[('cf.','N/A','O'),('Hom','N/A','O'),('Il.','N/A','B-REFAUWORK'),('1.1','N/A','I-REFAUWORK'),(';','N/A','I-REFAUWORK')]] 
        >>> test = [[('cf.','N/A','O'),('Hom','N/A','B-REFAUWORK'),('Il.','N/A','I-REFAUWORK'),('1.1','N/A','B-REFSCOPE'),(';','N/A','O')]]
        >>> print SimpleEvaluator.evaluate(tagged,test,label_index=2)[0] 
        {'false_pos': 3, 'true_pos': 1, 'true_neg': 1, 'false_neg': 1}
        
        Args:
            l_tagged_instances:
                A list of instances. Each instance is a list of tokens, the tokens being tuples.
                Each tuple has the token (i=0) and the assigned label (i=1).
                
            l_test_instances:
            
            pos_index:
                An integer: when set to -1 indicates that there is no POS "column" in the data. Otherwise provides the tuple index
                of the POS tag.
            
        Returns:
            A dictionary: 
            {
                "true_pos": <int>
                ,"false_pos": <int>
                ,"true_neg": <int>
                ,"false_neg": <int>
            }
        """
        # TODO: check same lenght and identity of tokens
        
        import logging
        l_logger = logging.getLogger('CREX.EVAL')
        
        fp = tp = fn = tn = token_counter = 0
        errors_by_tag = {}
        
        labels = ['O','B-AAUTHOR','I-AAUTHOR','B-AWORK','I-AWORK','B-REFAUWORK','I-REFAUWORK','B-REFSCOPE','I-REFSCOPE']
        import numpy
        error_matrix = numpy.zeros((len(labels),len(labels)),dtype=numpy.int)
        error_details = {}
        
        for n,inst in enumerate(l_tagged_instances):
            tag_inst = l_tagged_instances[n]
            gold_inst = l_test_instances[n]
            token_counter += len(tag_inst)
            for n,tok in enumerate(tag_inst):
                p_fp = p_tp = p_fn = p_tn = 0
                gold_token = gold_inst[n][0]
                tagged_token = tok[0]
                l_logger.debug("Gold token: %s"%gold_token)
                l_logger.debug("Tagged token: %s"%tagged_token)
                
                if(label_index != -1):
                    gold_label = gold_inst[n][label_index]
                    tagged_label = tok[label_index]
                else:
                    gold_label = gold_inst[n][1]
                    tagged_label = tok[1]
                l_logger.debug("Gold label: %s"%gold_label)
                l_logger.debug("Tagged label: %s"%tagged_label)
                    
                
                if(not errors_by_tag.has_key(gold_label)):
                    errors_by_tag[gold_label] = {"true_pos": 0
                            ,"false_pos": 0
                            ,"true_neg": 0
                            ,"false_neg": 0
                            }
                            
                error_matrix[labels.index(gold_label)][labels.index(tagged_label)] += 1
                error = "%s => %s"%(gold_label, tagged_label)
                if(gold_label != tagged_label):
                    if(error_details.has_key(error)):
                        error_details[error].append(gold_token)
                    else:
                        error_details[error] = []
                        error_details[error].append(gold_token)
                
                if(gold_label != negative_BIO_tag):
                    l_logger.debug("Label \"%s\" for token \"%s\" is not negative"%(gold_label,gold_token))
                    if(tagged_label == gold_label):
                        p_tp += 1
                        errors_by_tag[gold_label]["true_pos"] += 1
                        l_logger.info("[%s] \"%s\"=> tagged: %s / gold: %s"%("TP",tagged_token, tagged_label, gold_label))
                    elif(tagged_label != gold_label):
                        if(tagged_label == negative_BIO_tag):
                            p_fn += 1
                            errors_by_tag[gold_label]["false_neg"] += 1
                            l_logger.info("[%s] \"%s\"=> tagged: %s / gold: %s"%("FN",tagged_token, tagged_label, gold_label))
                        else:
                            p_fp += 1
                            errors_by_tag[gold_label]["false_pos"] += p_fp
                            l_logger.info("[%s] \"%s\"=> tagged: %s / gold: %s"%("FP",tagged_token, tagged_label, gold_label))
                elif(gold_label == negative_BIO_tag):
                    l_logger.debug("Label \"%s\" for token \"%s\" is negative"%(gold_label,gold_token)) 
                    if(tagged_label == gold_label):
                        p_tn += 1
                        errors_by_tag[gold_label]["true_pos"] += 1
                        l_logger.info("[%s] \"%s\"=> tagged: %s / gold: %s"%("TN",tagged_token, tagged_label, gold_label))
                    elif(tagged_label != gold_label):
                        if(tagged_label != negative_BIO_tag):
                            p_fp += 1
                            errors_by_tag[gold_label]["false_pos"] += 1
                            l_logger.info("[%s] \"%s\"=> tagged: %s / gold: %s"%("FP",tagged_token, tagged_label, gold_label))
                fp += p_fp
                tp += p_tp
                fn += p_fn
                tn += p_tn
        assert (tp+fp+tn+fn) == token_counter
        l_logger.debug("asserted %i (tp +fp + tn + fn) == %i (token counter)"%(tp+fp+tn+fn,token_counter))
        result = {"true_pos": tp
                ,"false_pos": fp
                ,"true_neg": tn
                ,"false_neg": fn
                },errors_by_tag
        global_sum = {"true_pos": 0
                      ,"false_pos": 0
                      ,"true_neg": 0
                      ,"false_neg": 0}
        for tag in result[1].keys():
            for value in result[1][tag]:
                global_sum[value]+= result[1][tag][value]
        assert (global_sum["true_pos"] + global_sum["false_pos"] + global_sum["false_neg"]) == token_counter
        l_logger.debug("asserted %i (tp +fp + fn) == %i (token counter)"%(tp+fp+tn+fn,token_counter))
        #SimpleEvaluator.render_error_matrix(error_matrix,labels)
        #print pprint.pprint(error_details)
        return result
    
    @staticmethod
    def render_error_matrix(matrix, labels):
        """
        TODO:
        
        Prints the error matrix
        
        """
        print '                        %11s'%" ".join(labels)
        for row_label, row in zip(labels, matrix):
            print '%11s [%s]' % (row_label, ' '.join('%09s' % i for i in row))
        return
    
    @staticmethod
    def calc_stats_by_tag(d_by_tag_errors):
        for tag in d_by_tag_errors:
            d_by_tag_errors[tag]["prec"] = SimpleEvaluator.calc_precision(d_by_tag_errors[tag])
            d_by_tag_errors[tag]["rec"] = SimpleEvaluator.calc_recall(d_by_tag_errors[tag])
            d_by_tag_errors[tag]["f-sc"] = SimpleEvaluator.calc_fscore(d_by_tag_errors[tag])
        return d_by_tag_errors
    
    @staticmethod
    def calc_stats_by_entity(d_by_tag_errors):
        """
        Aggregates results by entity (B-X and I-X are aggregated together.)
        
        Args:
            d_by_tag_errors:
                a dictionary containing error details by tag
                
        Example:
            >>> import core #doctest: +SKIP
            >>> from core import citation_extractor #doctest: +SKIP
            >>> from eval import SimpleEvaluator #doctest: +SKIP
            >>> import base_settings, settings #doctest: +SKIP
            >>> extractor_1 = citation_extractor(base_settings) #doctest: +SKIP
            >>> se = SimpleEvaluator([extractor_1,],["/Users/56k/phd/code/APh/experiments/C2/",]) #doctest: +SKIP
            >>> res = se.eval() #doctest: +SKIP
            >>> by_entity = se.calc_stats_by_entity(res[str(extractor_1)][1]) #doctest: +SKIP
            
            
        """
        overall_errors = d_by_tag_errors
        stats_by_entity = {}
        for tag in d_by_tag_errors:
                """
                logger.debug("(%s) True Positives (tp): %i"%(tag,overall_errors[tag]['true_pos']))
                logger.debug("(%s) False Positives (fp): %i"%(tag,overall_errors[tag]['false_pos']))
                logger.debug("(%s) False Negatives (fn): %i"%(tag,overall_errors[tag]['false_neg']))
                logger.debug("(%s) Total labels in test set: %i"%(tag,test_label_counts[tag]))
                logger.debug("(%s) precision: %f"%(tag,details[tag]["prec"]))
                logger.debug("(%s) recall: %f"%(tag,details[tag]["rec"]))
                logger.debug("(%s) F-score: %f"%(tag,details[tag]["f-sc"]))
                logger.debug("************")
                """
                if(tag != "O"):
                    aggreg_tag = tag.replace("B-","").replace("I-","")
                    if(not stats_by_entity.has_key(aggreg_tag)):
                        stats_by_entity[aggreg_tag] = {
                            "true_pos":0,
                            "true_neg":0,
                            "false_pos":0,
                            "false_neg":0,
                        }   
                    stats_by_entity[aggreg_tag]['false_pos'] += overall_errors[tag]['false_pos']
                    stats_by_entity[aggreg_tag]['true_pos'] += overall_errors[tag]['true_pos']
                    stats_by_entity[aggreg_tag]['true_neg'] += overall_errors[tag]['true_neg']
                    stats_by_entity[aggreg_tag]['false_neg'] += overall_errors[tag]['false_neg']
        for aggreg_tag in stats_by_entity:
                stats_by_entity[aggreg_tag]['prec'] = SimpleEvaluator.calc_precision(stats_by_entity[aggreg_tag])
                stats_by_entity[aggreg_tag]['rec'] = SimpleEvaluator.calc_recall(stats_by_entity[aggreg_tag])
                stats_by_entity[aggreg_tag]['f-sc'] = SimpleEvaluator.calc_fscore(stats_by_entity[aggreg_tag])
        return stats_by_entity
    
    @staticmethod       
    def calc_precision(d_errors):
        """
        Calculates the precision given the input error dictionary.
        """
        if(d_errors["true_pos"] + d_errors["false_pos"] == 0):
            return 0
        else:
            return d_errors["true_pos"] / float(d_errors["true_pos"] + d_errors["false_pos"])
    
    @staticmethod
    def calc_recall(d_errors):
        """
        Calculates the recall given the input error dictionary.
        """
        if(d_errors["true_pos"] + d_errors["false_neg"] == 0):
            return 0
        else:
            return d_errors["true_pos"] / float(d_errors["true_pos"] + d_errors["false_neg"])
    
    @staticmethod
    def calc_accuracy(d_errors):
        """
        Calculates the accuracy given the input error dictionary.
        """
        acc = (d_errors["true_pos"] + d_errors["true_neg"]) / float(d_errors["true_pos"] + d_errors["false_pos"] + d_errors["true_neg"] + d_errors["false_neg"])
        return acc
    
    @staticmethod
    def calc_fscore(d_errors):
        """
        Calculates the accuracy given the input error dictionary.
        """
        prec = SimpleEvaluator.calc_precision(d_errors)
        rec = SimpleEvaluator.calc_recall(d_errors)
        if(prec == 0 and rec == 0):
            return 0
        else:
            return 2*(float(prec * rec) / float(prec + rec))
class CrossEvaluator(SimpleEvaluator):
    """
    >>> import settings #doctest: +SKIP
    >>> import pprint #doctest: +SKIP
    >>> base_settings.DEBUG = False #doctest: +SKIP
    >>> extractor_1 = settings #doctest: +SKIP
    >>> test_files = ["/Users/56k/phd/code/APh/experiments/eff_cand_1_a/","/Users/56k/phd/code/APh/experiments/C1/","/Users/56k/phd/code/APh/experiments/C2/",] #doctest: +SKIP
    >>> ce = CrossEvaluator([extractor_1,],test_files,culling_size=100,fold_number=10,evaluation_dir="/Users/56k/Downloads/eval_temp/") #doctest: +SKIP
    >>> result = ce.run() #doctest: +SKIP
    >>> pprint.pprint(result) #doctest: +SKIP
    """

    def __init__(self,extractors,iob_test_file,culling_size=None,fold_number=10,evaluation_dir="./",label_index=-1):
        super(CrossEvaluator, self).__init__(extractors,iob_test_file,label_index=label_index)
        self.culling_size = culling_size
        self.fold_number = fold_number
        self.evaluation_dir = evaluation_dir
        import logging
        self.logger = init_logger(verbose=True,log_name='CREX.CROSSEVAL')
        if(self.culling_size is not None):
            self.logger.info("Culling set at %i"%self.culling_size)
            import random
            random.shuffle(self.test_instances)
            self.culled_instances = self.test_instances[:self.culling_size]
        else:
            self.logger.info("Culling not set.")
        self.logger.info("Evaluation type: %i-fold cross evaluations"%self.fold_number)
        self.logger.info("Training/Test set contains %i instances."%len(self.test_instances))
        self.create_datasets()
    
    def create_datasets(self):
        """
        TODO
        """
        
        from miguno.partitioner import *
        from miguno.crossvalidationdataconstructor import *
        from citation_extractor.Utils import IO
        positive_labels = ["B-REFSCOPE","I-REFSCOPE","B-AAUTHOR","I-AAUTHOR","B-REFAUWORK","I-REFAUWORK","B-AWORK","I-AWORK"]
        if(self.culling_size is not None):
            positives_negatives = [(n,IO.instance_contains_label(inst,positive_labels)) for n,inst in enumerate(self.culled_instances)]
            positives = [self.culled_instances[i[0]] for i in positives_negatives if i[1] is True]
            negatives = [self.culled_instances[i[0]] for i in positives_negatives if i[1] is False]
        else:
            positives_negatives = [(n,IO.instance_contains_label(inst,positive_labels)) for n,inst in enumerate(self.test_instances)]
            positives = [self.test_instances[i[0]] for i in positives_negatives if i[1] is True]
            negatives = [self.test_instances[i[0]] for i in positives_negatives if i[1] is False]
        self.logger.info("%i Positive instances"%len(positives))
        self.logger.info("%i Negative instances"%len(negatives))
        self.logger.info("%i Total instances"%(len(positives)+len(negatives)))
        self.dataSets_iterator = CrossValidationDataConstructor(positives, negatives, numPartitions=self.fold_number, randomize=False).getDataSets()
        pass
    
    def run(self):
        """
        TODO        
        """
        iterations = []
        results = {}
        results_by_entity = {}
        # first lets' create test and train set for each iteration
        for x,iter in enumerate(self.dataSets_iterator):
            self.logger.info("Iteration %i"%(x+1))
            train_set=[]
            test_set=[]
            for y,set in enumerate(iter):
                for n,group in enumerate(set):
                    if(y==0):
                        train_set+=group
                    else:
                        test_set+=group
            iterations.append((train_set,test_set))
        
        # let's go through all the iterations
        for i,iter in enumerate(iterations):
            results["iter-%i"%(i+1)] = {}
            results_by_entity["iter-%i"%(i+1)] = {}
            train_file="%sfold_%i.train"%(self.evaluation_dir,i+1)
            test_file="%sfold_%i.test"%(self.evaluation_dir,i+1)
            IO.write_iob_file(iter[0],train_file)
            IO.write_iob_file(iter[1],test_file)
            # the following line is a bit of a workaround
            # to avoid recomputing the features when training
            # each new classifier, I take them from the file created
            # to train the CRF model (which should always be the first extractor
            # to be evaluated).
            filename = "%sfold_%i.train.train"%(self.extractors[0][1].TEMP_DIR,(i+1))
            f=codecs.open(filename,'r','utf-8')
            data = f.read()
            f.close()
            feature_sets=[[[token.split('\t')[:len(token.split('\t'))-1],token.split('\t')[len(token.split('\t'))-1:]] for token in instance.split('\n')] for instance in data.split('\n\n')]
            order = FeatureExtractor().get_feature_order()
            labelled_feature_sets=[]
            for instance in feature_sets:
                for token in instance:
                    temp = [{order[n]:feature for n,feature in enumerate(token[0])},token[1][0]]
                    labelled_feature_sets.append(temp)
            self.logger.info("read %i labelled instances"%len(feature_sets))
            for n,extractor in enumerate(self.extractors):
                    extractor_settings = extractor[1]
                    extractor_name = extractor[0]
                    results["iter-%i"%(i+1)][extractor_name] = {}
                    self.logger.info("Running iteration #%i with extractor %s"%(i+1,extractor_name))
                    self.logger.info(train_file)
                    self.logger.info(test_file)
                    self.logger.info(extractor_settings)
                    extractor_settings.DATA_FILE = train_file
                    if(extractor_settings.CLASSIFIER is not None):
                        extractor = citation_extractor(extractor_settings, extractor_settings.CLASSIFIER,labelled_feature_sets)
                    else:
                        extractor = citation_extractor(extractor_settings)
                    self.logger.info(extractor.classifier)
                    se = SimpleEvaluator([(extractor_name, extractor),],iob_file=test_file)
                    results["iter-%i"%(i+1)][extractor_name] = se.eval()[extractor_name][0]
                    results_by_entity["iter-%i"%(i+1)][extractor_name] = SimpleEvaluator.calc_stats_by_entity(se.eval()[extractor_name][1])
                    #self.logger.info(results_by_entity["iter-%i"%(i+1)][extractor_name])
        return results,results_by_entity    
# NB:
# `evaluate_ned` is a legacy function
# it needs to be refactored and tested to make sure it works with `read_ann_file_new` 
# (now it works just with `read_ann_file`)
# also, it needs to take into account NIL entities (TODO)
def evaluate_ned(docid, gold_disambiguations, target_disambiguations, exclude_relations=False):
    """
    Evaluates the accuracy of the disambiguation of named entities and relations (i.e. references). 
    """
    gold_disambiguations =  {id:(label,urn) for id,label,urn in gold_disambiguations if len(gold_disambiguations)>0}
    target_disambiguations = {id:(label,urn) for id,label,urn in target_disambiguations if len(target_disambiguations)>0}
    """
    What it does?
    - a disambiguation that is in the target_set but not in the gold_set is a FP
    - a disambiguation that is in the gold_set but not in the target_set is a FN
    - when the disambiguation is identical is a TP
    - when the disambiguation differ is a FP
    """
    result = {
    "true_pos":0
    ,"false_pos":0
    ,"false_neg":0
    ,"true_neg":0
    }
    errors = {
    "true_pos":[]
    ,"false_pos":[]
    ,"false_neg":[]
    ,"true_neg":[]
    }
    for id in gold_disambiguations:
        if(exclude_relations and id.startswith("R")):
            pass
        else:
            try:
                gold_urn = gold_disambiguations[id][1]
                target_urn = target_disambiguations[id][1]
                if(gold_urn == target_urn):
                    errors["true_pos"].append((gold_disambiguations[id][0],gold_urn,target_urn))
                    result["true_pos"]+=1
                else:
                    errors["false_pos"].append((gold_disambiguations[id][0],gold_urn,target_urn))
                    result["false_pos"]+=1
            except Exception, e:
                result["false_neg"]+=1
                errors["false_neg"].append((gold_disambiguations[id][0],None))
    for id in target_disambiguations:
        if(exclude_relations and id.startswith("R")):
            pass
        else:
            try:
                pass
            except Exception, e:
                result["false_pos"]+=1
                errors["false_pos"].append((gold_disambiguations[id][0],None,target_urn))
    return result,errors
def analyse_errors(errors_dict):
    """
    The main goal of this function is to check how many of the FP errors are due to the wrong work
    being identified, and how many instead are due to an error in the parsing of the reference scope.

    :param errors_dict: the dictionary with errors returned by `evaluate_ned`.
    """
    correct_scope = 0
    passages = 0
    authors = 0
    works = 0
    for label,gold_urn,target_urn in errors_dict["false_pos"]:
        try:
            gold_urn = CTS_URN(gold_urn)
            target_urn = CTS_URN(target_urn)
            if(gold_urn.passage_component is not None):
                passages+=1
                if(gold_urn.passage_component == target_urn.passage_component):
                    correct_scope+=1
                else:
                    pass
            elif(gold_urn.work is not None):
                works+=1
            elif(gold_urn.textgroup is not None):
                authors+=1
        except Exception, e:
            pass
    print >> sys.stdout, "The {3} FPs contain: {0} aauthor entities; {1} awork entities; {2} scope relations".format(authors,works,passages,len(errors_dict["false_pos"]))
    print >> sys.stdout, "{0}% of the FPs has correct reference scope (n={1})".format((correct_scope*100)/passages,correct_scope)
if __name__ == "__main__":
    #Usage example: python eval.py aph_data_100_positive/ out/
    #main()
    import doctest
    doctest.testmod(verbose=True)