# -*- coding: utf-8 -*-
# author: Matteo Romanello, matteo.romanello@gmail.com

"""

Module containing classes and functions to perform the evaluation of the various steps of the pipeline (NER, RelEx, NED).

%load_ext autoreload
%autoreload 2

import logging
import tabulate
from citation_extractor.Utils.IO import init_logger
init_logger(loglevel=logging.DEBUG)
import pickle
import codecs
import pkg_resources
import pandas as pd
from citation_extractor.eval import evaluate_ned

with codecs.open(pkg_resources.resource_filename("citation_extractor", "data/pickles/test_gold_dataframe.pkl"),"rb") as pickle_file:
    testset_gold_df = pd.read_pickle(pickle_file)

with codecs.open(pkg_resources.resource_filename("citation_extractor", "data/pickles/test_target_dataframe_cm1.pkl"),"rb") as pickle_file:
    testset_target_df = pd.read_pickle(pickle_file)

ann_dir = "/Users/rromanello/Documents/crex/citation_extractor/citation_extractor/data/aph_corpus/testset/ann/"

scores, error_types, errors = evaluate_ned(testset_gold_df, ann_dir, testset_target_df, strict=True)

"""
from __future__ import division
import pdb # TODO: remove from production code
import sys,logging,re
import os
import glob
import math
from pyCTS import BadCtsUrnSyntax
from citation_extractor.core import *
from citation_extractor.crfpp_wrap import CRF_classifier
from citation_extractor.Utils import IO
from citation_extractor.Utils.IO import read_ann_file, read_ann_file_new, init_logger, NIL_ENTITY
# TODO: in the long run, remove `SimpleEvaluator` and `CrossEvaluator`
#   and generally the `miguno` library as a dependency (use `sklearn` instead) 
#from miguno.partitioner import *
#from miguno.crossvalidationdataconstructor import *
import pprint

global logger
logger = logging.getLogger(__name__)

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

class CrossEvaluator(SimpleEvaluator): # TODO: remove
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

def evaluate_ned(goldset_data, gold_directory, target_data, strict=False):
    """
    Evaluate the Named Entity Disambigutation taking in input the goldset data, the
    goldset directory and a target directory contaning files in the brat stand-off annotation format.

    The F1 score is computed over the macro-averaged precision and recall.
    self.

    :param goldset_data: a `pandas.DataFrame` with the goldset data read via `citation_extractor.Utils.IO.load_brat_data`

    :param gold_directory: the path to the gold set

    :param target: a `pandas.DataFrame` with the target data read via `citation_extractor.Utils.IO.load_brat_data`

    :param strict: whether to consider consecutive references to the same ancient work only once (i.e. `scope`
        relations with identical arg1).

    :return: a tuple where [0] is a dictionary with keys "precision", "recall", "fscore";
            [1] is a list of dictionaries (keys "true_pos", "true_neg", "false_pos" and "false_neg"), one for each document;
            [2] is a dictionary containing the actual URNs (gold and predicted) grouped by error types
            or None if the evaluation is aborted.

    """

    # variables to store aggregated results and errors
    disambig_results = []
    disambig_errors = {"true_pos":[], "true_neg":[], "false_pos":[], "false_neg":[]}
    aggregated_results = {"true_pos":0, "true_neg":0, "false_pos":0, "false_neg":0}
    results_by_entity_type = {}
    scores = {}

    # check that number/names of .ann files is the same
    doc_ids_gold = list(set(goldset_data["doc_id"]))
    docs_ids_target = list(set(target_data["doc_id"]))

    try:
        assert sorted(doc_ids_gold)==sorted(docs_ids_target)
    except AssertionError as e:
        logger.error("Evaluation aborted: the script expects identical filenames in gold and target directory.")
        return (None, None, None)

    logger.info("Evaluating NED: there are %i documents" % len(doc_ids_gold))

    for doc_id in doc_ids_gold:

        # create a dictionary like {"T1":"urn:cts:greekLit:tlg0012", }
        gold_disambiguations = {id.split('-')[2]: row["urn_clean"]
                                    for id, row in goldset_data[goldset_data["doc_id"]==doc_id].iterrows()}

        # pass on all relations data
        gold_entities, gold_relations = read_ann_file_new("%s.txt" % doc_id, os.path.join(gold_directory, ""))[:2]

        # create a dictionary like {"T1":"urn:cts:greekLit:tlg0012", }
        target_disambiguations = {id.split('-')[2]: row["urn_clean"]
                                    for id, row in target_data[target_data["doc_id"]==doc_id].iterrows()}

        # process each invidual file
        file_result, file_errors, result_by_entity_type = _evaluate_ned_file(doc_id
                                                    , gold_disambiguations
                                                    , gold_entities
                                                    , gold_relations
                                                    , target_disambiguations
                                                    , strict)


        # add error details
        for error_type in file_errors:
            disambig_errors[error_type]+=file_errors[error_type]

        for entity_type in result_by_entity_type:

            if not entity_type in results_by_entity_type:
                results_by_entity_type[entity_type] = {}

            for error_type in result_by_entity_type[entity_type]:

                if not error_type in results_by_entity_type[entity_type]:
                    results_by_entity_type[entity_type] = {"true":0, "false":0}

                results_by_entity_type[entity_type][error_type] += result_by_entity_type[entity_type][error_type]


        # NB: if the file contains only NIL entities we exclude it from the counts
        # used to computed the macro-averaged precision and recall
        NIL_entities = [urn for urn in gold_disambiguations.values() if urn == NIL_ENTITY]
        non_NIL_entities = [urn for urn in gold_disambiguations.values() if urn != NIL_ENTITY]

        if len(non_NIL_entities)>0:
            disambig_results.append(file_result)
        elif len(non_NIL_entities)==0:
            logger.debug("%s contains only NIL entities (or is empty): not considered when computing macro-averaged measures" % doc_id)

        # still, we include it in the counts used to compute the global accuracy
        for key in file_result:
            aggregated_results[key]+=file_result[key]

    precisions = [SimpleEvaluator.calc_precision(r) for r in disambig_results]
    recalls = [SimpleEvaluator.calc_recall(r) for r in disambig_results]
    assert len(precisions)==len(recalls)

    scores = {
        "precision" : sum(precisions)/len(precisions)
        , "recall" : sum(recalls)/len(recalls)
    }
    prec, rec = scores["precision"], scores["recall"]
    scores["fscore"] = 0.0 if prec == 0.0 and rec == 0.0 else 2*(float(prec * rec) / float(prec + rec))
    scores["accuracy"] = (aggregated_results["true_pos"] + aggregated_results["true_neg"]) \
                        / (aggregated_results["true_pos"] + aggregated_results["true_neg"] \
                            + aggregated_results["false_neg"] + aggregated_results["false_pos"])
    logger.info("Computing accuracy: %i (tp) + %i (tn) / %i (tp) + %i (tn) + %i (fn) + %i (fp" % (
                                                                aggregated_results["true_pos"]
                                                                , aggregated_results["true_neg"]
                                                                , aggregated_results["true_pos"]
                                                                , aggregated_results["true_neg"]
                                                                , aggregated_results["false_neg"]
                                                                , aggregated_results["false_pos"]
                                                                ))
    assert sum([results_by_entity_type[ent_type][err_type]
                for ent_type in results_by_entity_type
                for err_type in results_by_entity_type[ent_type]]) == sum(aggregated_results.values())

    logger.info("Precision and recall averaged over %i documents (documents with NIL-entities only are excluded)" % len(precisions))
    print("Precision %.2f%%" % (scores["precision"]*100))
    print("Recall %.2f%%" % (scores["recall"]*100))
    print("Fscore %.2f%%" % (scores["fscore"]*100))
    print("Accuracy %.2f%%" % (scores["accuracy"]*100))

    accuracy_by_type = {}
    print("\nAccuracy by type:")
    for entity_type in sorted(results_by_entity_type.keys()):
        true_matches = results_by_entity_type[entity_type]["true"]
        false_matches = results_by_entity_type[entity_type]["false"]
        total_matches = true_matches + false_matches
        accuracy_by_type[entity_type] = true_matches / total_matches
        print("%s: %.2f%%" % (entity_type, (accuracy_by_type[entity_type] * 100)))

    return (scores, accuracy_by_type, disambig_results, disambig_errors)

def _evaluate_ned_file(docid, gold_disambiguations, gold_entities, gold_relations, target_disambiguations, strict=False):
    """
    Evaluates NED of a single file.

    """
    # TODO expect data in this format
    unique_reference_urns = set() # for multiple relations having as arg1 entity X, count X only once

    result = {"true_pos":0, "false_pos":0 ,"false_neg":0 ,"true_neg":0}
    result_by_entity_type = {}
    errors = {"true_pos":[], "false_pos":[], "false_neg":[], "true_neg":[]}

    try:
        assert len(gold_disambiguations)>0 and len(target_disambiguations)>0
    except AssertionError as e:
        logger.info("No disambiguations to evaluate in file %s" % docid)
        return None, errors

    for disambiguation_id in gold_disambiguations:

        is_relation_disambiguation = True if disambiguation_id.startswith('R') else False

        try:
            gold_disambiguation = gold_disambiguations[disambiguation_id]
            gold_urn = CTS_URN(gold_disambiguation.strip()).get_urn_without_passage()
        except BadCtsUrnSyntax as e:
            logger.error("Skipping disambiguation %s-%s: gold URN malformed (\"%s\")" % (docid, disambiguation_id, gold_disambiguation))
            return result, errors

        try:
            target_disambiguation = target_disambiguations[disambiguation_id]
            target_urn = CTS_URN(target_disambiguation.strip()).get_urn_without_passage()
        except BadCtsUrnSyntax as e:
            logger.error("Skipping disambiguation %s-%s: target URN malformed (\"%s\")" % (docid, disambiguation_id, target_disambiguation))
            continue
        except AttributeError as e:
            logger.error("Disambiguation %s-%s: target URN is None (\"%s\")" % (docid, disambiguation_id, target_disambiguation))
            target_urn = None
        except KeyError as e:
            logger.debug("[%s] %s not contained in target: assuming a NIL entity" % (docid, disambiguation_id))
            target_urn = NIL_ENTITY
            continue

        arg1_entity_id = None

        if is_relation_disambiguation:
            logger.debug("[%s] unique_reference_urns=%s" % (docid, unique_reference_urns))
            arg1_entity_id = gold_relations[disambiguation_id]['arguments'][0]

            if strict:
                if "%s-R" % arg1_entity_id in unique_reference_urns:
                    logger.debug("%s was already considered; skipping this one" % "%s-R" % arg1_entity_id)
                    continue
                else:
                    unique_reference_urns.add("%s-R" % arg1_entity_id)

        # classify the error by type
        if gold_urn == NIL_ENTITY:
            error_type = "true_neg" if gold_urn == target_urn else "false_pos"
        else:
            # gold_urn is not a NIL entity
            if target_urn == NIL_ENTITY:
                error_type = "false_neg"
            # neither gold_urn nor target_urn are NIL entities
            else:
                if gold_urn == target_urn:
                    error_type = "true_pos"
                else:
                    error_type = "false_pos"

        if gold_urn != NIL_ENTITY:
            entity_type = gold_entities[disambiguation_id]["entity_type"] if not is_relation_disambiguation else "scope-%s" % gold_entities[arg1_entity_id]["entity_type"]
        else:
            entity_type = "NIL"

        error_by_entity_type = "true" if error_type == "true_pos" or error_type == "true_neg" else "false"

        if not entity_type in result_by_entity_type:
            result_by_entity_type[entity_type] = {"true":0, "false":0}

        result_by_entity_type[entity_type][error_by_entity_type]+=1

        result[error_type]+=1
        errors[error_type].append((docid, disambiguation_id, gold_urn, target_urn))
        logger.debug("[%s-%s] Comparing %s with %s => %s" % (docid, disambiguation_id, gold_urn, target_urn, error_type))

    logger.debug("Evaluated file %s: %s" % (docid, result))

    return result, errors, result_by_entity_type

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
