# -*- coding: utf-8 -*-
"""
.. module:: eval
   :platform: Unix
   :synopsis: A module to test and evaluate the citation extractor.

.. moduleauthor:: Matteo Romanello <matteo.romanello@gmail.com>


"""
import sys,logging,re
import os
import glob
from citation_extractor.core import *
from citation_extractor.crfpp_wrap import CRF_classifier
from Utils import IO
from miguno.partitioner import *
from miguno.crossvalidationdataconstructor import *
import pprint

global logger
EVAL_PATH="/home/ngs0554/eval/"
DATA_PATH="/home/ngs0554/crex_data/"

def init_logger(verbose=False, log_file=None, log_name='CREX.EVAL'):
	"""
    Initialise a logger.
	
    Parameters
    ----------
    verbose : bool, optional
        Verbosity of the logger. By default is turned off (False).
    log_file : str, optional
        The name of the file where all log messages are to be redirected.
		When no file name is provided the log will be printed to the standard output.
		
    Returns
    -------
    l_logger : logging.Logger
      The initialised array.
	
    Examples
    --------
    >>> l = init_logger() #doctest: +SKIP
    >>> type(l) #doctest: +SKIP
    <class 'logging.Logger'>
	
    """
	import logging
	l_logger = logging.getLogger(log_name)
	if(verbose):
		l_logger.setLevel(logging.DEBUG)
	else:
		l_logger.setLevel(logging.INFO)
	if(log_file is None):
		ch = logging.StreamHandler()
		ch.setLevel(logging.DEBUG)
		ch.setFormatter(logging.Formatter("%(asctime)s -- %(name)s -- %(levelname)s -- %(message)s"))
		l_logger.addHandler(ch)
		l_logger.info("Logger initialised.")
	else:
		ch = logging.FileHandler(log_file)
		ch.setLevel(logging.DEBUG)
		ch.setFormatter(logging.Formatter("%(asctime)s -- %(name)s -- %(levelname)s -- %(message)s"))
		l_logger.addHandler(ch)
		l_logger.info("Logger initialised.")
	return l_logger

def main():
	global DATA_PATH,EVAL_PATH,logger
	logger = init_logger(verbose=False,log_file="eval.log")
	DATA_PATH = sys.argv[1]
	EVAL_PATH = sys.argv[2]
	eval("%s"%(DATA_PATH),10)

def run_example(data_dir):
	DATA_PATH=data_dir
	eval("%s/test.txt"%DATA_PATH,10)

def run_example():
	DATA_PATH="/home/ngs0554/crex_data"
	run_example(DATA_PATH)

class SimpleEvaluator(object):
	"""
	>>> import base_settings, settings #doctest: +SKIP
	>>> extractor_1 = citation_extractor(base_settings) #doctest: +SKIP
	>>> extractor_2 = citation_extractor(settings) #doctest: +SKIP
	>>> se = SimpleEvaluator([extractor_1,],iob_file="data/75-02637.iob") #doctest: +SKIP
	>>> se = SimpleEvaluator([extractor_1,],["/Users/56k/phd/code/APh/experiments/C2/",]) #doctest: +SKIP
	>>> print se.eval() #doctest: +SKIP
	
	"""
	def __init__(self,extractors,iob_directories=[],iob_file=None):
		"""
		Args:
			extractors:
				the list of canonical citation extractors to evaluate
			iob_test_file: 
				the file in IOB format to be used for testing and evaluating the extactors
		"""
		# read the test instances from a list of directories containing the test data
		
		if(iob_file is None):
			self.test_instances = self.read_instances(iob_directories)
		else:
			self.test_instances = IO.file_to_instances(iob_file)
		temp = [[tok[0] for tok in inst]for inst in self.test_instances]
		self.string_instances = [" ".join(n) for n in temp]
		self.extractors = extractors
		return
	
	def eval(self):
		"""
		Run the evaluator.
		
		Returns:
			TODO
		
		"""
		extractor_results = {}
		for eng in self.extractors:
			input = [eng.tokenize(inst) for inst in self.string_instances]
			#IO.write_iob_file(self.test_instances,"/Users/56k/Downloads/puzzled/test_curr.data")
			output = eng.extract(input)
			to_evaluate = [[tuple([t["token"].decode("utf-8"),t["label"].decode("utf-8")]) for t in i] for i in output]
			results = self.evaluate(to_evaluate,self.test_instances)
			#self.output[str(eng)] = self.write_result(to_evaluate,self.test_instances)
			eval_results = results[0]
			by_tag_results = results[1]
			eval_results["f-score"] = self.calc_fscore(eval_results)
			eval_results["precision"] = self.calc_precision(eval_results)
			eval_results["recall"] = self.calc_recall(eval_results)
			by_tag_results = self.calc_stats_by_tag(by_tag_results)
			extractor_results[str(eng)] = results
		return extractor_results
	
	@staticmethod
	def read_instances(directories):
		result = []
		for d in directories:
			result += IO.read_iob_files(d)
		return result
	
	@staticmethod
	def evaluate(l_tagged_instances,l_test_instances,negative_BIO_tag = u'O'):
		"""
		Evaluates a list of tagged instances against one of test instances (gold standard).
		>>> tagged = [[('ü','O'),('Hom','O'),('Il.','B-CREF')]] #doctest: +SKIP
		>>> test = [[('ü','O'),('Hom','B-CREF'),('Il.','I-CREF')]] #doctest: +SKIP
		>>> res = SimpleEvaluator.evaluate(tagged,test) #doctest: +SKIP
		>>> print res[0] #doctest: +SKIP
		{'false_pos': 2, 'true_pos': 0, 'true_neg': 1, 'false_neg': 0}
		>>> print SimpleEvaluator.calc_stats_by_tag(res[1]) #doctest: +SKIP
		
		Args:
			l_tagged_instances:
				A list of instances. Each instance is a list of tokens, the tokens being tuples.
				Each tuple has the token (i=0) and the assigned label (i=1).
				For example:
				>>> l_instances = [[('vd.','O'),('Hom','B-CREF')]] #doctest: +SKIP
				
			l_test_instances:
			
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
		
		fp = 0 # false positive counter
		tp = 0 # true positive counter
		fn = 0 # false negative counter
		tn = 0 # true negative counter
		token_counter = 0
		errors_by_tag = {}
		
		for n,inst in enumerate(l_tagged_instances):
			tag_inst = l_tagged_instances[n]
			gold_inst = l_test_instances[n]
			token_counter += len(tag_inst)
			for n,tok in enumerate(tag_inst):
				p_fp = 0 # false positive counter
				p_tp = 0 # true positive counter
				p_fn = 0 # false negative counter
				p_tn = 0 # true negative counter
				
				tag_tok = tok
				gold_tok = gold_inst[n]
				if(not errors_by_tag.has_key(gold_tok[1])):
					errors_by_tag[gold_tok[1]] = {"true_pos": 0
							,"false_pos": 0
							,"true_neg": 0
							,"false_neg": 0
							}
				if(not errors_by_tag.has_key(tag_tok[1])):
					errors_by_tag[tag_tok[1]] = {"true_pos": 0
							,"false_pos": 0
							,"true_neg": 0
							,"false_neg": 0
							}
				if(gold_tok[1] == negative_BIO_tag and (gold_tok[1] == tag_tok[1])):
					p_tn += 1 # increment the value of true negative counter
					#errors_by_tag[gold_tok[1]]["true_neg"] += p_tn
					errors_by_tag[gold_tok[1]]["true_pos"] += 1
					l_logger.debug("comparing \"%s\" (%s) <=> \"%s\" (%s) :: true negative"%(gold_tok[0],gold_tok[1],tag_tok[0],tag_tok[1]))
				elif(gold_tok[1] != negative_BIO_tag and (gold_tok[1] == tag_tok[1])):
					p_tp += 1 # increment the value of true positive counter
					errors_by_tag[gold_tok[1]]["true_pos"] += p_tp
					l_logger.debug("comparing \"%s\" (%s) <=> \"%s\" (%s) :: true positive"%(gold_tok[0],gold_tok[1],tag_tok[0],tag_tok[1]))
				elif(gold_tok[1] != negative_BIO_tag and (tag_tok[1] != gold_tok[1])):
					p_fn += 1 # increment the value of false negative counter
					errors_by_tag[gold_tok[1]]["false_neg"] += p_fn
					l_logger.debug("comparing \"%s\" (%s) <=> \"%s\" (%s) :: false negative"%(gold_tok[0],gold_tok[1],tag_tok[0],tag_tok[1]))
				elif(gold_tok[1] == negative_BIO_tag and (tag_tok[1] != gold_tok[1])):
					p_fp += 1 # increment the value of false positive counter
					errors_by_tag[tag_tok[1]]["false_pos"] += p_fp
					errors_by_tag[gold_tok[1]]["false_neg"] += 1
					l_logger.debug("comparing \"%s\" (%s) <=> \"%s\" (%s) :: false positive"%(gold_tok[0],gold_tok[1],tag_tok[0],tag_tok[1]))
				
				fp += p_fp
				tp += p_tp
				fn += p_fn
				tn += p_tn
				
			l_logger.debug(errors_by_tag)
		#print errors_by_tag
		assert tp+fp+tn+fn == token_counter
		print "%i == %i"%(tp+fp+tn+fn,token_counter)
		return {"true_pos": tp
				,"false_pos": fp
				,"true_neg": tn
				,"false_neg": fn
				},errors_by_tag
	
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
	>>> import settings
	>>> import pprint
	>>> base_settings.DEBUG = False
	>>> extractor_1 = settings
	>>> test_files = ["/Users/56k/phd/code/APh/experiments/eff_cand_1_a/","/Users/56k/phd/code/APh/experiments/C1/","/Users/56k/phd/code/APh/experiments/C2/",]
	>>> ce = CrossEvaluator([extractor_1,],test_files,culling_size=100,fold_number=10,evaluation_dir="/Users/56k/Downloads/eval_temp/") 
	>>> result = ce.run()
	>>> pprint.pprint(result)
	"""
	def __init__(self,extractors,iob_test_file,culling_size=None,fold_number=10,evaluation_dir="./"):
		super(CrossEvaluator, self).__init__(extractors,iob_test_file)
		self.culling_size = culling_size
		self.fold_number = fold_number
		self.evaluation_dir = evaluation_dir
		import logging
		self.logger = init_logger(verbose=False,log_name='CREX.CROSSEVAL')
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
		docstring for create_datasets
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
		self.dataSets_iterator = CrossValidationDataConstructor(positives, negatives, numPartitions=self.fold_number, randomize=True).getDataSets()
		pass
	
	def run(self):
		"""
		docstring for run
		
		TODO:
			for each iteration
				for each engine (extractor)
					write to file the train set
					write to file the test set
					evaluate
					append to 
						results[extractors[str(extractor_1)]][round-n][fscore]
						results[extractors[str(extractor_1)]][round-n][prec]
						results[extractors[str(extractor_1)]][round-n][...]
		
		"""
		iterations = []
		results = {}
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
			for n,extractor_settings in enumerate(self.extractors):
					self.logger.info("Running iteration #%i with extractor #%i"%(i+1,n+1))
					self.logger.info(extractor_settings)
					train_file="%sfold_%i.train"%(self.evaluation_dir,i+1)
					test_file="%sfold_%i.test"%(self.evaluation_dir,i+1)
					self.logger.info(train_file)
					self.logger.info(test_file)
					import codecs
					file = codecs.open(train_file,'w','utf-8')
					tmp = [[("%s\t%s"%(token[0],token[1]))for token in instance] for instance in iter[0]]
					tmp = ["\n".join(x) for x in tmp]
					to_write = "\n\n".join(tmp)
					file.write(to_write)
					file.close()
					file = codecs.open(test_file,'w','utf-8')
					tmp = [[("%s\t%s"%(token[0],token[1]))for token in instance] for instance in iter[1]]
					tmp = ["\n".join(x) for x in tmp]
					to_write = "\n\n".join(tmp)
					file.write(to_write)
					file.close()
					extractor_settings.DATA_FILE = train_file
					extractor = citation_extractor(extractor_settings)
					results[str(extractor)] = {}
					se = SimpleEvaluator([extractor,],iob_file=test_file)
					results[str(extractor)]["iter-%i"%(i+1)]= se.eval()
		return results						

	
		
		pass
if __name__ == "__main__":
	#Usage example: python eval.py aph_data_100_positive/ out/
	#main()
	import doctest
	doctest.testmod(verbose=True)