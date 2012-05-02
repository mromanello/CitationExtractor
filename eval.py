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
    >>> l = init_logger()
    >>> type(l)
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

def eval(fname,n_folds):
	"""
    Evaluate...
	
    Parameters
    ----------
    fname : str
        File name.
    n_folds : int
        Number of iterations
		
    Returns
    -------
    ...
	
    Examples
    --------
	>>> print "here we go"
	here we go
    """
	valid_res=[]
	fe = FeatureExtractor()
	instances=[]
	try:
		for infile in glob.glob( os.path.join(fname, '*.iob') ):
			try:
				temp = read_instances(fe.prepare_for_training(infile))
				instances+=temp
				logger.debug("read %i instances from file %s"%(len(temp),infile))
			except:
				logger.error("failed reading from %s"%infile)
		#neg_inst=[ins for ins in instances if not instance_contains_label(ins,'B-REFAUWORK')] # TODO check if it works
		#pos_inst=[ins for ins in instances if instance_contains_label(ins,'B-REFAUWORK')] # TODO check if it works
		mn = len(instances)/2
		pos_inst = instances[:mn]
		neg_inst = instances[mn:]
		logger.debug(instances)
		shuffle(neg_inst)
		max=0
		logger.debug('# of negative instances: %i'%len(neg_inst))
		logger.debug('# of positive instances: %i'%len(pos_inst))
		if(len(pos_inst)>len(neg_inst)):
			max=len(neg_inst)
		else:
			max=len(pos_inst)
		print("Total instances: %i (Test set plus training set)"%(len(neg_inst)+len(pos_inst)))
		iterations_num=n_folds
		c = CrossValidationDataConstructor(pos_inst, neg_inst, numPartitions=iterations_num, randomize=False)
		dataSets_iterator = c.getDataSets()
		iterations=[]
		tot_acc=0.0
		tot_prec=0.0
		tot_fscore=0.0
		tot_rec=0.0
		html=""
		for x,iter in enumerate(dataSets_iterator):
			print "Iteration %i"%(x+1)
			train_set=[]
			test_set=[]
			for y,set in enumerate(iter):
				for n,group in enumerate(set):
					if(y==0):
						train_set+=group
					else:
						test_set+=group
			iterations.append((train_set,test_set))
			
			print"\tFirst 10 random instances of training (out of %i)"%len(train_set)
			shuffle(train_set)
			for n in range(0,10):
				print "\t\t%s"%instance_tostring(train_set[n]).encode("utf-8")
			print"\t\t..."
			print"\tFirst 10 random instances of testing (out of %i)"%len(test_set)
			shuffle(test_set)
			for n in range(0,10):
				try:
					print "\t\t%s"%instance_tostring(test_set[n]).encode("utf-8")
				except:
					logger.error("max limit reached")
			print"\t\t..."
		logger.debug('# of instances in training set: %i'%len(train_set))
		logger.debug('# of instances in test set: %i'%len(test_set))
			
		results=[]	
		for y,i in enumerate(iterations):
			out=""
			out2=""
			train=i[0]
			test=i[1]
			
			for c,ins in enumerate(train):
				if(c!=0):
					out+="\n"
				for z,t in enumerate(ins):
					out+=token_tostring(t)
					if(z<len(ins)-1):
						out+="\n"
				if(c<len(train)-1):
					out+="\n"
			for c,ins in enumerate(test):
				if(c!=0):
					out2+="\n"
				for z,t in enumerate(ins):
					out2+=token_tostring(t)
					if(z<len(ins)-1):
						out2+="\n"
				if(c<len(test)-1):
					out2+="\n"
			train_file="%sfold_%i.train"%(EVAL_PATH,y+1)
			test_file="%sfold_%i.test"%(EVAL_PATH,y+1)
			model_file="%sfold_%i.mdl"%(EVAL_PATH,y+1)
			file=open(train_file,"w").write(out.encode("utf-8"))
			open(test_file,"w").write(out2.encode("utf-8"))
			#train_crfpp("%screx.tpl"%"data/",train_file,model_file)
			#train_crfpp("%sbaseline.tpl"%"crfpp_templates/",train_file,model_file)
			train_crfpp("%stemplate_1.tpl"%"crfpp_templates/",train_file,model_file)
			crf=CRF_classifier(model_file)
			errors=0
			fp=0
			tp=0
			fn=0
			tn=0
			prec=0.0
			acc=0.0
			tot_tokens=0
			
			print "Processing #Fold %i..."%(y+1)
			results = []
			for ex in test:
				tokens=[]
				for t in ex: 
					tokens.append(token_tostring(t))
					tot_tokens+=1
				# classify the instance
				res=crf.classify(tokens)
				logger.debug(instance_tostring(ex))
				logger.debug(result_to_string(res))
				results.append(res)
				for i in range(0,len(tokens)):
					try:
						gt_label=tokens[i].split('\t')[len(tokens[i].split('\t'))-1]
						token=tokens[i].split('\t')[0]
						class_label=res[i]['label']
						res[i]['gt_label']=gt_label
						alpha=res[i]['probs'][class_label]['alpha']
						beta=res[i]['probs'][class_label]['beta']
						p=res[i]['probs'][class_label]['prob']
						if(gt_label==class_label):
							if(gt_label=="O"):
								tn+=1
							else:
								tp+=1
						else:
							if(gt_label=="O"):
								fp+=1
							else:
								fn+=1
						logger.debug("%s -> GT_Label \"%s\" : Classified_Label \"%s\"\talpha: %s\tbeta: %s\tp: %s"%(token,gt_label,class_label,alpha,beta,p))
						try:
							assert (gt_label==class_label)
						except AssertionError:
							errors+=1
					except RuntimeError,e:
						print "Something went wrong"
			prec=tp/float(tp+fp)
			acc=(tp+tn)/float(tp+fp+tn+fn)
			rec=tp/float(tp+fn)
			fscore=2*(float(prec*rec)/float(prec+rec))
			tot_acc+=acc
			tot_prec+=prec
			tot_rec+=rec
			tot_fscore+=fscore
			logger.info("%i Errors out of %i tokens (in %i testing instances)"%(errors,tot_tokens,len(test)))
			logger.info("FOLD#%i: TP:%i\tFP:%i\tTN:%i\tFN:%i\tACC:%f\tPREC:%f\tRECALL:%f\tFSCORE:%f"%((y+1),tp,fp,tn,fn,acc,prec,rec,fscore))
			res={
				'accuracy':acc
				,'precision':prec
				,'recall':rec
				,'fscore':fscore
				,'true_pos':tp
				,'true_neg':tn
				,'false_pos':fp
				,'false_neg':fn
				,'test_set_size':0
				,'train_set_size':0
				}
			valid_res.append(res)
		
		#open('%soutput.html'%EVAL_PATH,'w').write(eval_results_to_HTML(results.encode("utf-8"))) #this html export is not working any longer
		for n,r in enumerate(results):
			print "%i %s\n"%(n+1," ".join(["%s/%s"%(n["token"],n["label"]) for n in r]))
			
		print "*********"
		print "Average fscore: %f"%(tot_fscore/float(iterations_num))
		print "Average accuracy: %f"%(tot_acc/float(iterations_num))
		print "Average precision: %f"%(tot_prec/float(iterations_num))
		print "Average recall: %f"%(tot_fscore/float(iterations_num))
		#print "Output in HTML format written to output.html"
		print "*********"
		print valid_res
		
	except RuntimeError,e:
		"Not able to prepare %s for training!"%fname	

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

class SimpleEvaluator:
	"""
	>>> import base_settings, settings
	>>> extractor_1 = citation_extractor(base_settings)
	>>> extractor_2 = citation_extractor(settings)
	>>> se = SimpleEvaluator([extractor_1,extractor_2],"data/75-02637.iob")
	
	"""
	def __init__(self,extractors,iob_test_file):
		"""
		Args:
			extractors: the list of canonical citation extractors to evaluate
			iob_test_file: the file in IOB format to be used for testing and evaluating the extactors
		"""
		
		self.test_instances = IO.file_to_instances(iob_test_file)
		temp = [[tok[0] for tok in inst]for inst in self.test_instances]
		self.string_instances = [" ".join(n) for n in temp]
		
		for eng in extractors:
			input = [eng.tokenize(inst) for inst in self.string_instances]
			output = eng.extract(input)
			to_evaluate = [[tuple([t["token"].decode("utf-8"),t["label"].decode("utf-8")]) for t in i] for i in output]
			results = self.evaluate(to_evaluate,self.test_instances)
			eval_results = results[0]
			print eval_results
			print "f-score %f"%self.calc_fscore(eval_results)
			print "accuracy %f"%self.calc_accuracy(eval_results)
			print self.calc_stats_by_tag(results[1])
		return
	
	@staticmethod
	def evaluate(l_tagged_instances,l_test_instances,negative_BIO_tag = u'O'):
		"""
		Evaluates a list of tagged instances against one of test instances (gold standard).
		>>> tagged = [[('ü','O'),('Hom','O'),('Il.','B-CREF')]]
		>>> test = [[('ü','O'),('Hom','B-CREF'),('Il.','I-CREF')]]
		>>> res = SimpleEvaluator.evaluate(tagged,test)
		>>> print res[0]
		{'false_pos': 2, 'true_pos': 0, 'true_neg': 1, 'false_neg': 0}
		>>> print SimpleEvaluator.calc_stats_by_tag(res[1])
		
		Args:
			l_tagged_instances:
				A list of instances. Each instance is a list of tokens, the tokens being tuples.
				Each tuple has the token (i=0) and the assigned label (i=1).
				For example:
				>>> l_instances = [[('vd.','O'),('Hom','B-CREF')]]
				
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
	

if __name__ == "__main__":
	#Usage example: python eval.py aph_data_100_positive/ out/
	#main()
	import doctest
	doctest.testmod(verbose=True)