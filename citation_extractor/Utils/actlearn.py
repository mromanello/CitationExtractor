# -*- coding: utf-8 -*-
"""
.. module:: aph_corpus
   :platform: Unix
   :synopsis: TBD

.. moduleauthor:: Matteo Romanello <matteo.romanello@gmail.com>


"""

class Candidate:
	"""
	>>> tok = "Hom."
	>>> inst_id = 1
	>>> c = Candidate(tok,inst_id)
	>>> print c
	"""
	def __init__(self, token, instance_id, probs = None, ci = None):
		self.token = token
		self.instance = instance_id
		self.probs = probs
		self.ci_score = ci
		return
	
	def __str__(self):
		return "<Candidate: token=\"%s\", instance=\"%s\", best_2_labels=\"%s\">"%(self.token,self.instance,str(self.probs))
	
class ActiveLearner:
	"""
	TODO: give at least a reference to Poesio et al.
	>>> al = ActiveLearner()
	"""
	def __init__(self, extractor=None,threshold=0.2,dev_set=None,test_set=None):
		import logging
		#from citation_extractor import eval
		self.logger = logging.getLogger("CREX.ActiveLearner")
		self.token_count = 0
		# set extractor
		try:
			assert extractor is not None
			self.classifier = extractor
			self.logger.info("Classifier set to %s"%self.classifier)
		except Exception, e:
			raise e
		# set threshold
		try:
			self.threshold = float(threshold)
			self.logger.info("Threshold set to %f"%self.threshold)
		except Exception, e:
			raise e
		# set dev and test directories
		try:
			assert (dev_set is not None) and (test_set is not None)
			self.dev_set = dev_set
			self.test_set = test_set
		except Exception, e:
			raise e
		
		self.candidates = []
	
	@staticmethod
	def calc_confidence_interval(prob_lab1,prob_lab2):
		"""
		This function computes the Confidence Interval (CI) for a given token.
		"""
		return float(prob_lab1) - float(prob_lab2)
	
	def is_effective_candidate(self,cand):
		"""
		Check if a token is an effective candidate.
		Computes the CI value and return True if ? than the threshold values, False if otherwise.
		
		Args:
			cand:
				The candidate to check.
		"""
		# check if greater or smaller than... TODO
		cand.ci_score = self.calc_confidence_interval(cand.probs[0][1],cand.probs[1][1])
		result = cand.ci_score < self.threshold
		if(result):
			self.logger.info("Confidence Interval for \"%s\" %f = %f (%s) - %f (%s)"%(cand.token,cand.ci_score,float(cand.probs[0][1]),cand.probs[0][0],float(cand.probs[1][1]),cand.probs[1][0]))
		else:
			self.logger.debug("Confidence Interval for \"%s\" %f = %f (%s) - %f (%s)"%(cand.token,cand.ci_score,float(cand.probs[0][1]),cand.probs[0][0],float(cand.probs[1][1]),cand.probs[1][0]))
	
	def get_pruned_candidates(self):
		"""
		Prune the candidates: basically a distinct over APh notice ids.
		The number of candidates to keep is then left to the user.
		"""
		import re
		[self.logger.debug("%s -> %f"%(c.token,c.ci_score)) for c in self.candidates]
		results = [re.match(r"(.*?)(\d+-\d+[a-z]?)(\..*)",c.instance).groups()[1] for c in self.candidates]
		d1 = dict((k,v) for v,k in enumerate(reversed(results)))
		return sorted(d1,key=d1.get,reverse=True)
	
	def learn(self):
		"""
		What the function does:
			* read dev-set
			* for file in dev-set:
			 * for instance in file:
				* res = extract(instance)
				* for tok in res:
					* cand = Candidate(res) # more complex than this, actually
					* if(is_effective_candidate(cand)):
						* self.candidates.append(cand)
		"""
		import glob
		import os
		import operator
		from citation_extractor.Utils import IO
		
		for infile in glob.glob(os.path.join(self.dev_set, '*.iob')):
			instances = IO.file_to_instances(infile)
			string_instances = [[tok[0] for tok in i]for i in instances]
			results = self.classifier.extract([string_instances])
			for n,r in enumerate(results):
				for tok in r:
					probs = [(tag,tok["probs"][tag]["prob"]) for tag in tok["probs"].keys()] # extract the probabilities for each tag
					probs.sort(key=lambda tup: tup[1],reverse=True)
					self.logger.debug(probs)
					cand = Candidate(tok["token"],"%s#%i"%(infile,n),probs[:2]) # just the 2 top most likely tags are considered
					if(self.is_effective_candidate(cand)):
						self.candidates.append(cand)
					self.token_count+=1
		self.candidates.sort(key=operator.attrgetter('ci_score'),reverse=True)
		return self.candidates
	
	@staticmethod
	def select_candidates(settings):
		"""
		Run the ActiveLearner and select a set of effective candidates.
		"""
		from citation_extractor.core import citation_extractor
		from citation_extractor.Utils import aph_corpus
		
		extr = citation_extractor(settings)
		#example_text = u"Eschilo interprete di se stesso (Ar. Ran. 1126ss., 1138-1150)"
		#tokens = extr.tokenize(example_text)
		#result = extr.extract([tokens])
		# create an ActiveLearner instance
		al = ActiveLearner(extr,0.2,settings.DEV_DIR,settings.TEST_DIR)
		candidates = al.learn()
		pruned_candidates = al.get_pruned_candidates()
		al.logger.info("Total tokens classified: %i"%al.token_count)

		effective_candidates_detail = "\n".join(["[%s] %s -> %f"%(c.instance,c.token,c.ci_score) for c in candidates])
		file = open("%sec_details.txt"%settings.TEMP_DIR,"w")
		file.write(effective_candidates_detail)
		file.close()

		effective_candidate_list = "\n".join(["%s/%s\t%s"%(n,len(al.get_pruned_candidates()),id) for n,id in enumerate(pruned_candidates)])
		file = open("%sec_list.txt"%settings.TEMP_DIR,"w")
		file.write(effective_candidate_list)
		file.close()
	
	@staticmethod
	def test_improvement(pre_settings,post_settings):
		"""
		TODO: what this function should do:
		1. run without selected candidates in the train set and evaluate
		2. run with selected candidates in the train set and evaluate
		3. return: stats for the 1st run, stats for the 2nd run and improvement obtained 
		"""
		from citation_extractor.core import citation_extractor
		from citation_extractor.eval import SimpleEvaluator
		from citation_extractor.Utils import aph_corpus
		from citation_extractor.Utils import IO
		# extractor without selected candidates in the train set and evaluate
		pre_extractor = citation_extractor(pre_settings)
		# extractor with selected candidates in the train set and evaluate
		post_extractor = citation_extractor(post_settings)
		# initialise evaluator and evaluate against the test set
		se = SimpleEvaluator([pre_extractor,post_extractor],post_settings.TEST_DIR)
		results = se.eval()
		print "***data***"
		print "pre-active learning TRAIN-SET: %s"%str(pre_settings.DATA_DIRS)
		train_details = aph_corpus.get_collection_details(pre_settings.TRAIN_COLLECTIONS)
		print "pre-active learning TRAIN-SET: # tokens = %i; # NEs = %i"%(train_details['total_token_count'],train_details['ne_token_count'])
		train_details = aph_corpus.get_collection_details(post_settings.TRAIN_COLLECTIONS)
		print "post-active learning TRAIN-SET: %s"%str(post_settings.DATA_DIRS)
		print "post-active learning TRAIN-SET: # tokens = %i; # NEs = %i"%(train_details['total_token_count'],train_details['ne_token_count'])
		test_details = aph_corpus.get_collection_details(post_settings.TEST_COLLECTIONS)
		print "TEST-SET: %s"%str(post_settings.TEST_DIR)
		print "TEST-SET details: # tokens = %i; # NEs = %i\n"%(test_details['total_token_count'],test_details['ne_token_count'])
		print "*** pre-active learning ***"
		pre_al_results = results[str(pre_extractor)][0]
		print "fscore: %f \nprecision: %f\nrecall: %f\n"%(pre_al_results["f-score"]*100,pre_al_results["precision"]*100,pre_al_results["recall"]*100)
		print "*** post-active learning ***"
		post_al_results = results[str(post_extractor)][0]
		print "fscore: %f \nprecision: %f\nrecall: %f\n"%(post_al_results["f-score"]*100,post_al_results["precision"]*100,post_al_results["recall"]*100)
		print "*** post-active learning gain (%) ***"
		print "fscore: %f \nprecision: %f\nrecall: %f\n"%(post_al_results["f-score"]*100 - pre_al_results["f-score"]*100,post_al_results["precision"]*100 - pre_al_results["precision"]*100,post_al_results["recall"]*100 - pre_al_results["recall"]*100)
		IO.write_iob_file(se.output[str(pre_extractor)],"%spre_out.data"%post_settings.OUT_DIR)
		IO.write_iob_file(se.output[str(post_extractor)],"%spost_out.data"%post_settings.OUT_DIR)
	
	@staticmethod
	def tag_candidates(settings):
		import glob
		import os
		import codecs
		from citation_extractor.Utils import IO
		from citation_extractor.core import citation_extractor
		
		extractor = citation_extractor(settings)
		for infile in glob.glob( os.path.join(settings.CANDIDATES_DIR, '*.iob') ):
			print "processing %s"%infile
			instances = IO.file_to_instances(infile)
			string_instances = [[tok[0] for tok in i]for i in instances]
			results = extractor.extract([string_instances])
			out_dir = settings.OUT_DIR
			out_fname = "%s%s"%(out_dir,os.path.basename(infile))
			file = codecs.open(out_fname, 'w',encoding="utf-8")
			instances = ["\n".join(["%s\t%s"%(t["token"].decode("utf-8"),t["label"]) for t in r]) for r in results]
			file.write("\n\n".join(instances))
			file.close()
			print "output written to %s"%out_fname

if __name__ == "__main__":
	import doctest
	doctest.testmod(verbose=True)