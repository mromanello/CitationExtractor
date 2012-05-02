# -*- coding: utf-8 -*-
"""
.. module:: aph_corpus
   :platform: Unix
   :synopsis: TBD

.. moduleauthor:: Matteo Romanello <matteo.romanello@gmail.com>


"""
import urllib,json

def save_collection_allinone(download_dir,data):
	fname = "%s%s"%(download_dir,"all_in_one.iob")
	try:
		all_in_one_file = open(fname,"w")
		all_in_one = "\n\n".join(data)
		all_in_one_file.write(all_in_one)
		all_in_one_file.close()
		return fname
	except IOError as (errno, strerror):
		print "there was a problem writing file %s"%fname
		print "I/O error({0}): {1}".format(errno, strerror)

def save_collection_onebyone(download_dir,files,data):
	"""
	>>> coll_url = "http://cwkb.webfactional.com/aph_corpus/collections/C1/gold"
	>>> coll = download_collection(coll_url)
	>>> print len(coll)
	2
	>>> save_collection_allinone("/Users/56k/phd/code/APh/experiments/20121603/train/",coll[0],coll[1])
	/Users/56k/phd/code/APh/experiments/20121603/train/all_in_one.iob
	>>> save_collection_onebyone("/Users/56k/phd/code/APh/experiments/20121603/train/",coll[0],coll[1])
	"""
	out_files = []
	try:
		for i,f in enumerate(files):
			fname = "%s%s"%(download_dir,f.replace("/iob",".iob").replace("/",""))
			#print "Writing %s to disk"%fname
			file = open(fname,"w")
			file.write(data[i])
			file.close()
			#print "File %s written to disk."%fname
			#print "done"
			out_files.append(fname)
		return out_files
	except IOError as (errno, strerror):
		print "there was a problem writing file %s"%fname
		print "I/O error({0}): {1}".format(errno, strerror)

def download_collection(coll_url,download_dir):
	"""
	Simple utility function to download an APh collection.
	"""
	import urllib
	import json
	import codecs
	#coll_url = "http://cwkb.webfactional.com/aph_corpus/collections/C1/gold"
	data = json.loads(urllib.urlopen(coll_url).read()) # the json returned here is just a list of file names
	results = []
	for n,id in enumerate(data):
		url = "%s%s"%("http://cwkb.webfactional.com",id)
		print "Downloaded %s [%i/%i]"%(url,n+1,len(data))
		content = urllib.urlopen("%s"%url).read()
		try:
			fname = "%s%s"%(download_dir,id.replace("/iob",".iob").replace("/",""))
			file = codecs.open(fname,"w","utf-8")
			file.write(content.decode("utf-8"))
			file.close()
			print "Saved to file %s"%fname
		except Exception, e:
			raise e

def tokenise_collection(coll_name):
	return

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
		self.logger = logging.getLogger("CREX.ActiveLearner")
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
		self.logger.debug("Confidence Interval for \"%s\"%f"%(cand.token,cand.ci_score))
		return cand.ci_score < self.threshold
	
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
					cand = Candidate(tok["token"],"%s#%i"%(infile,n),probs[:2]) # just the 2 top most likely tags are considered
					if(self.is_effective_candidate(cand)):
						self.candidates.append(cand)
		self.candidates.sort(key=operator.attrgetter('ci_score'),reverse=True)
		return self.candidates
	
if __name__ == "__main__":
	import doctest
	doctest.testmod(verbose=True)