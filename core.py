# -*- coding: utf-8 -*-
# author: Matteo Romanello, matteo.romanello@gmail.com

import getopt
from ConfigParser import SafeConfigParser
import os,re,string,logging,pprint,types,xmlrpclib,json
import citation_extractor
from citation_extractor.crfpp_wrap import *
from citation_extractor.Utils.IO import *

"""
This module contains the core of the citation extractor.
"""
global logger
logger = logging.getLogger('CREX')

def determine_path():
    """
	Borrowed from wxglade.py
	TODO move to Utils module
	"""
    try:
        root = __file__
        if os.path.islink (root):
            root = os.path.realpath (root)
        return os.path.dirname (os.path.abspath (root))
    except:
        print "I'm sorry, but something is wrong."
        print "There is no __file__ variable. Please contact the author."
        sys.exit ()

class citation_extractorService:
	"""
	TODO: Document
	"""
	def __init__(self,cfg_file=None):
		self.core = citation_extractor(cfg_file)
	#replace this method
	def test(self,arg):
		res = self.core.clf(arg)
		return self.core.output(res,"xml")
	def json(self,arg):
		res = self.core.clf(arg)
		return self.core.output(res,"json")
	
	def test_unicode(self,arg,outp):
		temp = arg.data.decode("utf-8")
		res = self.core.clf(temp)
		return self.core.output(res,outp)
	
	def version(self): 
		"""
		Return the version of citation_extractor
		"""
		logger.debug("Printing version")
		return citation_extractor.__version__

class CRFPP_Classifier:
	"""	
	This class should extend an abstract classifier	
	"""
	def __init__(self,train_file_name,template_file_name,dir):
		#dir=determine_path()+"/data/"
		fe = FeatureExtractor()
		path,fn = os.path.split(train_file_name)
		train_fname=dir+fn+'.train'
		t = fe.prepare_for_training(train_file_name)
		out=open(train_fname,'w').write(t.encode("utf-8"))
		# TODO the .mdl file should go somewhere else
		model_fname=dir+fn+'.mdl' 
		template_fname = template_file_name
		train_crfpp(template_fname,train_fname,model_fname)
		self.crf_model=CRF_classifier(model_fname)
		return
	
	def classify(self,tagged_tokens_list):
		"""
		Args:
			tagged_tokens_list: the list of tokens with tab separated tags.
		Returns:
			TODO document
		"""
		return self.crf_model.classify(tagged_tokens_list)
	

class citation_extractor:
	"""
	A Canonical Citation Extractor.
	First off, import the settings via module import
	>>> import settings #doctest: +SKIP
	
	Then create an extractor passing as argument the settings
	>>> extractor = citation_extractor(settings) #doctest: +SKIP
	
	Let's suppose now that want to extract the canonical references from the following string
	>>> example_text = u"Eschilo interprete di se stesso (Ar. Ran. 1126ss., 1138-1150)" #doctest: +SKIP
	
	Tokenise the text before passing it to the extractor
	>>> tokenised_example = extractor.tokenize(example_text) #doctest: +SKIP
	>>> result  = extractor.extract([tokenised_example,]) #doctest: +SKIP
	>>> " ".join(["%s/%s"%(n["token"],n["label"]) for n in result[0]]) #doctest: +SKIP
	'Eschilo/O interprete/O di/O se/O stesso/O (Ar./B-REFAUWORK Ran./I-REFAUWORK 1126ss.,/B-REFSCOPE 1138-1150)/I-REFSCOPE'
	
	Now let's create a second extractor, initialised with different settings
	
	>>> import settings #doctest: +SKIP
	>>> second_extractor = citation_extractor(settings) #doctest: +SKIP
	>>> result = second_extractor.extract([tokenised_example]) #doctest: +SKIP
	
	"""
	def __init__(self,options):
		self.classifier=None
		logfile = ""
		if(options.DEBUG):
			self.init_logger(loglevel=logging.DEBUG, log_file=options.LOG_FILE)
		else:
			self.init_logger(loglevel=logging.INFO, log_file=options.LOG_FILE)
		self.fe = FeatureExtractor()
		if(options.DATA_FILE != ""):
			self.classifier=CRFPP_Classifier(options.DATA_FILE,"%s%s"%(options.CRFPP_TEMPLATE_DIR,options.CRFPP_TEMPLATE),options.TEMP_DIR)
		elif(options.DATA_DIRS != ""):
			import glob
			import codecs
			all_in_one = []
			for dir in options.DATA_DIRS:
				# get all .iob files
				# concatenate their content with line return
				# write to a new file
				logger.debug("Processing %s"%dir)
				for infile in glob.glob( os.path.join(dir, '*.iob') ):
					logger.debug("Found the file %s"%infile)
					file_content = codecs.open("%s"%(infile), 'r',encoding="utf-8").read()
					all_in_one.append(file_content)
			result = "\n\n".join(all_in_one)
			codecs.open("%sall_in_one.iob"%options.TEMP_DIR, 'w',encoding="utf-8").write(result)
			self.classifier=CRFPP_Classifier("%sall_in_one.iob"%options.TEMP_DIR,"%s%s"%(options.CRFPP_TEMPLATE_DIR,options.CRFPP_TEMPLATE),options.TEMP_DIR)
	
	def init_logger(self,log_file=None, loglevel=logging.DEBUG):
		"""
		Initialise the logger
		"""
		if(log_file !="" or log_file is not None):
			logging.basicConfig(
				filename=log_file
				,level=loglevel,format='%(asctime)s - %(name)s - [%(levelname)s] %(message)s',filemode='w',datefmt='%a, %d %b %Y %H:%M:%S'
			)
			logger = logging.getLogger('CREX')
			logger.info("Logger initialised")
		else:
			logger = logging.getLogger('CREX')
			logger.setLevel(loglevel)
			ch = logging.StreamHandler()
			ch.setLevel(loglevel)
			formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
			ch.setFormatter(formatter)
			logger.addHandler(ch)
			logger.info("Logger initialised")
	
	def tokenize(self, blurb):
		"""
		Tokenize a string of text.
		
		Args:
			blurb: the string to tokenise.
		Returns:
			A list of tokens.
			
		"""
		return [y.split(" ") for y in blurb.split("\n")]
	
	def output(self,result,outp=None):
		"""
		Outputs the result of extraction/classification in several formats.
		"""
		fname = determine_path()+"/data/"+"temp.xml"
		f = open(fname,"w")
		temp = verbose_to_XML(result)
		f.write(temp)
		f.close()
		if(outp=="xml"):
			return temp
		elif(outp=="html"):
			import codecs
			fp = codecs.open(fname, "r", "utf-8")
			text = fp.read()
			fp.close()
			return out_html(text).decode("utf-8")
		elif(outp=="json"):
			return json.dumps(result)
	
	def extract(self, instances,legacy_features=None):
		"""
		Extracts canonical citations from a list of instances, such as sentences or other meaningful and 
		comparable subvisions of a text. This method acts as a proxy for the classify() method of the classifier.
		
		Args:
			instances: A list of instances, for example sentences.
		Returns:
			TODO describe
		"""
		result = []
		for n,instance in enumerate(instances):
			if(legacy_features is not None):
				feat_sets = self.fe.get_features(instances[n],[],False,legacy_features[n])
			else:
				feat_sets = self.fe.get_features(instances[n],[],False)
			result.append(self.classifier.classify(instance_to_string(feat_sets)))
		return result
	
class FeatureExtractor:
	"""
	A feature extractor to extract features from tokens.
	
	Usage:
		>>> fe = FeatureExtractor()
	"""

	def __init__(self):
		self.OTHERS=0
		# brackets
		self.PAIRED_ROUND_BRACKETS=1
		self.UNPAIRED_ROUND_BRACKETS=2
		self.PAIRED_SQUARE_BRACKETS=3
		self.UNPAIRED_SQUARE_BRACKETS=4
		# case
		self.MIXED_CAPS=5
		self.ALL_CAPS=6
		self.INIT_CAPS=7
		self.ALL_LOWER=8
		# punctuation
		self.FINAL_DOT=10
		self.CONTINUING_PUNCTUATION=11
		self.STOPPING_PUNCTUATION=12
		self.QUOTATION_MARK=13
		self.HAS_HYPHEN=14
		self.NO_PUNCTUATION=15
		# number
		self.YEAR=16
		self.RANGE=17
		self.DOT_SEPARATED_NUMBER=18
		self.DOT_SEPARATED_PLUS_RANGE=19
		self.NUMBER=20
		self.ROMAN_NUMBER=21
		self.NO_DIGITS=9
		self.MIXED_ALPHANUM=22
		# dictionaries
		self.MATCH_AUTHORS_DICT=23
		self.MATCH_WORKS_DICT=24
		self.CONTAINED_AUTHORS_DICT=25
		self.CONTAINED_WORKS_DICT=26
		# misc
		
		self.feat_labels=['i']*30
		self.feat_labels[self.OTHERS]="OTHERS"
		# brackets
		self.feat_labels[self.PAIRED_ROUND_BRACKETS]="PAIRED_ROUND_BRACKETS"
		self.feat_labels[self.UNPAIRED_ROUND_BRACKETS]="UNPAIRED_ROUND_BRACKETS"
		self.feat_labels[self.PAIRED_SQUARE_BRACKETS]="PAIRED_SQUARE_BRACKETS"
		self.feat_labels[self.UNPAIRED_SQUARE_BRACKETS]="UNPAIRED_SQUARE_BRACKETS"
		# case
		self.feat_labels[self.MIXED_CAPS]="MIXED_CAPS"
		self.feat_labels[self.ALL_CAPS]="ALL_CAPS"
		self.feat_labels[self.INIT_CAPS]="INIT_CAPS"
		self.feat_labels[self.ALL_LOWER]="ALL_LOWER"
		# punctuation
		self.feat_labels[self.FINAL_DOT]="FINAL_DOT"
		self.feat_labels[self.CONTINUING_PUNCTUATION]="CONTINUING_PUNCTUATION"
		self.feat_labels[self.STOPPING_PUNCTUATION]="STOPPING_PUNCTUATION"
		self.feat_labels[self.QUOTATION_MARK]="QUOTATION_MARK"
		self.feat_labels[self.HAS_HYPHEN]="HAS_HYPHEN"
		self.feat_labels[self.NO_PUNCTUATION]="NO_PUNCTUATION"
		# number
		self.feat_labels[self.NO_DIGITS]="NO_DIGITS"
		self.feat_labels[self.YEAR]="YEAR"
		self.feat_labels[self.RANGE]="RANGE"
		self.feat_labels[self.DOT_SEPARATED_NUMBER]="DOT_SEPARATED_NUMBER"
		self.feat_labels[self.DOT_SEPARATED_PLUS_RANGE]="DOT_SEPARATED_PLUS_RANGE"
		self.feat_labels[self.NUMBER]="NUMBER"
		self.feat_labels[self.ROMAN_NUMBER]="ROMAN_NUMBER"
		self.feat_labels[self.MIXED_ALPHANUM]="MIXED_ALPHANUM"
		# dictionaries
		self.feat_labels[self.MATCH_AUTHORS_DICT]="MATCH_AUTHORS_DICT"
		self.feat_labels[self.MATCH_WORKS_DICT]="MATCH_WORKS_DICT"
		self.feat_labels[self.CONTAINED_AUTHORS_DICT]="CONTAINED_AUTHORS_DICT"
		self.feat_labels[self.CONTAINED_WORKS_DICT]="CONTAINED_WORKS_DICT"
		# dictionary matching
		
		self.init_dictionaries()
	
	def init_dictionaries(self):
		from citation_extractor.Utils.FastDict import LookupDictionary
		import codecs
		try:
			# initialise works dictionary
			fname = dir="%s/data/works.csv"%determine_path()
			file = codecs.open(fname,"r","utf-8")
			raw_data = file.read()
			file.close()
			self.works_dict = LookupDictionary(raw_data.encode('utf-8'))
		except Exception, e:
			raise e
		
		try:
			# initialise authors dictionary
			fname = dir="%s/data/authors.csv"%determine_path()
			file = codecs.open(fname,"r","utf-8")
			raw_data = file.read()
			file.close()
			self.authors_dict = LookupDictionary(raw_data.encode('utf-8'))
		except Exception, e:
			raise e
		return
	
	def extract_bracket_feature(self,check_str):
		"""
		Extract a feature concerning the eventual presence of brackets
		
		Args:
			check_str: the string for which we need to extract features

		Returns:
			a tuple:
				result[0] : is the name of the feature
				result[1] : is the feature value expressed as integer

		Example:
			>>> tests = [u'(one)',u'another']
			>>> fe = FeatureExtractor()
			>>> [(tests[n],fe.feat_labels[fe.extract_bracket_feature(t)[1]]) for n,t in enumerate(tests)]
		"""
		res = None
		# define check regexps
		pair_sq_bra=re.compile(r'\[.*?\]')
		unpair_sq_bra=re.compile(r'[\[\]]')
		pair_rd_bra=re.compile(r'\(.*?\)')
		unpair_rd_bra=re.compile(r'[\(\)]')
		# execute checks
		if(pair_sq_bra.search(check_str)):
			res = self.PAIRED_SQUARE_BRACKETS
		elif(unpair_sq_bra.search(check_str)):
			res = self.UNPAIRED_SQUARE_BRACKETS
		elif(pair_rd_bra.search(check_str)):
			res = self.PAIRED_ROUND_BRACKETS
		elif(unpair_rd_bra.search(check_str)):
			res = self.UNPAIRED_ROUND_BRACKETS
		else:
			res = self.OTHERS
		return ("c_brackets",res)
	
	def extract_case_feature(self,check_str):
		"""
		Extracts a feature concerning the ortographic case of a token.

		Args:
			check_str: the string from which the feature will be extracted.
		Returns:
			A tuple TODO -> explain
		"""
		naked = re.sub('[%s]' % re.escape(string.punctuation), '', check_str)
		res = self.OTHERS
		if(naked.isalpha()):
			if(naked.isupper()):
				res = self.ALL_CAPS
			elif(naked.islower()):
				res = self.ALL_LOWER
			elif(naked[0].isupper()):
				res = self.INIT_CAPS
		return ("d_case",res)
	
	def extract_punctuation_feature(self,check_str):
		"""
		Checks the presence of hyphen and quotation marks.
		
		Args:
			check_str: the string for which we need to extract features

		Returns:
			a tuple:
				result[0] : is the name of the feature
				result[1] : is the feature value expressed as integer

		Example:
			>>> tests = [u'"',u'Iliad',u'"']
			>>> fe = FeatureExtractor()
			>>> [(tests[n],fe.feat_labels[fe.extract_punctuation_feature(t)[1]]) for n,t in enumerate(tests)]
			>>> tests = [u'«',u'De',u'uirginitate',u'»']
			>>> [(tests[n],fe.feat_labels[fe.extract_punctuation_feature(t)[1]]) for n,t in enumerate(tests)]
			
		"""
		res = self.OTHERS
		punct_exp=re.compile('[%s]' % re.escape(string.punctuation))
		final_dot=re.compile(r'.*?\.$')
		three_dots=re.compile(r'.*?\.\.\.$')
		cont_punct=re.compile(r'.*?[,;:]$')
		quot_punct=re.compile(r'.*?[\"\'«»]')
		if(three_dots.match(check_str)):
			res = self.OTHERS
		elif(final_dot.match(check_str)):
			res = self.FINAL_DOT
		elif(cont_punct.match(check_str)):
			res = self.CONTINUING_PUNCTUATION
		elif(quot_punct.match(check_str)):
			res = self.QUOTATION_MARK
		#elif(punct_exp.match(check_str)):
			#res = self.OTHER_PUNCTUATION
		return ("b_punct",res)
	
	def extract_number_feature(self,check_str):
		"""
		TODO
		1. first part of the features concerns the whole string
		2. second part should relate to the presence of number in a string
		* presence of range
		* presence of modern dates
		* is an ordinale number (Roman)?
		
		Example:
			>>> tests = [u'100',u'1994',u'1990-1999',u'23s.',u'10-11']
			>>> fe = FeatureExtractor()
			>>> [(tests[n],fe.feat_labels[fe.extract_number_feature(t)[1]]) for n,t in enumerate(tests)]
		"""
		res = self.OTHERS
		naked = re.sub('[%s]' % re.escape(string.punctuation), '', check_str).lower()
		is_modern_date_range = r"(\d{4}-\d{4})"
		
		if(naked.isdigit()):
			res = self.NUMBER
		elif(naked.isalpha()):
			res = self.NO_DIGITS
		elif(naked.isalnum()):
			res = self.MIXED_ALPHANUM
		return ("e_number",res)
	
	def extract_char_ngrams(self,inp,size=4):
		"""
		Extract ngram features (prefixes and suffixes), provided that the input string has a minimum length

		Args:
			inp: the string for which we need to extract features

		Returns:
			a list of tuples. each tuple:
				result[0] : is the name of the feature
				result[1] : is the feature value, in this case a string

		Example:
			>>> tests = [u'Hom',u'Esiodo',u'a']
			>>> fe = FeatureExtractor()
			>>> [fe.extract_char_ngrams(t) for t in tests]
		"""
		out=[]
		nd="ND"
		inp  = u"%s"%inp
		for i in range(0,4): # ngram prefixes
			i+=1
			if(len(inp) >= size): # string length matches minimum size				
				temp = ("f_ngram_%i"%i,inp[0:i])
			else:
				#  string length below minimum size
				temp = ("f_ngram_%i"%i,nd)
			out.append(temp)
		for i in range(0,4): # ngram suffixes
			i+=1
			if(len(inp) >= size):  # string length matches minimum size
				temp = ("g_ngram_%i"%(i),inp[len(inp)-i:])
			else:
				#  string length below minimum size
				temp = ("g_ngram_%i"%i,nd)
			out.append(temp)
		return out
	
	def extract_string_features(self,check_str):
		"""
		Extract string length and text only string lowercase
		"""
		out = re.sub('[%s]' % re.escape(string.punctuation), '', check_str)
		res = []
		if(not out==""):
			t = ('h_lowcase',out.lower())
			res.append(t)
			t = ('i_str-length',str(len(out)))
			res.append(t)
		else:
			t = ('h_lowcase','_')
			res.append(t)
			t = ('i_str-length',str(len(out)))
			res.append(t)
		res.append(('a_token',check_str))
		return res
	
	def extract_dictionary_feature(self,check_str):
		"""
		TODO
		* check that the string is actually a word (may not be necessary with different tokenisation)
		
		Example:
			>>> tests = [u'Hom.',u'Homér']
			>>> fe = FeatureExtractor()
			>>> [(tests[n],fe.feat_labels[fe.extract_dictionary_feature(t)[1]]) for n,t in enumerate(tests)]
		
		"""
		feature_name = "n_works_dictionary"
		match_works = self.works_dict.lookup(check_str.encode("utf-8"))
		match_authors = self.authors_dict.lookup(check_str.encode("utf-8"))
		result = (feature_name,self.OTHERS)
		
		if(len(match_authors)>0):
			for key in match_authors:
				if(len(match_authors[key]) == len(check_str)):
					result = (feature_name,self.MATCH_AUTHORS_DICT)
				else:
					result = (feature_name,self.CONTAINED_AUTHORS_DICT)
		elif(len(match_works)>0):
			for key in match_works:
				if(len(match_works[key]) == len(check_str)):
					result = (feature_name,self.MATCH_WORKS_DICT)
				else:
					result = (feature_name,self.CONTAINED_WORKS_DICT)
		else:
			result = (feature_name,self.OTHERS)
		return result
	
	def extract_word_length_feature(self,check_str,threshold=5):
		"""
		Features which gets fired when len(check_str) > threshold.
		TODO We should probably calculate (periodically) the average length for diff tags (aauthor,awork,refauwork).
		"""
		pass
	
	def extract_pattern_feature(self,check_str):
		"""
		>>> fe = FeatureExtractor()
		>>> test = u"Homéro,1999"
		>>> value = fe.extract_pattern_feature(test)
		>>> print value[1]
		Aaaaaa-0000
		"""
		result=[]
		for n,char in enumerate(check_str):
			if(char.isalnum()):
				if(char.isalpha()):
					if(char.islower()):
						result.append('a')
					else:
						result.append('A')
				else:
					result.append('0')
			else:
				result.append('-')
		return ('l_pattern',"".join(result))
	
	def extract_compressed_pattern_feature(self,check_str):
		"""
		>>> fe = FeatureExtractor()
		>>> test = u"Homéro,1999"
		>>> value = fe.extract_compressed_pattern_feature(test)
		>>> print value[1]
		Aa-0
		"""
		result=[]
		for n,char in enumerate(check_str):
			if(char.isalnum()):
				if(char.isalpha()):
					if(char.islower()):
						if(n+1 <= len(check_str)-1 and check_str[n+1].islower()):
							pass
						else:
							result.append('a')
					else:
						if(n+1 <= len(check_str)-1 and check_str[n+1].isupper()):
							pass
						else:
							result.append('A')
				else:
					if(n+1 <= len(check_str)-1 and check_str[n+1].isdigit()):
						pass
					else:
						result.append('0')
			else:
				if(n+1 <= len(check_str)-1 and (check_str[n+1].isalnum() is False)):
					pass
				else:
					result.append('-')
		return ('m_compressed-pattern',"".join(result))
	
	def extract_features(self,inp):
		feature_set=[]
		feat_funcs=[self.extract_punctuation_feature
		,self.extract_bracket_feature
		,self.extract_case_feature
		,self.extract_number_feature
		,self.extract_char_ngrams
		,self.extract_string_features
		,self.extract_pattern_feature
		,self.extract_compressed_pattern_feature
		,self.extract_dictionary_feature]
		for f in feat_funcs:
			result = f(inp)
			if(type(result) == types.TupleType):
				feature_set.append(result)
			elif(type(result) == types.ListType):
				for r in result:
					feature_set.append(r)	
		return feature_set
	
	def get_features(self,instance,labels=[],outp_label=True, legacy_features=None):
		"""
		Args:
			instance:
				the instance to be classified, represented as a list of tokens.
			labels:
				...
			outp_label:
				...
			
		Example:
			>>> fe = FeatureExtractor()
			>>> test = ['cf.', 'Hom', 'Il.', '1.1', ';']
			>>> postags = [('z_POS','N/A'),('z_POS','N/A'),('z_POS','N/A'),('z_POS','N/A'),('z_POS','N/A')]
			>>> tmp = fe.get_features(test,outp_label=False,legacy_features=postags)
		"""
		features = [self.extract_features(tok) for tok in instance]
		tok1 = features[0]
		keys = [f[0] for f in tok1]
		res = [dict(r) for r in features]
		logger = logging.getLogger('CREX.FeatureExtractor')
		logger.debug(res)
		for n,x in enumerate(res):
			# transform the numeric values into strings
			for key in keys:
				if(type(x[key]) is type(12)):
					x[key] = self.feat_labels[x[key]] # get the string label corresponding to a given int value
					#x[key] = str(x[key]) # leave the numeric feature value
			if(outp_label is True):
				x['z_gt_label']=labels[n]
		if(legacy_features is not None):
			for n,token in enumerate(res):
				token[legacy_features[n][0]] = legacy_features[n][1]
		return res
	
	def get_feature_order(self):
		"""
		Returns the order in which the features are output.
		
		Example:
			>>> fe = FeatureExtractor()
			>>> fe.get_feature_order()
		"""
		dumb_tok = ("Test.","O")
		temp = self.get_features([dumb_tok[0]],[dumb_tok[1]])[0]
		return [k for k in sorted(temp.keys())]
	
	def prepare_for_training(self,file_name):
		"""
		#TODO: can be made staticmethod at some point

		Args:
			file_name: the input file in IOB format

		Returns:
			TODO document

		Example:
			>>> fe = FeatureExtractor() #doctest: +SKIP
			>>> print fe.prepare_for_training("data/75-02637.iob") #doctest: +SKIP
		"""
		import codecs
		fp = codecs.open(file_name, "r", "utf-8")
		comment=re.compile(r'#.*?')
		lines = fp.read()
		instances=[group.split('\n')for group in lines.split("\n\n")]
		all_tokens = []
		all_labels = []
		all_postags = []
		for inst in instances:
			labels= []
			tokens=[]
			postags=[]
			for line in inst:
				if(not comment.match(line)):
					temp = line.split('\t')
					if(len(temp) == 2):
						tokens.append(temp[0])
						labels.append(temp[1])
					else:
						tokens.append(temp[0])
						labels.append(temp[2])
						postags.append(temp[1])
			all_labels.append(labels)
			all_tokens.append(tokens)
			if(len(postags) > 0):
				all_postags.append(postags)
		if(len(all_postags) > 0): 
			all_postags = [[("z_POS",token) for token in instance] for instance in all_postags]
			res2 = [self.get_features(r,all_labels[n],legacy_features=all_postags[n]) for n,r in enumerate(all_tokens)]
		else:
			res2 = [self.get_features(r,all_labels[n]) for n,r in enumerate(all_tokens)]
		# all this fuss is to have instances and feature sets as text
		res2 = [instance_to_string(r) for r in res2]
		res3 = ["\n".join(i) for i in res2]
		out = "\n\n".join(res3)
		return out
	
if __name__ == "__main__":
	import doctest
	doctest.testmod()

