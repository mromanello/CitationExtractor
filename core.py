# -*- coding: utf-8 -*-
# author: Matteo Romanello, matteo.romanello@gmail.com

import getopt
from ConfigParser import SafeConfigParser
import os,re,string,logging,pprint,types,xmlrpclib,json
import citation_extractor
from citation_extractor.crfpp_wrap import *
from citation_extractor.Utils.IO import *

"""
Description
"""

logger = logging.getLogger('CREX')

pp = pprint.PrettyPrinter(indent=5)

def determine_path ():
    """Borrowed from wxglade.py"""
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
	def __init__(self,cfg_file=None):
		self.core = CRefEx(cfg_file)
	
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
		Return the version of CRefEx
		"""
		logger.debug("Printing version")
		return citation_extractor.__version__

"""	
This class should extend an abstract classifier	
"""
class CRFPP_Classifier:
	def __init__(self,train_file_name):
		dir=determine_path()+"/data/"
		fe = FeatureExtractor()
		path,fn = os.path.split(train_file_name)
		train_fname=dir+fn+'.train'
		t = fe.prepare_for_training(train_file_name)
		out=open(train_fname,'w').write(t.encode("utf-8"))
		model_fname=dir+fn+'.mdl'
		train_crfpp(dir+"crex.tpl",train_fname,model_fname)
		self.crf_model=CRF_classifier(model_fname)
		return
	
	def classify(self,tagged_tokens_list):
		"""
		@param tagged_tokens_list the list of tokens with tab separated tags
		"""
		return self.crf_model.classify(tagged_tokens_list)
		
class CRefEx:
	"""Canonical Reference Extractor"""
	def __init__(self,cfg_file=None,training_model=None,training_file=None):
		#self.init_logger(loglevel=logging.DEBUG,log_file="/Users/56k/phd/code/python_crex/citation_extractor/crfx_3.log")
		self.read_config_file(cfg_file)
		self.training_model=training_model
		self.classifier=None
		self._default_training_dir="data/"
		self._default_training_file="test.txt"
		self.fe = FeatureExtractor()
		
		if(training_model=="CRF" or training_model is None):
			if(training_file is not None):
				self.classifier=CRFPP_Classifier(training_file)
			else:
				# read the default value
				self.classifier=CRFPP_Classifier("%s/%s%s"%(determine_path(),self._default_training_dir,self._default_training_file))
	
	def read_config_file(self,cfg_file):
		parser = SafeConfigParser()
		parser.read(cfg_file)
		verbose = parser.getboolean("logging","verbose")
		logfile = parser.get("logging","file")
		if(verbose):
			self.init_logger(loglevel=logging.DEBUG, log_file=logfile)
		else:
			self.init_logger(loglevel=logging.INFO, log_file=logfile)
		logger.info("Reading configuration from file %s"%cfg_file)
		pass
	
	"""
	Initialise the logger
	"""
	def init_logger(self,log_file=None, loglevel=logging.DEBUG):
		if(log_file !="" or log_file is not None):
			logging.basicConfig(filename=log_file,level=loglevel,format='%(asctime)s - %(name)s - [%(levelname)s] %(message)s',filemode='w',datefmt='%a, %d %b %Y %H:%M:%S')
			logger = logging.getLogger('CREX')
			logger.info("Logger initialised with %s, %s"%(log_file,str(loglevel)))
		else:
			logger = logging.getLogger('CREX')
			logger.setLevel(loglevel)
			ch = logging.StreamHandler()
			ch.setLevel(loglevel)
			formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
			ch.setFormatter(formatter)
			logger.addHandler(ch)
			logger.info("Logger initialised")
			
			
	
	"""
	Utility function to tokenise a text.
	"""
	def tokenize(self, blurb):
		return [y.split(" ") for y in blurb.split("\n")]
	"""
	Classify method.
	"""
	def clf(self, text):
		if(type(text) is not type(unicode("string"))):
			text = unicode(text,"utf-8")
		temp = self.tokenize(text)
		return self.classify(temp)
		
	def output(self,result,outp=None):
		"""docstring for output"""
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
				
	# actually it's a proxy method
	def classify(self, instances,input="text"):
		res = []
		for i in instances:
			feat_sets = self.fe.get_features(i,[],False)
			res.append(self.classifier.classify(instance_to_string(feat_sets)))		
		return res
		
class FeatureExtractor:
	"""
	A feature extractor to extract features from tokens.
	"""
	def __init__(self):
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
		self.OTHERS=0
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
		self.feat_labels=['i']*30
		# brackets
		self.feat_labels[1]="PAIRED_ROUND_BRACKETS"
		self.feat_labels[2]="UNPAIRED_ROUND_BRACKETS"
		self.feat_labels[3]="PAIRED_SQUARE_BRACKETS"
		self.feat_labels[4]="UNPAIRED_SQUARE_BRACKETS"
		# case
		self.feat_labels[5]="MIXED_CAPS"
		self.feat_labels[6]="ALL_CAPS"
		self.feat_labels[7]="INIT_CAPS"
		self.feat_labels[8]="ALL_LOWER"
		# punctuation
		self.feat_labels[10]="FINAL_DOT"
		self.feat_labels[11]="CONTINUING_PUNCTUATION"
		self.feat_labels[12]="STOPPING_PUNCTUATION"
		self.feat_labels[13]="QUOTATION_MARK"
		self.feat_labels[14]="HAS_HYPHEN"
		self.feat_labels[15]="NO_PUNCTUATION"
		# number
		self.feat_labels[9]="NO_DIGITS"
		self.feat_labels[16]="YEAR"
		self.feat_labels[17]="RANGE"
		self.feat_labels[18]="DOT_SEPARATED_NUMBER"
		self.feat_labels[19]="DOT_SEPARATED_PLUS_RANGE"
		self.feat_labels[20]="NUMBER"
		self.feat_labels[21]="ROMAN_NUMBER"
		self.feat_labels[22]="MIXED_ALPHANUM"
		self.feat_labels[0]="OTHERS"
		
	def extract_bracket_feature(self,check_str):
		"""
		Extract a feature concerning the eventual presence of brackets
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
		return ("brackets",res)
		
	def extract_case_feature(self,check_str):
		"""
		Extract a feature concerning the ortographic case of a token
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
		return ("case",res)
		
	def extract_punctuation_feature(self,check_str):
		res = self.OTHERS
		punct_exp=re.compile('[%s]' % re.escape(string.punctuation))
		final_dot=re.compile(r'.*?\.$')
		three_dots=re.compile(r'.*?\.\.\.$')
		cont_punct=re.compile(r'.*?[,;:]$')
		if(three_dots.match(check_str)):
			res = self.OTHERS
		elif(final_dot.match(check_str)):
			res = self.FINAL_DOT
		elif(cont_punct.match(check_str)):
			res = self.CONTINUING_PUNCTUATION
		return ("punct",res)
		
	def extract_number_feature(self,check_str):
		"""
		TODO
		"""
		res = self.OTHERS
		naked = re.sub('[%s]' % re.escape(string.punctuation), '', check_str).lower()
		if(naked.isdigit()):
			res = self.NUMBER
		elif(naked.isalpha()):
			res = self.NO_DIGITS
		elif(naked.isalnum()):
			res = self.MIXED_ALPHANUM
		return ("number",res)
		
	def extract_char_ngrams(self,inp):
		"""
		"""
		size=4
		out=[]
		inp  = u"%s"%inp
		for i in range(0,4):
			i+=1
			temp = ("subs %i"%i,inp[0:i])
			out.append(temp)
		for i in range(0,4):
			i+=1
			temp = ("susb -%i"%(i),inp[len(inp)-i:])
			out.append(temp)
		return out
		
	def extract_string_features(self,check_str):
		"""
		Extract string length and text only string lowercase
		"""
		out = re.sub('[%s]' % re.escape(string.punctuation), '', check_str)
		res = []
		if(not out==""):
			t = ('lowcase',out.lower())
			res.append(t)
			t = ('str-length',str(len(out)))
			res.append(t)
		else:
			t = ('lowcase','_')
			res.append(t)
			t = ('str-length',str(len(out)))
			res.append(t)
		res.append(('a_token',check_str))
		return res
	
	def extract_features(self,inp):
		feature_set=[]
		feat_funcs=[self.extract_punctuation_feature,
		self.extract_bracket_feature,
		self.extract_case_feature,
		self.extract_number_feature,
		self.extract_char_ngrams,
		self.extract_string_features]
		for f in feat_funcs:
			result = f(inp)
			if(type(result) == types.TupleType):
				feature_set.append(result)
			elif(type(result) == types.ListType):
				for r in result:
					feature_set.append(r)	
		return feature_set
		
	def get_features(self,instance,labels=[],outp_label=True):
		out = [self.extract_features(tok) for tok in instance]
		res = [dict(r) for r in out]
		# transform the numeric values into strings
		logger = logging.getLogger('CREX.FeatureExtractor')
		logger.debug(res)
		for n,x in enumerate(res):
			for m,key in enumerate(x.iterkeys()):
				if(type(x[key]) is type(12)):
					x[key] = self.feat_labels[x[key]]
			if(outp_label is True):
				x['z_gt_label']=labels[n]
		return res

	def prepare_for_training(self,file_name):
		"""
		@param file_name the input file in IOB format
		@return 
		"""
		import codecs
		fp = codecs.open(file_name, "r", "utf-8")
		comment=re.compile(r'#.*?')
		lines = fp.read()
		instances=[group.split('\n')for group in lines.split("\n\n")]
		res = []
		all_labels = []
		for inst in instances:
			labels= []
			tokens=[]
			for line in inst:
				if(not comment.match(line)):
					tokens.append(line.split('\t')[0])
					labels.append(line.split('\t')[1])
			all_labels.append(labels)
			res.append(tokens)
		res2=[]
		for n,r in enumerate(res):
			res2.append(self.get_features(r,all_labels[n]))
		# all this fuss is to have instances and feature sets as text
		res2=[instance_to_string(r) for r in res2]
		res3 = ["\n".join(i) for i in res2]
		out = "\n\n".join(res3)
		return out
	
def main():
	pass

	

if __name__ == "__main__":
	conf_file=None
	tests=["Hom. Il. 1.125) is a citation", u"this is a string Il. 1.125 randomÜ Hom. Il. 1.125 γρα",u"Eschilo interprete di Ü se stesso (Ar. Ran. 1126s. e 1138-1150)"]
	try:
		opts, args = getopt.getopt(sys.argv[1:], "hc:i:", ["help","conf_file","input"])
	except getopt.GetoptError, err:
		print str(err) # will print something like "option -a not recognized"
		usage()
		sys.exit(2)
	for o, a in opts:
		if o == "-v":
			verbose = True
		elif o in ("-h", "--help"):
			usage()
			sys.exit()
		elif o in ("-i", "--inp"):
			import codecs
			f = codecs.open(a, "r", "utf-8" )
			tests = f.read().split("\n")
		elif o in ("-c", "--config"):
			conf_file = a
			if(conf_file is not None):
				c=CRefEx(cfg_file=conf_file)
				"""
				TODO:	The tests should be read directly from a file.
				Path to file should go in the configuration file
				"""
				
				for n,t in enumerate(tests):
					logger.info("*** Quick Test %i***"%(n+1))
					result = c.clf(t)
					readable_result = ["%s/%s"%(n["token"],n["label"]) for n in result[0]]
					logger.info(" ".join(readable_result).decode("UTF-8"))
				
		else:
			assert False, "unhandled option"

