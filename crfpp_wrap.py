"""
Creates a wrapper around the CRF++ implementation
"""

import CRFPP
import sys,logging,os
import pprint

#global logger
moodule_logger = logging.getLogger('CREX.CRFPP_WRAP')

def train_crfpp(template_file,train_data_file,model_file):
		cmd="crf_learn -f 1 -t %s %s %s"%(template_file,train_data_file,model_file)
		os.popen(cmd).readlines()
		return

class CRF_classifier:
	def __init__(self,model_file,verb_level=2,best_out_n=2):
		self.logger = logging.getLogger('CREX.CRFPP_WRAP.CRFPP')
		try:
			self.m,self.v,self.bn=model_file,verb_level,best_out_n
			self.tagger = CRFPP.Tagger("-m %s -v %i -n%i"%(model_file,verb_level,best_out_n))
			self.logger.info("CRFPP Tagger initialized with command %s"%("-m %s -v %i -n%i"%(self.m,self.v,self.bn)))
		except RuntimeError, e:
			print "RuntimeError: ", e,
	
	def classify(self,l_tokens):
		out=[]
		self.tagger.clear()
		for t in l_tokens:
			t=t.encode("utf-8")
			self.tagger.add(t.decode("string-escape"))
		self.tagger.parse()
		size = self.tagger.size()
		xsize = self.tagger.xsize()
		ysize = self.tagger.ysize()
		for i in range(0, (size)):
		   res={}
		   feats=[]
		   res['id']=i+1
		   for j in range(0, (xsize)):
			if(j==0):
				res['token']=self.tagger.x(i, j)
			else:
				feats.append(self.tagger.x(i, j))
			res['features']=feats
		   self.logger.debug(res['features'])
		   res['label']=self.tagger.y2(i)
		   res['probs']={}
		   for j in range(0, (ysize)):
			tag=self.tagger.yname(j)
			probs={}
			vals = (float(self.tagger.prob(i,j)),float(self.tagger.alpha(i, j)),float(self.tagger.beta(i, j)))
			probs['prob']="%f"%vals[0]
			probs['alpha']="%f"%vals[1]
			probs['beta']="%f"%vals[2]
			res['probs'][tag]=probs
		   self.logger.info("%s => %s (%s)"%(res["token"].decode("utf-8"),res["label"].decode("utf-8"),str(res["probs"][res["label"]]['prob'])))
		   out.append(res)
		return out
	
if __name__ == "__main__":
    # crf_learn -t /56k/phd/code/python/crfx.tpl /56k/phd/code/python/doc1.train /56k/phd/code/python/crfx.mdl
    train_crfpp("/56k/phd/code/python/crfx.tpl","/56k/phd/code/python/doc1.train","/56k/phd/code/python/eval/new.mdl")
			