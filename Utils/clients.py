#!/usr/bin/python
# -*- coding: UTF-8 -*-

import xmlrpclib,pprint,traceback
import simplejson as json
#from nltk import regexp_tokenize
from citation_extractor.Utils.IO import *
from citation_extractor.core import *
pp = pprint.PrettyPrinter(indent=5)

class CRefEx_XMLRPC_client:
	"""docstring for CRefEx_XMLRPC_client"""
	def __init__(self, host, port, path):
		self.host = host
		self.port = port
		self.path = path
		self.proxy = None
		self.connect()
	def connect(self):
		"""docstring for fname"""
		try:
			self.proxy = xmlrpclib.ServerProxy("http://%s:%i%s"%(self.host,self.port,self.path))
		except Exception, e:
			traceback.print_exc()
			raise e