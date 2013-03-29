#!/usr/bin/python
# -*- coding: UTF-8 -*-
# author: Matteo Romanello, matteo.romanello@gmail.com

from SimpleXMLRPCServer import SimpleXMLRPCServer
from SimpleXMLRPCServer import SimpleXMLRPCRequestHandler
from citation_extractor.core import citation_extractorService
import sys,logging

global logger
logger = logging.getLogger('CREX.service')

class CRefEx_XMLRPC_server:
	"""docstring for CRefEx_XMLRPC_server"""
	class RequestHandler(SimpleXMLRPCRequestHandler):
		rpc_paths = ["/rpc/crex"]
		
	def __init__(self, host="localhost",port=8001,path="/rpc/crex",config=None):
		try:
			server = SimpleXMLRPCServer((host, port),requestHandler=CRefEx_XMLRPC_server.RequestHandler)
			server.register_introspection_functions()
			server.register_instance(citation_extractorService(config))
			server.serve_forever()
			global logger
			logger.info("Service started!")
		except Exception, e:
			raise e
