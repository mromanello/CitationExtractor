#!/usr/bin/python
# -*- coding: utf-8 -*-
# author: Matteo Romanello, matteo.romanello@gmail.com


from SimpleXMLRPCServer import SimpleXMLRPCServer
from SimpleXMLRPCServer import SimpleXMLRPCRequestHandler
from citation_extractor.core import citation_extractorService
from citation_extractor.Utils.services import *
import sys,getopt
import daemon
		
def launch_server(host,port,path,cfg):
	try:
		CRefEx_XMLRPC_server(host,port,path,cfg)
	except Exception as e:
		print e

if __name__ == "__main__":
	port=8001
	host="localhost"
	path="/rpc/crex"
	config=None

	try:
		opts, args = getopt.getopt(sys.argv[1:], "ho:p:t:c:", ["help","host","port","path"])
	except getopt.GetoptError, err:
		print str(err) # will print something like "option -a not recognized"
		usage()
		sys.exit(2)
	for o, a in opts:
		if o in ("-h", "--help"):
			usage()
			sys.exit()
		elif o in ("-o", "--host"):
			host = a
		elif o in ("-p", "--port"):
			port = a
		elif o in ("-t", "--path"):
			path = a
		elif o in ("-c", "--config"):
			config = a

	if(config is not None):
		launch_server(host=host,port=int(port),path=path,cfg=config)
	else:
		print "You need at least a config file!"
