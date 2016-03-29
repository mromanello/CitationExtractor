# -*- coding: utf-8 -*-
# author: Matteo Romanello, matteo.romanello@gmail.com

import pprint
import logging
import pkg_resources
import random
from citation_extractor.relex import prepare_for_training

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_prepare_training():
	dir = pkg_resources.resource_filename('citation_extractor','data/aph_corpus/goldset/ann/')
	files = [file.replace('-doc-1.ann','') for file in pkg_resources.resource_listdir('citation_extractor','data/aph_corpus/goldset/ann/') if '.ann' in file]
	logger.info("Found %i \".ann\" files in %s"%(len(files),dir))
	prepare_for_training("75-02534.txt",dir)
	#pprint.pprint(prepare_for_training("75-00992.txt",dir))
	#pprint.pprint(prepare_for_training(files[random.randint(0,len(files)-1)],dir))
def test_relation_extractor():
	"""
	create a `relation_extractor`
	train it with the goldset
	test it against some instances from the devset
	"""
	pass
def test_pickleability():
	pass