# -*- coding: utf-8 -*-
# author: Matteo Romanello, matteo.romanello@gmail.com

import pprint
import logging
import pkg_resources
import pickle
import random
import codecs
from citation_extractor.relex import prepare_for_training,relation_extractor
from citation_extractor.pipeline import read_ann_file_new
from sklearn.ensemble import RandomForestClassifier

#logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)
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
	rf_classifier = RandomForestClassifier(verbose=False,n_jobs=7)
	train_dir_name = 'data/aph_corpus/goldset/ann/'
	train_dir = pkg_resources.resource_filename('citation_extractor',train_dir_name)
	train_files = [(train_dir,file.replace('-doc-1.ann',''))
				for file in pkg_resources.resource_listdir('citation_extractor',train_dir_name) if '.ann' in file]
	rel_extractor = relation_extractor(rf_classifier,train_files)
	test_dir_name = 'data/aph_corpus/devset/ann/'
	test_dir = pkg_resources.resource_filename('citation_extractor',test_dir_name)
	test_files = [(test_dir,file.replace('-doc-1.ann',''))
				for file in pkg_resources.resource_listdir('citation_extractor',test_dir_name) if '.ann' in file]
	logger.info(rel_extractor)
	for test_dir,test_file in test_files[:20]:
		entities = read_ann_file_new(test_file, test_dir)[0]
		fulltext = codecs.open("%s%s-doc-1.txt"%(test_dir,test_file),'r','utf-8').read()
		if(len(entities.keys())>1):
			logger.info("Relations extracted for %s%s: %s"%(test_dir,test_file,rel_extractor.extract(entities,fulltext)))

def test_pickleability():
	"""
	TODO
	"""
	rf_classifier = RandomForestClassifier(verbose=False,n_jobs=7)
	train_dir_name = 'data/aph_corpus/goldset/ann/'
	train_dir = pkg_resources.resource_filename('citation_extractor',train_dir_name)
	train_files = [(train_dir,file.replace('-doc-1.ann',''))
				for file in pkg_resources.resource_listdir('citation_extractor',train_dir_name) if '.ann' in file]
	rel_extractor = relation_extractor(rf_classifier,train_files)
	pickled_rel_extractor = pickle.dumps(rel_extractor)
	unpickled_rel_extractor = pickle.loads(pickled_rel_extractor)