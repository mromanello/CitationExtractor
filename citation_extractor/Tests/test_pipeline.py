# -*- coding: utf-8 -*-
# author: Matteo Romanello, matteo.romanello@gmail.com

import pkg_resources
import logging
from citation_extractor.pipeline import read_ann_file_new

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_read_ann_file_new():
	dir = pkg_resources.resource_filename('citation_extractor','data/aph_corpus/goldset/ann/')
	files = [file.replace('-doc-1.ann','') for file in pkg_resources.resource_listdir('citation_extractor','data/aph_corpus/goldset/ann/') if '.ann' in file]
	for file in files[:10]:
		logger.debug(file)
		entities,relations,annotations = read_ann_file_new(file,dir)
		logger.debug("Entities: %s"%entities)
		for rel_id in relations:
			logger.debug(relations[rel_id])
			for entity_id in relations[rel_id]["arguments"]:
				assert entities.has_key(entity_id)
		logger.debug(annotations)
		for annotation in annotations:
			assert (relations.has_key(annotation["anchor"]) or entities.has_key(annotation["anchor"])) 
def test_preprocessing():
	"""
	TODO
	"""
	return
def test_do_ned():
	"""
	TODO
	"""
	return
def test_do_ner():
	"""
	TODO
	"""
	return
def test_do_relex():
	"""
	TODO
	"""
	return