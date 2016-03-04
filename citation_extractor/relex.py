# -*- coding: utf-8 -*-
# author: Matteo Romanello, matteo.romanello@gmail.com

import logging
from citation_extractor.pipeline import read_ann_file_new

global logger
logger = logging.getLogger(__name__)

def prepare_for_training(doc_id, basedir):
	"""
	result = [
		[
			[
				"arg1_entity":"AAUTHOR"
				,"arg2_entity":"REFSCOPE"
				,"concent":"AAUTHORREFSCOPE"
			]
			,'scope_pos'
		]
		,[
			[
				"arg1_entity":"REFSCOPE"
				,"arg2_entity":"AAUTHOR"
				,"concent":"REFSCOPEAAUTHOR"
			]
			,'scope_neg'
		]
	]
	"""
	instances = []
	entities, relations, annotations = read_ann_file_new(doc_id, basedir)
	fulltext = ""
	logger.info("Preparing for training %s%s"%(basedir,doc_id))
	for rel_id in relations:
		arg1,arg2 = relations[rel_id]["arguments"]
		instances.append([extract_relation_features(arg1,arg2,entities,fulltext),'scope_pos'])
		instances.append([extract_relation_features(arg2,arg1,entities,fulltext),'scope_neg'])
	return instances
def extract_relation_features(arg1,arg2,entities,fulltext):
	"""
	the following features should be extracted:
		✓ Arg1_entity:AAUTHOR
		✓ Arg2_entity:REFSCOPE
		✓ ConcEnt: AAUTHORREFSCOPE
		✓ Thuc.=True (bow_arg1)
		✓ 1.8=True (bow_arg2)
		WordsBtw:0
		EntBtw:0 
		word_before_arg1
		word_after_arg1
		word_before_arg2
		word_after_arg2
	"""
	features = {}
	features["Arg1_entity"] = entities[arg1]["entity_type"]
	features["Arg2_entity"] = entities[arg2]["entity_type"]
	features["ConcEnt"] = "%s%s"%(entities[arg1]["entity_type"]
								,entities[arg2]["entity_type"])
	features["bow_arg1"] = entities[arg1]["surface"]
	features["bow_arg2"] = entities[arg2]["surface"]
	return features