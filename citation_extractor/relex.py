# -*- coding: utf-8 -*-
# author: Matteo Romanello, matteo.romanello@gmail.com

import logging
import codecs
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
	fulltext = codecs.open("%s%s-doc-1.txt"%(basedir,doc_id),'r','utf-8').read()
	logger.debug(entities)
	logger.debug(relations)
	logger.debug(fulltext)
	logger.info("Preparing for training %s%s"%(basedir,doc_id))
	for rel_id in relations:
		arg1,arg2 = relations[rel_id]["arguments"]
		instances.append([extract_relation_features(arg1,arg2,entities,fulltext),'scope_pos'])
		# TODO: try all other combinations to generate negative training data.
		instances.append([extract_relation_features(arg2,arg1,entities,fulltext),'scope_neg'])
	return instances
def extract_relation_features(arg1,arg2,entities,fulltext):
	"""
	the following features should be extracted:
		✓ Arg1_entity:AAUTHOR
		✓ Arg2_entity:REFSCOPE
		✓ ConcEnt: AAUTHORREFSCOPE
		✓ Thuc.=True (head_arg1)
		✓ 1.8=True (head_arg2)
		WordsBtw:0
		EntBtw:0 
		word_before_arg1
		word_after_arg1
		word_before_arg2
		word_after_arg2
	"""
	def count_words_between_arguments(arg1_entity,arg2_entity,fulltext):
		"""
		Count the number of words between arg1 and arg2
		"""
		# this for properties going left to right (AWORK -> REFSCOPE)
		if(arg1_entity["offset_end"] < arg2_entity["offset_start"]):
			span_ends = int(arg1_entity["offset_end"])
			span_begins = int(arg2_entity["offset_start"])
		# this for properties going right to left (REFSCOPE -> AWORK)
		else:
			span_begins = int(arg1_entity["offset_start"])
			span_ends = int(arg2_entity["offset_end"])
		span_between = fulltext[span_ends:span_begins]
		return len(span_between.split())
	def count_entities_between_arguments(arg1_entity,arg2_entity,entities):
		"""
		same as functione above, but iterate through entities and retain only those
		within the boundaries then count
		"""
		pass
	def get_word_before_after(argument):
		pass
	features = {}
	features["arg1_entity"] = entities[arg1]["entity_type"]
	features["arg2_entity"] = entities[arg2]["entity_type"]
	features["conc_entities"] = "%s%s"%(entities[arg1]["entity_type"]
								,entities[arg2]["entity_type"])
	features["arg1_head"] = entities[arg1]["surface"]
	features["arg2_head"] = entities[arg2]["surface"]
	features["words_btw"] = count_words_between_arguments(entities[arg1],entities[arg2],fulltext)
	#features["entities_btw"] = count_entities_between_arguments(entities[arg1],entities[arg2])
	return features