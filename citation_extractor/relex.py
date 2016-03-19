# -*- coding: utf-8 -*-
# author: Matteo Romanello, matteo.romanello@gmail.com

import logging
import codecs
import itertools
from citation_extractor.pipeline import read_ann_file_new

global logger
logger = logging.getLogger(__name__)

def prepare_for_training(doc_id, basedir):
	"""

	Generate tagged instances for the task of relation extraction from a given document.

	If the document contains at least 1 relation:
		- try all possible pairs (arg1,arg2) from the entities contained in the document
		- if the pair is an actual relation, then label the feature set as a positive instances
		- otherwise label it as a negative instance.
		- this approach generates way more negative than positive examples
			- when training the classifier select so many negative examples as the positive examples (randomly selected) 

	The result returned by this function is a list of lists that looks like this:

	result = [
		[
			[
				"arg1_entity":"AAUTHOR"
				,"arg2_entity":"REFSCOPE"
				,"concent":"AAUTHORREFSCOPE"
				...
			]
			,'scope_pos'
		]
		,[
			[
				"arg1_entity":"REFSCOPE"
				,"arg2_entity":"AAUTHOR"
				,"concent":"REFSCOPEAAUTHOR"
				...
			]
			,'scope_neg'
		]
	]
	"""
	instances = []
	entities, relations, annotations = read_ann_file_new(doc_id, basedir)
	fulltext = codecs.open("%s%s-doc-1.txt"%(basedir,doc_id),'r','utf-8').read()
	logger.info("Preparing for training %s%s (\"%s\")"%(basedir,doc_id,fulltext))
	entity_ids = [entities[entity]["ann_id"] for entity in entities]
	relation_list = [relations[rel_id]["arguments"] for rel_id in relations]
	logger.debug(entity_ids)
	if(len(relations.keys())>0):
		for relation in itertools.combinations(entity_ids,2):
			is_positive_instance = relation in relation_list
			if(is_positive_instance):
				class_label = "scope_pos"
			else:
				class_label = "scope_neg"
			features = extract_relation_features(relation[0],relation[1],entities,fulltext)
			if(is_positive_instance):
				instances.append([features,class_label])
			else:
				instances.append([features,class_label])
			logger.info("[%s] Features for %s %s (%s-%s): %s"%(class_label,entities[relation[0]]["surface"],entities[relation[1]]["surface"],relation[0],relation[1],features))
			# now reverse the arguments...
			is_positive_instance = (relation[1],relation[0]) in relation_list
			if(is_positive_instance):
				class_label = "scope_pos"
			else:
				class_label = "scope_neg"
			features = extract_relation_features(relation[1],relation[0],entities,fulltext)
			if(is_positive_instance):
				instances.append([features,class_label])
			else:
				instances.append([features,class_label])
			logger.info("[%s] Features for %s %s (%s-%s): %s"%(class_label,entities[relation[1]]["surface"],entities[relation[0]]["surface"],relation[1],relation[0],features))
	return instances
def extract_relation_features(arg1,arg2,entities,fulltext):
	"""
	the following features should be extracted:
		✓ Arg1_entity:AAUTHOR
		✓ Arg2_entity:REFSCOPE
		✓ ConcEnt: AAUTHORREFSCOPE
		✓ Thuc.=True (head_arg1)
		✓ 1.8=True (head_arg2)
		✓ WordsBtw:0
		✓ EntBtw:0 
		✓ word_before_arg1
		✓ word_after_arg1
		✓ word_before_arg2
		✓ word_after_arg2
		? perhaps feature with concatenated entities between arguments ?
	"""
	def count_words_between_arguments(arg1_entity,arg2_entity,fulltext):
		"""
		Count the number of words between arg1 and arg2
		"""
		# this for properties going left to right (AWORK -> REFSCOPE)
		if(arg1_entity["offset_end"] < arg2_entity["offset_start"]):
			span_begins = int(arg1_entity["offset_end"])
			span_ends = int(arg2_entity["offset_start"])
		# this for properties going right to left (REFSCOPE -> AWORK)
		else:
			span_ends = int(arg1_entity["offset_start"])
			span_begins = int(arg2_entity["offset_end"])
		span_between = fulltext[span_begins:span_ends]
		logger.debug("Counting words between %s-%s"%(arg1_entity["ann_id"],arg2_entity["ann_id"]))
		logger.debug("In the text span \"%s\" there are %i words"%(span_between,len(span_between.split())))
		return len(span_between.split())
	def count_entities_between_arguments(arg1_entity,arg2_entity,entities):
		"""
		same as functione above, but iterate through entities and retain only those
		within the boundaries then count them
		"""
		# this for properties going left to right (AWORK -> REFSCOPE)
		if(arg1_entity["offset_end"] < arg2_entity["offset_start"]):
			span_begins = int(arg1_entity["offset_end"])
			span_ends = int(arg2_entity["offset_start"])
		# this for properties going right to left (REFSCOPE -> AWORK)
		else:
			span_ends = int(arg1_entity["offset_start"])
			span_begins = int(arg2_entity["offset_end"])
		logger.debug("Counting entities between arguments %s-%s"%(arg1_entity["ann_id"],arg2_entity["ann_id"]))
		logger.debug("Count entities within the offsets %s-%s"%(span_begins,span_ends))
		entities_between = [entity for entity in entities 
								if int(entities[entity]["offset_start"]) > span_begins and int(entities[entity]["offset_end"]) < span_ends]
		logger.debug("Found %i entities in between (%s)"%(len(entities_between),entities_between))
		return len(entities_between)
	def get_word_before_after(argument_entity,fulltext):
		"""
		Get the word at the left (preceding) and at the right (following) of a relation argument.
		"""
		word_before = fulltext[:int(argument_entity["offset_start"])].split()[-1:len(fulltext[:int(argument_entity["offset_start"])].split())]
		word_after = fulltext[int(argument_entity["offset_end"]):].split()[0:1]
		if(word_before==[]):
			word_before = "None"
		else:
			word_before = word_before[0]
		if(word_after==[]):
			word_after = "None"
		else:
			word_after = word_after[0]
		logger.debug("The word before \"%s\" is \"%s\" and the one after is \"%s\""%(argument_entity["surface"],word_before,word_after))
		return (word_before,word_after)
	features = {}
	features["arg1_entity"] = entities[arg1]["entity_type"]
	features["arg2_entity"] = entities[arg2]["entity_type"]
	features["conc_entities"] = "%s%s"%(entities[arg1]["entity_type"]
								,entities[arg2]["entity_type"])
	features["arg1_head"] = entities[arg1]["surface"]
	features["arg2_head"] = entities[arg2]["surface"]
	features["n_words_btw"] = count_words_between_arguments(entities[arg1],entities[arg2],fulltext)
	features["n_entities_btw"] = count_entities_between_arguments(entities[arg1],entities[arg2],entities)
	features["word_before_arg1"],features["word_after_arg1"] = get_word_before_after(entities[arg1],fulltext)   
	features["word_before_arg2"],features["word_after_arg2"] = get_word_before_after(entities[arg2],fulltext)
	return features
class relation_extractor(object):
	"""docstring for relation_extractor"""
	def __init__(self, classifier, training_directories):
		pass
	def extract(self,entities,fulltext):
		pass
