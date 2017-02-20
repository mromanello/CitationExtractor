# -*- coding: utf-8 -*-
# author: Matteo Romanello, matteo.romanello@gmail.com

import pprint
import logging
import pkg_resources
import pickle
from pytest import fixture
from citation_extractor.ned import KnowledgeBase
from citation_extractor.ned import CitationMatcher

#logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@fixture
def knowledge_base():
	"""
	Initialises and returns a HuCit KnowledgeBase.
	"""
	dir = pkg_resources.resource_filename('citation_extractor','data/kb/')
	files = ["%s%s"%(dir,file) for file in pkg_resources.resource_listdir('citation_extractor','data/kb/')]
	kb = KnowledgeBase(files,"turtle")
	logger.info("The KnowledgeBase %s was initialised"%kb)
	return kb


def test_pickle_kb(knowledge_base):
	"""
	Tests whether instances of `KnowledgeBase` can be pickled.
	"""
	pickled_kb = pickle.dumps(knowledge_base)
	unpickled_kb = pickle.loads(pickled_kb)
	logger.info("The KnowledgeBase contains %i author names"%len(unpickled_kb.author_names))


def test_pickle_citation_matcher(knowledge_base):
	"""
	Tests whether instances of `CitationMatcher` (and contained objects) can be pickled.
	"""
	cm = CitationMatcher(knowledge_base)
	pickled_citation_matcher = pickle.dumps(cm)
	unpickled_citation_matcher = pickle.loads(pickled_citation_matcher)


def test_dummy(knowledge_base):
	"""
	Dummy test..
	"""
	ml_cm = MLCitationMatcher()
	ml_cm.disambiguate('Hello', '1.2.3')
