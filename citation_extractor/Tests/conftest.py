# -*- coding: utf-8 -*-
# author: Matteo Romanello, matteo.romanello@gmail.com

import pytest
import pprint
import logging
import pkg_resources
from pytest import fixture
from citation_extractor.ned import KnowledgeBase
from citation_extractor.ned import CitationMatcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@fixture
def knowledge_base():
	"""
	Initialises and returns a HuCit KnowledgeBase (old version).
	"""
	dir = pkg_resources.resource_filename('citation_extractor','data/kb/')
	files = ["%s%s"%(dir,file) for file in pkg_resources.resource_listdir('citation_extractor','data/kb/')]
	kb = KnowledgeBase(files,"turtle")
	logger.info("The KnowledgeBase %s was initialised"%kb)
	return kb
@fixture
def citation_matcher(knowledge_base):
	"""
	Initialises and returns a CitationMatcher.
	"""
	return CitationMatcher(knowledge_base)
@fixture
def new_knowledge_base():
	"""
	Initialises and returns a HuCit KnowledgeBase (new version, standalone package).
	"""
	dir = pkg_resources.resource_filename('citation_extractor','data/kb/')
	files = ["%s%s"%(dir,file) for file in pkg_resources.resource_listdir('citation_extractor','data/kb/')]
	kb = KnowledgeBase(files,"turtle")
	logger.info("The KnowledgeBase %s was initialised"%kb)
	return kb
@fixture