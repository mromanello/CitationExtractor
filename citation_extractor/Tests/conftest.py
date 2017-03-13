# -*- coding: utf-8 -*-
# author: Matteo Romanello, matteo.romanello@gmail.com

import pytest
import pprint
import logging
import pkg_resources
from pytest import fixture
from citation_extractor import pipeline
from citation_extractor.settings import crf
from citation_extractor.core import citation_extractor
from citation_extractor.ned import KnowledgeBase
from citation_extractor.ned import CitationMatcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@fixture
def crf_citation_extractor(tmpdir):
	"""
	Initialise a citation extractor with CRF model trained on the
	goldset of the APh corpus.
	"""
	crf.TEMP_DIR = str(tmpdir.mkdir('tmp'))+"/"
	crf.OUTPUT_DIR = str(tmpdir.mkdir('out'))+"/"
	crf.LOG_FILE = "%s/extractor.log"%crf.TEMP_DIR
	return pipeline.get_extractor(crf)
@fixture
def old_knowledge_base():
	"""
	Initialises and returns a HuCit KnowledgeBase (old version).
	"""
	dir = pkg_resources.resource_filename('citation_extractor','data/kb/')
	files = ["%s%s"%(dir,file) for file in pkg_resources.resource_listdir('citation_extractor','data/kb/')]
	kb = KnowledgeBase(files,"turtle")
	logger.info("The KnowledgeBase %s was initialised"%kb)
	return kb
@fixture
def citation_matcher_legacy(old_knowledge_base):
	"""
	Initialises and returns a CitationMatcher.
	"""
	return CitationMatcher(knowledge_base)
@pytest.mark.skip
@fixture
def new_knowledge_base():
	"""
	Initialises and returns a HuCit KnowledgeBase (new version, standalone package).
	"""
	pass
