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

@fixture(scope="session")
def crf_citation_extractor(tmpdir_factory):
	"""
	Initialise a citation extractor with CRF model trained on the
	goldset of the APh corpus.
	"""
	crf.TEMP_DIR = str(tmpdir_factory.mktemp('tmp'))+"/"
	crf.OUTPUT_DIR = str(tmpdir_factory.mktemp('out'))+"/"
	crf.LOG_FILE = crf.TEMP_DIR.join("extractor.log")
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
@fixture(scope="session")
def postaggers():
	"""
	A dictionary where keys are language codes and values are the corresponding PoS tagger.
	"""
	abbreviations = pkg_resources.resource_filename('citation_extractor'
													, 'data/aph_corpus/extra/abbreviations.txt')
	return pipeline.get_taggers(abbrev_file = abbreviations)
@fixture
def aph_title():
	return """Problemi di colometria Eschilo, « Prometeo » 
				(vv. 526-44 ; 887-900) « Agamennone » (vv. 104-21)"""

