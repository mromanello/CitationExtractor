# -*- coding: utf-8 -*-
# author: Matteo Romanello, matteo.romanello@gmail.com

"""TODO."""

import pytest
import pprint
import logging
import pkg_resources
from pytest import fixture
import pandas as pd
from citation_extractor import pipeline
from citation_extractor.Utils.IO import load_brat_data
from citation_extractor.settings import crf, svm, maxent, crfsuite
from citation_extractor.core import citation_extractor
from citation_extractor.ned import KnowledgeBase
from citation_extractor.ned import CitationMatcher
from knowledge_base import KnowledgeBase as KnowledgeBaseNew

logging.basicConfig(level=logging.INFO)
#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@fixture(scope="session")
def crf_citation_extractor(tmpdir_factory):
    """Initialise a citation extractor trained with CRF on APh corpus."""
    crf.TEMP_DIR = str(tmpdir_factory.mktemp('tmp'))+"/"
    crf.OUTPUT_DIR = str(tmpdir_factory.mktemp('out'))+"/"
    crf.LOG_FILE = crf.TEMP_DIR.join("extractor.log")
    return pipeline.get_extractor(crf)


@fixture(scope="session")
def svm_citation_extractor():
    """Initialise a citation extractor trained with SVM on APh corpus."""
    return pipeline.get_extractor(svm)


@fixture(scope="session")
def maxent_citation_extractor():
    """Initialise a citation extractor trained with MaxEnt on APh corpus."""
    return pipeline.get_extractor(maxent)


@fixture(scope="session")
def crfsuite_citation_extractor():
    """Initialise a citation extractor trained with CRFSuite on APh corpus."""
    return pipeline.get_extractor(crfsuite)


@fixture(scope="session")
def processing_directories(tmpdir_factory):
    return {
        'input' : pkg_resources.resource_filename('citation_extractor','data/aph_corpus/devset/orig/')
        , 'iob' : str(tmpdir_factory.mktemp('iob'))+"/"
        , 'txt' : str(tmpdir_factory.mktemp('txt'))+"/"
        , 'iob_ne' : str(tmpdir_factory.mktemp('iob_ne'))+"/"
        , 'ann' :  str(tmpdir_factory.mktemp('ann'))+"/"
    }

@fixture(scope="session")
def aph_gold_ann_files():
    ann_dir = pkg_resources.resource_filename('citation_extractor', 'data/aph_corpus/goldset/ann')
    ann_files = pkg_resources.resource_listdir('citation_extractor', 'data/aph_corpus/goldset/ann')
    return ann_dir, ann_files

@fixture(scope="session")
def aph_test_ann_files():
    ann_dir = pkg_resources.resource_filename('citation_extractor', 'data/aph_corpus/testset/ann')
    ann_files = pkg_resources.resource_listdir('citation_extractor', 'data/aph_corpus/testset/ann')
    return ann_dir, ann_files

@fixture(scope="session")
def aph_testset_dataframe(crf_citation_extractor, knowledge_base, postaggers, aph_test_ann_files, aph_titles):
    """
    A pandas DataFrame containing the APh test-set data: may be useful to perform evaluation.
    """
    logger.info("Loading test-set data (%i documents) from %s" % (len(aph_test_ann_files[1]), aph_test_ann_files[0]))
    dataframe = load_brat_data(crf_citation_extractor, knowledge_base, postaggers, aph_test_ann_files, aph_titles)
    assert dataframe is not None and type(dataframe)==type(pd.DataFrame()) and dataframe.shape[0]>0
    return dataframe

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
    Initialises and returns a CitationMatcher (legacy).
    """
    return CitationMatcher(knowledge_base)

@fixture(scope="session")
def citation_matcher(knowledge_base):
    """
    Initialises and returns a CitationMatcher.
    """
    return CitationMatcher(knowledge_base)

@fixture(scope="session")
def knowledge_base():
    """
    Initialises and returns a HuCit KnowledgeBase (new version, standalone package).
    """
    try:
        config_file = pkg_resources.resource_filename('knowledge_base','config/virtuoso.ini')
        kb = KnowledgeBaseNew(config_file)
        kb.get_authors()[0]
        return kb
    except Exception, e:
        config_file = pkg_resources.resource_filename('knowledge_base','config/inmemory.ini')
        return KnowledgeBaseNew(config_file)

@fixture(scope="session")
def postaggers():
    """
    A dictionary where keys are language codes and values are the corresponding PoS tagger.
    """
    abbreviations = pkg_resources.resource_filename('citation_extractor'
                                                    , 'data/aph_corpus/extra/abbreviations.txt')
    return pipeline.get_taggers(abbrev_file = abbreviations)

@fixture(scope="session")
def aph_titles():
    titles_dir = pkg_resources.resource_filename('citation_extractor', 'data/aph_corpus/titles.csv')
    titles = pd.read_csv(titles_dir).set_index('id')[['title']]
    return titles
