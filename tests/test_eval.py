# -*- coding: utf-8 -*-
# author: Matteo Romanello, matteo.romanello@gmail.com

import pdb
import pandas as pd
import pkg_resources
import pytest
from pytest import fixture
import pickle
import logging
import codecs
import parmap
import traceback
import multiprocessing as mp
import citation_extractor
from tabulate import tabulate
from citation_extractor.eval import evaluate_ned
from citation_extractor.ned import CitationMatcher
from knowledge_base import KnowledgeBase


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _pprocess(datum, citation_matcher):
    """
    Function to use for parallel processing.
    """
    n, id, instance = datum
    try:
        disambiguation_result = citation_matcher.disambiguate(instance['surface'], instance["type"], instance["scope"])
        logger.debug("%i - %s - %s" % (n, id, disambiguation_result.urn))
        return (id, str(disambiguation_result.urn))
    except Exception as e:
        logger.error("Disambiguation of %s raised the following error: %s" % (id, e))
        traceback.print_stack()
        disambiguation_result = None
        logger.debug("%i - %s - %s" % (n, id, disambiguation_result))
        return (id, disambiguation_result)

def test_eval_ned_baseline(aph_testset_dataframe, aph_test_ann_files):

    ann_dir, ann_files = aph_test_ann_files
    testset_gold_df = aph_testset_dataframe
    #testset_target_df = testset_gold_df.copy()

    # TODO remove it once done with testing
    with codecs.open(pkg_resources.resource_filename("citation_extractor", "data/pickles/kb_data.pkl"),"rb") as pickle_file:
        kb_data = pickle.load(pickle_file)

    # TODO remove it once done with testing
    with codecs.open(pkg_resources.resource_filename("citation_extractor", "data/pickles/testset_target_df.pkl"),"rb") as pickle_file:
        testset_target_df = pd.read_pickle(pickle_file)        
    
    logger.info(tabulate(testset_gold_df.head(20)[["type", "surface", "scope", "urn"]]))
    logger.info(tabulate(testset_target_df.head(20)[["type", "surface", "scope", "urn"]]))

    # TODO: replace hard-coded path with pkg_resources
    kb = KnowledgeBase("/Users/rromanello/Documents/ClassicsCitations/hucit_kb/knowledge_base/config/virtuoso.ini")
    
    ##############################
    # Test 1: default parameters #
    ##############################

    
    cm = CitationMatcher(kb, fuzzy_matching_entities=True, fuzzy_matching_relations=True, **kb_data)

    results = parmap.map(_pprocess, ((n, row[0], row[1]) for n, row in enumerate(testset_target_df.iterrows())), cm)
    
    for instance_id, urn in results:
        testset_target_df.loc[instance_id]["urn_clean"] = urn

    print cm.settings

    #evaluate_ned(testset_gold_df, ann_dir, testset_target_df, strict=False)
    evaluate_ned(testset_gold_df, ann_dir, testset_target_df, strict=True)
    

    ##############################
    # Test 2: best parameters    #
    ##############################
    cm = CitationMatcher(kb
                        , fuzzy_matching_entities=True
                        , fuzzy_matching_relations=True
                        , min_distance_entities=4
                        , max_distance_entities=7
                        , distance_relations=4
                        , **kb_data)

    results = parmap.map(_pprocess, ((n, row[0], row[1]) for n, row in enumerate(testset_target_df.iterrows())), cm)
    
    for instance_id, urn in results:
        testset_target_df.loc[instance_id]["urn_clean"] = urn

    print cm.settings

    #evaluate_ned(testset_gold_df, ann_dir, testset_target_df, strict=False)
    evaluate_ned(testset_gold_df, ann_dir, testset_target_df, strict=True)
