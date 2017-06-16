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
        #traceback.print_stack()
        disambiguation_result = None
        logger.debug("%i - %s - %s" % (n, id, disambiguation_result))
        return (id, disambiguation_result)

def test_eval_ned_baseline(aph_testset_dataframe, aph_test_ann_files):

    ann_dir, ann_files = aph_test_ann_files
    testset_gold_df = aph_testset_dataframe

    # TODO remove it once done with testing
    with codecs.open(pkg_resources.resource_filename("citation_extractor", "data/pickles/kb_data.pkl"),"rb") as pickle_file:
        kb_data = pickle.load(pickle_file)

    logger.info(tabulate(testset_gold_df.head(20)[["type", "surface", "scope", "urn"]]))

    # TODO: replace hard-coded path with pkg_resources
    kb = KnowledgeBase("/Users/rromanello/Documents/ClassicsCitations/hucit_kb/knowledge_base/config/virtuoso.ini")
    
    cms = {}

    ##############################
    # Test 1: default parameters #
    ##############################

    cms["cm1"] = CitationMatcher(kb, fuzzy_matching_entities=False, fuzzy_matching_relations=False, **kb_data)

    ##############################
    # Test 2: best parameters    #
    ##############################
    cms["cm2"] = CitationMatcher(kb
                        , fuzzy_matching_entities=True
                        , fuzzy_matching_relations=True
                        , min_distance_entities=4
                        , max_distance_entities=7
                        , distance_relations=4
                        , **kb_data)

    #####################################
    # Test 3: alternative parameters    #
    #####################################
    cms["cm3"] = CitationMatcher(kb
                        , fuzzy_matching_entities=True
                        , fuzzy_matching_relations=False
                        , min_distance_entities=4
                        , max_distance_entities=7
                        , **kb_data)
    
    comp_evaluation = []
    comp_accuracy_by_type = []
    
    for key in sorted(cms.keys()):
        cm = cms[key]
        testset_target_df = testset_gold_df.copy()
        results = parmap.map(_pprocess, ((n, row[0], row[1]) for n, row in enumerate(testset_target_df.iterrows())), cm)
        
        for instance_id, urn in results:
            testset_target_df.loc[instance_id]["urn_clean"] = urn

        with codecs.open("citation_extractor/data/pickles/test_target_dataframe_%s.pkl" % key,"wb") as pickle_file:
            pickle.dump(testset_target_df, pickle_file)

        # aggregate and format the data already with percentages
        scores, accuracy_by_type, error_types, errors = evaluate_ned(testset_gold_df, ann_dir, testset_target_df, strict=True)
        
        scores  = {score_key : "%.2f%%" % (scores[score_key]*100) for score_key in scores}
        scores["id"] = key
        comp_evaluation.append(scores)

        accuracy = {type_key : "%.2f%%" % (accuracy_by_type[type_key]*100) for type_key in accuracy_by_type}
        accuracy["id"] = key
        comp_accuracy_by_type.append(accuracy)

    comp_evaluation_df = pd.DataFrame(comp_evaluation, index=[score["id"] for score in comp_evaluation])
    del comp_evaluation_df["id"] # we don't need it twice

    comp_accuracy_by_type_df = pd.DataFrame(comp_accuracy_by_type, index=[accuracy["id"] for accuracy in comp_accuracy_by_type])
    del comp_accuracy_by_type_df["id"] # we don't need it twice

    logger.info("\n" + tabulate(comp_evaluation_df, headers=comp_evaluation_df.columns))
    logger.info("\n" + tabulate(comp_accuracy_by_type_df, headers=comp_accuracy_by_type_df.columns))
    logger.info("\n" + "\n".join(["%s: %s" % (key, cms[key].settings) for key in cms]))
    pdb.set_trace()
