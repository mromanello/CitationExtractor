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
    #testset_gold_df.to_pickle("citation_extractor/data/pickles/test_gold_dataframe.pkl")

    logger.info(tabulate(testset_gold_df.head(20)[["type", "surface", "scope", "urn"]]))

    # TODO: replace hard-coded path with pkg_resources
    kb_cfg_file = pkg_resources.resource_filename('knowledge_base','config/virtuoso.ini')
    kb = KnowledgeBase(kb_cfg_file)
    """
    kb_data = {
            "author_names": kb.author_names
            , "author_abbreviations": kb.author_abbreviations
            , "work_titles": kb.work_titles
            , "work_abbreviations": kb.work_abbreviations
            }

    with codecs.open("citation_extractor/data/pickles/kb_data.pkl","wb") as pickle_file:
        pickle.dump(kb_data, pickle_file)
    """
    
    cms = {}

    ##############################
    # Test 1: default parameters #
    ##############################

    cms["cm1"] = CitationMatcher(kb, fuzzy_matching_entities=False, fuzzy_matching_relations=False)

    """
    ##############################
    # Test 2: best parameters    #
    ##############################
    cms["cm2"] = CitationMatcher(kb
                        , fuzzy_matching_entities=True
                        , fuzzy_matching_relations=True
                        , min_distance_entities=4
                        , max_distance_entities=7
                        , distance_relations=4)

    #####################################
    # Test 3: alternative parameters    #
    #####################################
    
    cms["cm3"] = CitationMatcher(kb
                        , fuzzy_matching_entities=True
                        , fuzzy_matching_relations=False
                        , min_distance_entities=4
                        , max_distance_entities=7)
    """
    
    comp_evaluation = []
    comp_accuracy_by_type = []
    
    # for each citation matcher disambiguate the records in the test set,
    # carry out the evaluation and store the results in two temporary lists (to be
    # transformed later on into two dataframes)
    for key in sorted(cms.keys()):
        cm = cms[key]
        testset_target_df = testset_gold_df.copy()
        
        # run the parallel processing of records
        results = parmap.map(_pprocess, ((n, row[0], row[1]) for n, row in enumerate(testset_target_df.iterrows())), cm)
        
        # collect the results and update the dataframe
        for instance_id, urn in results:
            testset_target_df.loc[instance_id]["urn_clean"] = urn

        # save pickle for later 
        #testset_target_df.to_pickle("citation_extractor/data/pickles/test_target_dataframe_%s.pkl" % key)

        scores, accuracy_by_type, error_types, errors = evaluate_ned(testset_gold_df, ann_dir, testset_target_df, strict=True)

        # aggregate and format the evaluation measure already with percentages        
        scores  = {score_key : "%.2f%%" % (scores[score_key]*100) for score_key in scores}
        scores["id"] = key
        comp_evaluation.append(scores)

        # aggregate and format the accuracy by type already with percentages  
        accuracy = {type_key : "%.2f%%" % (accuracy_by_type[type_key]*100) for type_key in accuracy_by_type}
        accuracy["id"] = key
        comp_accuracy_by_type.append(accuracy)

    comp_evaluation_df = pd.DataFrame(comp_evaluation, index=[score["id"] for score in comp_evaluation])
    del comp_evaluation_df["id"] # we don't need it twice (already in the index)

    comp_accuracy_by_type_df = pd.DataFrame(comp_accuracy_by_type, index=[accuracy["id"] for accuracy in comp_accuracy_by_type])
    del comp_accuracy_by_type_df["id"] # we don't need it twice (already in the index)

    logger.info("\n" + tabulate(comp_evaluation_df, headers=comp_evaluation_df.columns))
    logger.info("\n" + tabulate(comp_accuracy_by_type_df, headers=comp_accuracy_by_type_df.columns))
    logger.info("\n" + "\n".join(["%s: %s" % (key, cms[key].settings) for key in cms]))
