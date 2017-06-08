# -*- coding: utf-8 -*-
# author: Matteo Romanello, matteo.romanello@gmail.com

import pdb
import pytest
from pytest import fixture
import pickle
import logging
import citation_extractor
from citation_extractor.eval import evaluate_ned

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_eval_ned_baseline(aph_testset_dataframe, aph_test_ann_files):
	"""
	logic:
		- read the test-set data (GT) into a `DataFrame`
		- make a copy of the dataframe, our target-data
		- invoke the `citation_matcher` for each row in the `DataFrame` and store the predicted URN
		- pass the two 
	"""
	ann_dir, ann_files = aph_test_ann_files
	testset_gold_df = aph_testset_dataframe
	testset_target_df = testset_gold_df.copy()

	evaluate_ned(testset_gold_df, ann_dir, testset_target_df, strict=False)
	evaluate_ned(testset_gold_df, ann_dir, testset_target_df, strict=True)
	
	logger.info(testset_gold_df.head())
	#pdb.set_trace()

	# for each row in the `df` invoke the CitationMatcher
