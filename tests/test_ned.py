# -*- coding: utf-8 -*-
# author: Matteo Romanello, matteo.romanello@gmail.com

"""Tests for the module `citation_extractor.ned.matchers`."""

import pdb
import pytest
import logging
import pickle
import pandas as pd
from citation_extractor.Utils.IO import load_brat_data
from citation_extractor.ned.ml_matchers import FeatureExtractor

logger = logging.getLogger(__name__)


@pytest.mark.skip
def test_pickle_kb(knowledge_base):
    """Tests whether instances of `KnowledgeBase` can be pickled."""
    pickled_kb = pickle.dumps(knowledge_base)
    unpickled_kb = pickle.loads(pickled_kb)
    logger.info(
        "The KnowledgeBase contains %i author names" % len(
            unpickled_kb.author_names
        )
    )


@pytest.mark.skip
def test_pickle_citation_matcher(citation_matcher):
    """Test whether instances of `CitationMatcher` can be pickled."""
    pickled_citation_matcher = pickle.dumps(citation_matcher)
    unpickled_citation_matcher = pickle.loads(pickled_citation_matcher)


def test_instantiate_ml_citation_matcher(
        knowledge_base,
        aph_gold_ann_files,
        crf_citation_extractor,
        postaggers,
        aph_titles
        ):
    """Create an instance of MLCitationMatcher."""

    train_df_data = load_brat_data(
        crf_citation_extractor,
        knowledge_base,
        postaggers,
        aph_gold_ann_files,
        aph_titles
    )

    fe = FeatureExtractor(knowledge_base, train_df_data)
    logger.info(fe)
