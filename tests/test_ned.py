# -*- coding: utf-8 -*-
# author: Matteo Romanello, matteo.romanello@gmail.com

"""Tests for the module `citation_extractor.ned.matchers`."""

import pdb
import pytest
import logging
import pickle
import pandas as pd
from citation_extractor.Utils.IO import load_brat_data
from citation_extractor.ned.features import FeatureExtractor
from citation_extractor.ned.ml import SVMRank
import random

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


@pytest.mark.skip
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

    """
    train_df_data = pd.read_pickle(
        "citation_extractor/data/pickles/aph_gold_df.pkl"
    )
    """

    fe = FeatureExtractor(knowledge_base, train_df_data)
    logger.info(fe)


def test_svm_rank():
    lowb, upperb, shift = 0, 1, 1

    # Generate two groups with 3 points each
    X = [
        dict(x=random.uniform(lowb, upperb), y=random.uniform(lowb, upperb)),
        dict(x=random.uniform(lowb, upperb), y=random.uniform(lowb, upperb)),
        dict(x=random.uniform(lowb, upperb) + shift, y=random.uniform(lowb, upperb) + shift),  # true one
        dict(x=random.uniform(lowb, upperb), y=random.uniform(lowb, upperb)),
        dict(x=random.uniform(lowb, upperb), y=random.uniform(lowb, upperb)),
        dict(x=random.uniform(lowb, upperb) + shift, y=random.uniform(lowb, upperb) + shift)  # true one
    ]
    print(X)
    y = [
        0,
        0,
        1,
        0,
        0,
        1
    ]
    print(y)
    groups = [
        0,
        0,
        0,
        1,
        1,
        1
    ]
    print(groups)

    # Fit the ranker
    ranker = SVMRank()
    ranker.fit(X=X, y=y, groups=groups)

    # Generate a group of three points, the second (index=1) is the true one
    candidates = [
        dict(x=random.uniform(lowb, upperb), y=random.uniform(lowb, upperb)),
        dict(x=random.uniform(lowb, upperb) + shift, y=random.uniform(lowb, upperb) + shift),  # true one
        dict(x=random.uniform(lowb, upperb), y=random.uniform(lowb, upperb))
    ]

    # Predict
    ranked_candidates, scores = ranker.predict(candidates)
    winner_index = ranked_candidates[0]

    assert winner_index == 1
