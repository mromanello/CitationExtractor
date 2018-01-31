# -*- coding: utf-8 -*-
# author: Matteo Romanello, matteo.romanello@gmail.com

"""Tests for the module `citation_extractor.ned.matchers`."""

import pdb
import pytest
import logging
import pickle
import pandas as pd
from citation_extractor.pipeline import NIL_URN
from citation_extractor.Utils.IO import load_brat_data
from citation_extractor.ned.features import FeatureExtractor
from citation_extractor.ned.ml import LinearSVMRank
import random

logger = logging.getLogger(__name__)


# TODO: move this test somewhere else
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


# When finished testing, transform into a fixture
# and move to conftest.py
def test_instantiate_featureextractor(
        knowledge_base,
        aph_gold_ann_files,
        crf_citation_extractor,
        postaggers,
        aph_titles,
        aph_testset_dataframe
):
    """Create an instance of MLCitationMatcher."""
    train_df_data = load_brat_data(  # TODO create a fixture out of thi  s
        crf_citation_extractor,
        knowledge_base,
        postaggers,
        aph_gold_ann_files,
        aph_titles
    )

    # initialise a FeatureExtractor
    fe = FeatureExtractor(knowledge_base, train_df_data)
    logger.info(fe)

    for id_row, row in aph_testset_dataframe.iterrows():
        # NB: should be called on the candidates!
        if row["urn"] != NIL_URN:
            fv = fe.extract(
                row["surface_norm"],
                row["scope"],
                row["type"],
                row["doc_title_mentions"],
                row["doc_title_norm"],
                row["doc_text"],
                row["other_mentions"],
                row["urn_clean"]
            )
            # TODO: call `fe.extract_nil`
            logger.debug(fv)
        else:
            logger.debug("Skipped {}".format(row))

    aph_testset_dataframe.to_pickle(
        'citation_extractor/data/pickles/aph_test_df.pkl'
    )

    # pickle probability dataframes
    fe._prior_prob.to_pickle('citation_extractor/data/pickles/prior_prob.pkl')
    fe._em_prob.to_pickle('citation_extractor/data/pickles/em_prob.pkl')
    fe._me_prob.to_pickle('citation_extractor/data/pickles/me_prob.pkl')

    # serialize normalized authors
    with open(
        'citation_extractor/data/pickles/kb_norm_authors.pkl',
        'wb'
    ) as f:
        pickle.dump(fe._kb_norm_authors, f)

    # serialize normalized works
    with open(
        'citation_extractor/data/pickles/kb_norm_works.pkl',
        'wb'
    ) as f:
        pickle.dump(fe._kb_norm_works, f)

    # serialize the FeatureExtractor
    with open(
        'citation_extractor/data/pickles/ml_feature_extractor.pkl',
        'wb'
    ) as f:
        pickle.dump(fe, f)


def test_instantiate_featureextractor_quick():
    """Instantiate an instance of FeatureExtractor from pickled data."""
    prior_prob = pd.read_pickle(
        'citation_extractor/data/pickles/prior_prob.pkl'
    )

    em_prob = pd.read_pickle(
        'citation_extractor/data/pickles/em_prob.pkl'
    )

    me_prob = pd.read_pickle(
        'citation_extractor/data/pickles/me_prob.pkl'
    )

    fname = 'citation_extractor/data/pickles/kb_norm_authors.pkl'
    with open(fname, 'rb') as f:
        kb_norm_authors = pickle.load(f)

    with open('citation_extractor/data/pickles/kb_norm_works.pkl', 'rb') as f:
        kb_norm_works = pickle.load(f)

    fe = FeatureExtractor(
        kb_norm_authors=kb_norm_authors,
        kb_norm_works=kb_norm_works,
        prior_prob=prior_prob,
        mention_entity_prob=me_prob,
        entity_mention_prob=em_prob
    )

    logger.info(fe)

    assert fe is not None
    assert fe._prior_prob is not None
    assert fe._em_prob is not None
    assert fe._me_prob is not None

    test_df_data = pd.read_pickle(
        'citation_extractor/data/pickles/aph_test_df.pkl'
    )
    logger.debug(test_df_data.info())

    for id_row, row in test_df_data.iterrows():
        # TODO: should be called on the candidates!
        if row["urn"] != NIL_URN:
            fv = fe.extract(
                row["surface_norm"],
                row["scope"],
                row["type"],
                row["doc_title_mentions"],
                row["doc_title_norm"],
                row["doc_text"],
                row["other_mentions"],
                row["urn_clean"]
            )
            # TODO: call `fe.extract_nil`
            logger.debug(fv)
        else:
            logger.debug("Skipped {}".format(row))


def test_svm_rank():
    lowb, upperb, shift = 0, 1, 1

    # Generate two groups with 3 points each
    X = [
        dict(x=random.uniform(lowb, upperb), y=random.uniform(lowb, upperb)),
        dict(x=random.uniform(lowb, upperb), y=random.uniform(lowb, upperb)),
        dict(x=random.uniform(lowb, upperb) + shift, y=random.uniform(
            lowb,
            upperb
        ) + shift),  # true one
        dict(x=random.uniform(lowb, upperb), y=random.uniform(lowb, upperb)),
        dict(x=random.uniform(lowb, upperb), y=random.uniform(lowb, upperb)),
        dict(x=random.uniform(lowb, upperb) + shift, y=random.uniform(
            lowb,
            upperb
        ) + shift)  # true one
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
    ranker = LinearSVMRank()
    ranker.fit(X=X, y=y, groups=groups)

    # Generate a group of three points, the second (index=1) is the true one
    candidates = [
        dict(x=random.uniform(lowb, upperb), y=random.uniform(lowb, upperb)),
        dict(x=random.uniform(lowb, upperb) + shift, y=random.uniform(
            lowb,
            upperb
        ) + shift),  # true one
        dict(x=random.uniform(lowb, upperb), y=random.uniform(lowb, upperb))
    ]

    # Predict
    ranked_candidates, scores = ranker.predict(candidates)
    winner_index = ranked_candidates[0]

    assert winner_index == 1
