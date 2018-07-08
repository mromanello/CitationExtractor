"""Tests for the module `citation_extractor.ned`."""
# -*- coding: utf-8 -*-
# author: Matteo Romanello, matteo.romanello@gmail.com

import pytest
import logging
import pickle
import pkg_resources
import pandas as pd
from citation_extractor.pipeline import NIL_URN
from citation_extractor.ned.candidates import CandidatesGenerator
from citation_extractor.ned.ml import LinearSVMRank
from citation_extractor.ned.matchers import MLCitationMatcher
import random

logger = logging.getLogger(__name__)


@pytest.mark.skip
def test_pickle_citation_matcher(citation_matcher):
    """Test whether instances of `CitationMatcher` can be pickled."""
    pickled_citation_matcher = pickle.dumps(citation_matcher)
    unpickled_citation_matcher = pickle.loads(pickled_citation_matcher)
    assert unpickled_citation_matcher is not None


def test_extract_features(feature_extractor_quick, aph_testset_dataframe):
    fe = feature_extractor_quick
    test_df_data = aph_testset_dataframe

    logger.debug(test_df_data.info())

    for id_row, row in test_df_data.iterrows():
        if row["urn"] != NIL_URN:
            # TODO: should be called on the candidates!
            # and `fv` should be a list
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
            logger.info(
                "Feature vector for {} {}: {}".format(
                    row["surface_norm"],
                    row["scope"],
                    fv
                )
            )
            nfv = fe.extract_nil(row["type"], row["scope"], [fv])
            logger.info(
                "Feature vector (nil) for {} {}: {}".format(
                    row["surface_norm"],
                    row["scope"],
                    nfv
                )
            )
        else:
            logger.debug("Skipped {}".format(row))


def test_candidate_generator(
    feature_extractor_quick,
    knowledge_base,
    aph_testset_dataframe
):

    fe = feature_extractor_quick
    _kb_norm_authors = fe._kb_norm_authors
    _kb_norm_works = fe._kb_norm_works

    cg = CandidatesGenerator(
        knowledge_base,
        kb_norm_authors=_kb_norm_authors,
        kb_norm_works=_kb_norm_works
    )

    for mention_id, row in aph_testset_dataframe.iterrows():

        surface = row['surface_norm_dots']
        scope = row['scope']
        entity_type = row['type']

        candidates = cg.generate_candidates(surface, entity_type, scope)
        logger.info(
            'Generated {} candidates for {}'.format(
                len(candidates),
                mention_id
            )
        )


def test_generate_candidates_parallel(
    feature_extractor_quick,
    knowledge_base,
    aph_testset_dataframe
):
    fe = feature_extractor_quick
    _kb_norm_authors = fe._kb_norm_authors
    _kb_norm_works = fe._kb_norm_works

    cg = CandidatesGenerator(
        knowledge_base,
        kb_norm_authors=_kb_norm_authors,
        kb_norm_works=_kb_norm_works
    )

    candidates = cg.generate_candidates_parallel(
        aph_testset_dataframe.head(50)
    )
    logger.debug(candidates)


# @pytest.mark.skip
def test_ml_citation_matcher(
    feature_extractor_quick,
    # aph_testset_dataframe,
    # aph_goldset_dataframe
):
    aph_testset_dataframe = pd.read_pickle(
        pkg_resources.resource_filename(
            'citation_extractor',
            'data/pickles/aph_test_df.pkl'
        )
    )

    aph_goldset_dataframe = pd.read_pickle(
        pkg_resources.resource_filename(
            'citation_extractor',
            'data/pickles/aph_gold_df.pkl'
        )
    )
    cm = MLCitationMatcher(
        aph_goldset_dataframe.head(50),
        # aph_goldset_dataframe,
        feature_extractor=feature_extractor_quick,
        include_nil=True,
        parallelize=True
    )

    logger.info(cm.settings)

    for row_id, row in aph_testset_dataframe.head(100).iterrows():
        result = cm.disambiguate(
            row["surface"],
            row["scope"],
            row["type"],
            row["doc_title"],
            row["doc_title_mentions"],
            row["doc_text"],
            row["other_mentions"],
        )

        logger.info(u'Disambiguation for {} ({}): {}'.format(
            row["surface"],
            row["scope"],
            result
        ))
        logger.info("Predicted: {}; ground truth: {}".format(
            result.urn,
            row["urn_clean"]
        ))
    logger.info(cm)


def test_svm_rank():
    lowb, upperb, shift = 0, 1, 1

    # Generate some fake data
    X, y, groups = [], [], []
    nb_groups = 4
    nb_points = 3
    for group_id in range(nb_groups):

        # Add false points
        for i in range(nb_points - 1):
            X.append(
                dict(
                    x=random.uniform(lowb, upperb),
                    y=random.uniform(lowb, upperb),
                    z=False
                )
            )
            y.append(0)
            groups.append(group_id)

        # true point
        X.append(
            dict(
                x=random.uniform(lowb, upperb) + shift,
                y=random.uniform(lowb, upperb) + shift,
                z=False
            )
        )
        y.append(1)
        groups.append(group_id)

    print(X)
    print(y)
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
