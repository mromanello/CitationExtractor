"""py.test config file."""

# -*- coding: utf-8 -*-
# author: Matteo Romanello, matteo.romanello@gmail.com

import os
import pickle
import pytest
import pprint
import logging
import pkg_resources
from pytest import fixture
import pandas as pd
from citation_extractor import pipeline
from citation_extractor.settings import crf, svm, maxent, crfsuite
from citation_extractor.io.brat import load_brat_data
from citation_extractor.ned.matchers import CitationMatcher, MLCitationMatcher
from citation_extractor.ned.features import FeatureExtractor
from knowledge_base import KnowledgeBase as KnowledgeBaseNew

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@fixture(scope="session")
def feature_extractor_quick():
    """Instantiate an instance of FeatureExtractor from pickled data."""
    prior_prob = pd.read_pickle(
        'citation_extractor/data/pickles/prior_prob.pkl'
    )

    em_prob = pd.read_pickle('citation_extractor/data/pickles/em_prob.pkl')

    me_prob = pd.read_pickle('citation_extractor/data/pickles/me_prob.pkl')

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
        entity_mention_prob=em_prob,
    )
    return fe


@fixture(scope="session")
def feature_extractor_nopickles(
    knowledge_base,
    aph_gold_ann_files,
    crfsuite_citation_extractor,
    postaggers,
    aph_titles,
    aph_testset_dataframe,
):
    """Create an instance of MLCitationMatcher."""
    train_df_data = load_brat_data(  # TODO create a fixture out of thi  s
        crf_citation_extractor,
        knowledge_base,
        postaggers,
        aph_gold_ann_files,
        aph_titles,
    )

    # initialise a FeatureExtractor
    fe = FeatureExtractor(knowledge_base, train_df_data)
    logger.info(fe)

    aph_testset_dataframe.to_pickle(
        'citation_extractor/data/pickles/aph_test_df.pkl'
    )

    # pickle probability dataframes
    fe._prior_prob.to_pickle('citation_extractor/data/pickles/prior_prob.pkl')
    fe._em_prob.to_pickle('citation_extractor/data/pickles/em_prob.pkl')
    fe._me_prob.to_pickle('citation_extractor/data/pickles/me_prob.pkl')

    # serialize normalized authors
    with open('citation_extractor/data/pickles/kb_norm_authors.pkl', 'wb') as f:
        pickle.dump(fe._kb_norm_authors, f)

    # serialize normalized works
    with open('citation_extractor/data/pickles/kb_norm_works.pkl', 'wb') as f:
        pickle.dump(fe._kb_norm_works, f)

    # serialize the FeatureExtractor
    with open(
        'citation_extractor/data/pickles/ml_feature_extractor.pkl', 'wb'
    ) as f:
        pickle.dump(fe, f)

    return fe


@fixture(scope="session")
def crf_citation_extractor(tmpdir_factory):
    """Initialise a citation extractor trained with CRF on APh corpus."""
    crf.TEMP_DIR = str(tmpdir_factory.mktemp('tmp')) + "/"
    crf.OUTPUT_DIR = str(tmpdir_factory.mktemp('out')) + "/"
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
        'input': pkg_resources.resource_filename(
            'citation_extractor', 'data/aph_corpus/devset/orig/'
        ),
        'iob': str(tmpdir_factory.mktemp('iob')) + "/",
        'txt': str(tmpdir_factory.mktemp('txt')) + "/",
        'iob_ne': str(tmpdir_factory.mktemp('iob_ne')) + "/",
        'ann': str(tmpdir_factory.mktemp('ann')) + "/",
        'json': str(tmpdir_factory.mktemp('json')) + "/",
    }


@fixture(scope="session")
def aph_gold_ann_files():
    ann_dir = pkg_resources.resource_filename(
        'citation_extractor', 'data/aph_corpus/goldset/ann'
    )
    ann_files = pkg_resources.resource_listdir(
        'citation_extractor', 'data/aph_corpus/goldset/ann'
    )
    return ann_dir, ann_files


@fixture(scope="session")
def aph_test_ann_files():
    ann_dir = pkg_resources.resource_filename(
        'citation_extractor', 'data/aph_corpus/testset/ann'
    )
    ann_files = pkg_resources.resource_listdir(
        'citation_extractor', 'data/aph_corpus/testset/ann'
    )
    return ann_dir, ann_files


@fixture(scope="session")
def aph_testset_dataframe(
    crfsuite_citation_extractor,
    knowledge_base,
    postaggers,
    aph_test_ann_files,
    aph_titles,
):
    """Return a pandas DataFrame containing the APh data for testing."""
    pickle_path = pkg_resources.resource_filename(
        'citation_extractor', 'data/pickles/aph_test_df.pkl'
    )
    if os.path.exists(pickle_path):
        logger.info("Read in pickled dataset {}".format(pickle_path))
        return pd.read_pickle(pickle_path)
    else:
        logger.info(
            "Loading test set data (%i documents) from %s"
            % (len(aph_test_ann_files[1]), aph_test_ann_files[0])
        )
        dataframe = load_brat_data(
            crfsuite_citation_extractor,
            knowledge_base,
            postaggers,
            aph_test_ann_files,
            aph_titles,
        )

        # save for later
        dataframe.to_pickle(pickle_path)

        assert dataframe is not None
        assert isinstance(dataframe, pd.DataFrame)
        assert dataframe.shape[0] > 0

        return dataframe


@fixture(scope="session")
def aph_goldset_dataframe(
    crfsuite_citation_extractor,
    knowledge_base,
    postaggers,
    aph_gold_ann_files,
    aph_titles,
):
    """Return a pandas DataFrame containing the APh data for training."""
    pickle_path = pkg_resources.resource_filename(
        'citation_extractor', 'data/pickles/aph_gold_df.pkl'
    )
    if os.path.exists(pickle_path):
        logger.info("Read in pickled dataset {}".format(pickle_path))
        return pd.read_pickle(pickle_path)
    else:
        logger.info(
            "Loading training set data ({} documents) from {}".format(
                len(aph_gold_ann_files[1]), aph_gold_ann_files[0]
            )
        )
        dataframe = load_brat_data(
            crfsuite_citation_extractor,
            knowledge_base,
            postaggers,
            aph_gold_ann_files,
            aph_titles,
        )
        # save for later
        dataframe.to_pickle(pickle_path)
        assert dataframe is not None
        assert isinstance(dataframe, pd.DataFrame)
        assert dataframe.shape[0] > 0

        return dataframe


@fixture(scope="session")
def ml_citation_matcher(feature_extractor_quick, aph_goldset_dataframe):
    matcher = MLCitationMatcher(
        aph_goldset_dataframe,
        feature_extractor=feature_extractor_quick,
        include_nil=True,
        parallelize=True,
        C=10,
    )
    return matcher


@fixture(scope="session")
def citation_matcher(knowledge_base):
    """Initialise a CitationMatcher."""
    return CitationMatcher(knowledge_base)


@fixture(scope="session")
def knowledge_base():
    """Initialise a HuCit KnowledgeBase (new version, standalone package)."""
    try:
        config_file = pkg_resources.resource_filename(
            'knowledge_base', 'config/virtuoso.ini'
        )
        kb = KnowledgeBaseNew(config_file)
        kb.get_authors()[0]
        return kb
    except Exception:
        config_file = pkg_resources.resource_filename(
            'knowledge_base', 'config/inmemory.ini'
        )
        return KnowledgeBaseNew(config_file)


@fixture(scope="session")
def postaggers():
    """Return a dict with language codes as keys and PoS taggers as values."""
    abbreviations = pkg_resources.resource_filename(
        'citation_extractor', 'data/aph_corpus/extra/abbreviations.txt'
    )
    return pipeline.get_taggers(abbrev_file=abbreviations)


@fixture(scope="session")
def aph_titles():
    """Return a datafrmae with the tiles of APh abstracts in the dataset."""
    titles_dir = pkg_resources.resource_filename(
        'citation_extractor', 'data/aph_corpus/titles.csv'
    )
    titles = pd.read_csv(titles_dir).set_index('id')[['title']]
    return titles
