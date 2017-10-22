# -*- coding: utf-8 -*-

import logging

LOGGER = logging.getLogger(__name__)

NIL_URN = 'urn:cts:GreekLatinLit:NIL'
LANGUAGES = ['en', 'es', 'de', 'fr', 'it']
PREPS = [u'di', u'da', u'of', u'von', u'de']
AUTHOR_TYPE = 'AAUTHOR'
WORK_TYPE = 'AWORK'
REFAUWORK_TYPE = 'REFAUWORK'


# TODO: how to deal with parallel computing?
# TODO: how to deal/apply with (optional?) refinement step?


class MLCitationMatcher(object):
    def __init__(self, kb):
        LOGGER.info('Initializing Citation Matcher')
        # TODO: Load, pre-process, ... the KB (?)
        self._kb = None
        self._feature_extractor = None
        self._ranker = None

    def train(self, train_data, wikipages_dir, include_nil=True):
        LOGGER.info('Starting training')
        # TODO: get tf-idf data from wiki texts
        # TODO: compute probs from train data
        # TODO: generate features for candidates (FeatureExtractor)
        # TODO: generate ranking function (SVMRank)

    def disambiguate(self, surface, scope, type, doc_title, mentions_in_title, doc_text, other_mentions, **kwargs):
        LOGGER.info('Disambiguating ...')
        # TODO: get candidates (pre-computed?)
        # TODO: generate features for candidates (FeatureExtractor)
        # TODO: rank candidates (SVMRank)
        return None


class FeatureExtractor(object):
    def __init__(self, tfidf_data, entities_prior_prob, m_given_e_prob, e_given_m_prob):
        LOGGER.info('Initializing Feature Extractor')
        self.tfidf = tfidf_data
        self.prior_prob = entities_prior_prob
        self.me_prob = m_given_e_prob
        self.em_prob = e_given_m_prob

    def extract_nil(self, m_type, m_scope, feature_dicts):
        LOGGER.info('Extracting NIL features for ...')
        # TODO: extract features
        return None

    def extract(self, m_surface, m_scope, m_type, m_title_mentions, m_title, m_doc_text, m_prev_entities,
                candidate_urn):
        LOGGER.info('Extracting features for ...')
        # TODO: extract features
        return None


class SVMRank(object):
    def __init__(self):
        LOGGER.info('Initializing SVM Rank')
        self._svm = None

    def fit(self, X, y, groups):
        LOGGER.info('Fitting data ...')
        # TODO: apply pairwise transform
        # TODO: (optional?) compute best C parameter (k-folded)
        # TODO: fit linear SVM

    def predict(self, x):
        LOGGER.info('Predicting ...')
        # TODO: apply dot product + sort
        return None
