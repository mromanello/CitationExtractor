# -*- coding: utf-8 -*-

from __future__ import print_function
import logging
import pkg_resources
import os
from stop_words import safe_get_stop_words
from sklearn.feature_extraction.text import TfidfVectorizer

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
    def __init__(self, kb=None):
        LOGGER.info('Initializing Citation Matcher')
        # TODO: Load, pre-process, ... the KB (?)
        self._kb = None
        self._feature_extractor = None
        self._ranker = None

    def train(self, train_data=None, wikipages_dir=None, include_nil=True):
        LOGGER.info('Starting training')
        # TODO: get tf-idf data from wiki texts
        tfidf = self._compute_tfidf_matrix()
        for k, v in  tfidf.iteritems():
            print(k, v)
        # TODO: compute probs from train data
        # TODO: generate features for candidates (FeatureExtractor)
        # TODO: generate ranking function (SVMRank)

    def disambiguate(self, surface, scope, type, doc_title, mentions_in_title, doc_text, other_mentions, **kwargs):
        LOGGER.info('Disambiguating ...')
        # TODO: get candidates (pre-computed?)
        # TODO: generate features for candidates (FeatureExtractor)
        # TODO: rank candidates (SVMRank)
        return None

    # TODO: should be in FeatureExtractor
    def _compute_tfidf_matrix(self, base_dir=None):
        LOGGER.info('Computing TF-IDF matrix (base_dir={})'.format(base_dir))
        tfidf_data = {}

        # Compute tf-idf distribution for each language
        for lang in LANGUAGES:
            lang_data = {}

            if not base_dir:
                resources_dir = 'data/wikipages/text/authors/{}'.format(lang)
                text_authors_dir_lang = pkg_resources.resource_filename('citation_extractor', resources_dir)
                text_authors_files = pkg_resources.resource_listdir('citation_extractor', resources_dir)
            else:
                text_authors_dir_lang = os.path.join(base_dir, lang)
                text_authors_files = os.listdir(text_authors_dir_lang)

            texts = []
            urn_to_index = {}
            index = 0
            for file in text_authors_files:
                if not file.endswith('.txt'):
                    continue

                urn = file.replace('.txt', '')
                filepath = os.path.join(text_authors_dir_lang, file)
                with open(filepath) as txt_file:
                    text = txt_file.read()
                texts.append(text)
                urn_to_index[urn] = index
                index += 1

            # Dictionary mapping a URN to an index (row)
            lang_data['urn_to_index'] = urn_to_index

            tfidf_vectorizer = TfidfVectorizer(
                input='content',
                strip_accents='unicode',
                analyzer='word',
                stop_words=safe_get_stop_words(lang)
            )

            # Language-specific vectorizer
            lang_data['vectorizer'] = tfidf_vectorizer

            # Tf-idf matrix computed with the specific vectorizer
            tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
            lang_data['matrix'] = tfidf_matrix

            tfidf_data[lang] = lang_data

        return tfidf_data


class FeatureExtractor(object):
    def __init__(self, tfidf_data=None, entities_prior_prob=None, m_given_e_prob=None, e_given_m_prob=None):
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


def main():
    logging.basicConfig(level=logging.INFO)

    matcher = MLCitationMatcher()
    matcher.train()



if __name__ == '__main__':
    main()
