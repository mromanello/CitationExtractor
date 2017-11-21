# -*- coding: utf-8 -*-

from __future__ import print_function

import logging

import os
import re
import unicodedata, sys
import string

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import svm, linear_model, cross_validation, preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

import jellyfish
from stop_words import get_stop_words, safe_get_stop_words

import itertools
from itertools import combinations
from random import shuffle
import scipy.sparse as sp

import multiprocessing

import pylab as pl

import pkg_resources

import citation_extractor.Utils.RankingSVM as rsvm

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
        for k, v in tfidf.iteritems():
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
        feature_vector = {}

        if not feature_dicts:
            return feature_vector

        bool_features = set()
        float_features = set()
        for d in feature_dicts:
            for k, v in d.iteritems():
                if '_ss_' in k:  # string sim only
                    if type(v) == bool:
                        bool_features.add(k)
                    else:
                        float_features.add(k)

        dv = DictVectorizer(sparse=False)
        x = dv.fit_transform(feature_dicts)

        for bf in bool_features:
            bfi = dv.vocabulary_.get(bf)
            values = x[:, bfi]
            feature_vector['NIL_NO_' + bf] = sum(values) == 0.0

        for ff in float_features:
            ffi = dv.vocabulary_.get(ff)
            values = x[:, ffi]
            feature_vector['NIL_MAX_' + ff] = max(values)
            feature_vector['NIL_AVG_' + ff] = avg(values)
            feature_vector['NIL_MAX-AVG_' + ff] = max(values) - avg(values)

        return feature_vector

    def extract(self, m_surface, m_scope, m_type, m_title_mentions, m_title, m_doc_text, m_prev_entities,
                candidate_urn):
        LOGGER.info('Extracting features for ...')

        feature_vector = {}

        string_sim = True
        context_sim = True
        prob = True

        # The mention is an author name and searching for an author
        if m_type == AUTHOR_TYPE and not m_scope:

            surf = m_surface
            names = kb_norm_authors.loc[candidate_urn, 'norm_names_clean']
            abbr = kb_norm_authors.loc[candidate_urn, 'norm_abbr']

            if string_sim:
                self.add_string_similarities(feature_vector, 'ANS_ss_', surf, names)
                self.add_abbr_match(feature_vector, 'ANS_ss_', surf, abbr)

            if context_sim:
                self.add_tfidf_similarity(feature_vector, 'ANS_cxt_tfidf_', candidate_urn, m_doc_text)
                self.add_title_similarities(feature_vector, 'ANS_cxt_title_', m_title_mentions, m_title, candidate_urn)
                self.add_other_mentions_string_similarities(feature_vector, 'ANS_cxt_om_', m_surface, m_scope, m_type,
                                                            m_prev_entities, candidate_urn)

            if prob:
                self.add_prior_prob(feature_vector, 'ANS_prob_entity_prior', candidate_urn)
                # self.add_me_prob(feature_vector, 'ANS_prob_m_given_e', surf, candidate_urn)
                # self.add_em_prob(feature_vector, 'ANS_prob_e_given_m', surf, candidate_urn)

        # The mention is an author name but searching for a work
        elif m_type == AUTHOR_TYPE and m_scope:

            surf = m_surface
            aurn = kb_norm_works.loc[candidate_urn, 'author']
            names = kb_norm_authors.loc[aurn, 'norm_names_clean']
            abbr = kb_norm_authors.loc[aurn, 'norm_abbr']

            if string_sim:
                self.add_string_similarities(feature_vector, 'AS_ss_', surf, names)
                self.add_abbr_match(feature_vector, 'AS_ss_', surf, abbr)

            if context_sim:
                self.add_tfidf_similarity(feature_vector, 'AS_cxt_tfidf_', candidate_urn, m_doc_text)
                self.add_title_similarities(feature_vector, 'AS_cxt_title_', m_title_mentions, m_title, candidate_urn)
                self.add_other_mentions_string_similarities(feature_vector, 'AS_cxt_om_', m_surface, m_scope, m_type,
                                                            m_prev_entities, candidate_urn)

            if prob:
                self.add_prior_prob(feature_vector, 'AS_prob_entity_prior', candidate_urn)
                # self.add_me_prob(feature_vector, 'AS_prob_m_given_e', surf, candidate_urn)
                self.add_em_prob(feature_vector, 'AS_prob_e_given_m', surf, candidate_urn)

        # The mention is a work name and searching for a work
        elif m_type == WORK_TYPE:

            surf = m_surface
            names = kb_norm_works.loc[candidate_urn, 'norm_titles_clean']
            abbr = kb_norm_works.loc[candidate_urn, 'norm_abbr']

            if string_sim:
                self.add_string_similarities(feature_vector, 'W_ss_', surf, names)
                self.add_abbr_match(feature_vector, 'W_ss_', surf, abbr)

            if context_sim:
                self.add_tfidf_similarity(feature_vector, 'W_cxt_tfidf_', candidate_urn, m_doc_text)
                self.add_title_similarities(feature_vector, 'W_cxt_title_', m_title_mentions, m_title, candidate_urn)
                self.add_other_mentions_string_similarities(feature_vector, 'W_cxt_om_', m_surface, m_scope, m_type,
                                                            m_prev_entities, candidate_urn)

            if prob:
                self.add_prior_prob(feature_vector, 'W_prob_entity_prior', candidate_urn)
                # self.add_me_prob(feature_vector, 'W_prob_m_given_e', surf, candidate_urn)
                # self.add_em_prob(feature_vector, 'W_prob_e_given_m', surf, candidate_urn)

        # The mention is an author name, work name or mixed and searching for a work
        elif m_type == REFAUWORK_TYPE:

            surf = m_surface
            names = kb_norm_works.loc[candidate_urn, 'norm_titles_clean']
            abbr = kb_norm_works.loc[candidate_urn, 'norm_abbr']
            aurn = kb_norm_works.loc[candidate_urn, 'author']
            anames = kb_norm_authors.loc[aurn, 'norm_names_clean']
            aabbr = kb_norm_authors.loc[aurn, 'norm_abbr']

            if string_sim:
                self.add_mixed_string_similarities(feature_vector, 'R_ss_mix_', surf, names, anames)
                self.add_string_similarities(feature_vector, 'R_ss_w_', surf, names)
                self.add_abbr_match(feature_vector, 'R_ss_w_', surf, abbr)
                self.add_string_similarities(feature_vector, 'R_ss_a_', surf, anames)
                self.add_abbr_match(feature_vector, 'R_ss_a_', surf, aabbr)

            if context_sim:
                self.add_tfidf_similarity(feature_vector, 'R_cxt_tfidf_', candidate_urn, m_doc_text)
                self.add_title_similarities(feature_vector, 'R_cxt_title_', m_title_mentions, m_title, candidate_urn)
                self.add_other_mentions_string_similarities(feature_vector, 'R_cxt_om_', m_surface, m_scope, m_type,
                                                            m_prev_entities, candidate_urn)

            if prob:
                self.add_prior_prob(feature_vector, 'R_prob_entity_prior', candidate_urn)
                # self.add_me_prob(feature_vector, 'R_prob_m_given_e', surf, candidate_urn)
                self.add_em_prob(feature_vector, 'R_prob_e_given_m', surf, candidate_urn)

        else:
            LOGGER.error('Unknown mention type: {}'.format(m_type))

        return feature_vector

    def add_abbr_match(self, feature_vector, feat_prefix, surf, abbr):
        surf = surf.replace(u'.', u'')
        feature_vector[feat_prefix + 'kb_abbr_ex_match'] = exact_match(surf, abbr)

    def add_mixed_string_similarities(self, feature_vector, feat_prefix, surf, names, anames):
        surf = self.clean_surface(surf)
        surf_words = surf.split()
        names_words = set(u' '.join(names).split())
        anames_words = set(u' '.join(anames).split())

        # s1 s2
        matched = re.match(ur'^([a-z]+) ([a-z]+)$', surf)
        if matched:
            s1, s2 = matched.group(1), matched.group(2)

            # author work
            feature_vector[feat_prefix + 'a_ex_match_and_w_ex_match'] = exact_match(s1, anames) and exact_match(s2,
                                                                                                                names)

            # ~author ~work
            feature_vector[feat_prefix + 'a_fuz_match_and_w_fuz_match'] = fuzzy_match(s1, anames) and fuzzy_match(s2,
                                                                                                                  names)

        # s1 s2.
        matched = re.match(ur'^([a-z]+) ([a-z]+)\.$', surf)
        if matched:
            s1, s2 = matched.group(1), matched.group(2)

            # author work.
            feature_vector[feat_prefix + 'a_ex_match_and_w_abbr_match'] = exact_match(s1,
                                                                                      anames_words) and abbreviation_match(
                s2, names_words)

            # author ?.
            feature_vector[feat_prefix + 'a_ex_match_and_unknown'] = exact_match(s1, anames_words)

            # replace u with v
            if u'u' in surf:
                s1, s2 = s1.replace(u'u', u'v'), s2.replace(u'u', u'v')
                feature_vector[feat_prefix + 'a_ex_match_and_w_abbr_match_ureplaced'] = exact_match(s1,
                                                                                                    anames_words) and abbreviation_match(
                    s2, names_words)

        # s1. s2.
        matched = re.match(ur'^([a-z]+)\. ([a-z]+)\.$', surf)
        if matched:
            s1, s2 = matched.group(1), matched.group(2)

            # auth. work.
            feature_vector[feat_prefix + 'a_abbr_match_and_w_abbr_match'] = abbreviation_match(s1,
                                                                                               anames_words) and abbreviation_match(
                s2, names_words)

            # work1. work2.
            feature_vector[feat_prefix + 'w_abbr_seq_match'] = abbreviation_sequence_match([s1, s2], names)

            # auth1. auth2.
            feature_vector[feat_prefix + 'a_abbr_seq_match'] = abbreviation_sequence_match([s1, s2], anames)

            # wrk1. wrk2.
            feature_vector[feat_prefix + 'w_abbr_sparse_match'] = abbreviation_sparse_match(s1 + s2, names)

            # ath1. ath2.
            feature_vector[feat_prefix + 'a_abbr_sparse_match'] = abbreviation_sparse_match(s1 + s2, anames)

            # ?. work.
            feature_vector[feat_prefix + 'unknown_and_w_abbr_match'] = abbreviation_match(s2, names_words)

            # ?. auth.
            feature_vector[feat_prefix + 'unknown_and_a_abbr_match'] = abbreviation_match(s2, anames_words)

            # work. ?
            feature_vector[feat_prefix + 'w_abbr_match_and_unknown'] = abbreviation_match(s1, names_words)

            # auth. ?
            feature_vector[feat_prefix + 'a_abbr_match_and_unknown'] = abbreviation_match(s1, anames_words)

        # s1. s2. s3.
        matched = re.match(ur'^([a-z]+)\. ([a-z]+)\. ([a-z]+)\.$', surf)
        if matched:
            s1, s2, s3 = matched.group(1), matched.group(2), matched.group(3)

            # auth. work1. work2.
            feature_vector[feat_prefix + 'a_abbr_match_and_w_abbr_seq_match'] = abbreviation_match(s1,
                                                                                                   anames) and abbreviation_sequence_match(
                [s2, s3], names)

        # s1 s2. s3.
        matched = re.match(ur'^([a-z]+) ([a-z]+)\. ([a-z]+)\.$', surf)
        if matched:
            s1, s2, s3 = matched.group(1), matched.group(2), matched.group(3)

            # author work1. work2.
            feature_vector[feat_prefix + 'a_ex_match_and_w_abbr_seq_match'] = exact_match(s1,
                                                                                          anames) and abbreviation_sequence_match(
                [s2, s3], names)

            # author work1. work2.
            feature_vector[feat_prefix + 'a_ex_match_and_w_abbr_seq_match'] = exact_match(s1,
                                                                                          anames_words) and abbreviation_sequence_match(
                [s2, s3], names)

        # s1 s2 s3.
        matched = re.match(ur'^([a-z]+) ([a-z]+) ([a-z]+)\.$', surf)
        if matched:
            s1, s2, s3 = matched.group(1), matched.group(2), matched.group(3)

            # author1 author2 work.
            feature_vector[feat_prefix + 'aa_ex_match_and_w_abbr_match'] = exact_match(u' '.join([s1, s2]),
                                                                                       anames) and abbreviation_match(
                s3, names)

            # author1 author2 work.
            feature_vector[feat_prefix + 'aa_fuz_match_and_w_abbr_match'] = fuzzy_match(u' '.join([s1, s2]),
                                                                                        anames) and abbreviation_match(
                s3, names)

            # author work1 work2.
            feature_vector[feat_prefix + 'a_ex_match_and_w_ex_match_and_w_abbr_match'] = exact_match(s1,
                                                                                                     anames) and abbreviation_sequence_match(
                [s2, s3], names)

            # author work1 work2.
            feature_vector[feat_prefix + 'a_ex_match_nwords_and_w_ex_match_and_w_abbr_match'] = exact_match(s1,
                                                                                                            anames_words) and abbreviation_sequence_match(
                [s2, s3], names)

        # s1 s2 s3
        matched = re.match(ur'^([a-z]+) ([a-z]+) ([a-z]+)$', surf)
        if matched:
            s1, s2, s3 = matched.group(1), matched.group(2), matched.group(3)

            # author work1 work2
            feature_vector[feat_prefix + 'a_ex_match_and_ww_ex_match'] = exact_match(s1, anames) and exact_match(
                u' '.join([s2, s3]), names)

    def add_string_similarities(self, feature_vector, feat_prefix, surf, names):

        surf = self.clean_surface(surf)
        surf_words = surf.split()
        names_words = set(u' '.join(names).split())

        if len(surf_words) == 1:
            feat_prefix = feat_prefix + '1w_'

            # Exact match
            feature_vector[feat_prefix + 'ex_match'] = exact_match(surf, names)
            feature_vector[feat_prefix + 'ex_match_nwords'] = exact_match(surf, names_words)

            # Fuzzy match
            feature_vector[feat_prefix + 'fuz_match'] = fuzzy_match(surf, names)
            feature_vector[feat_prefix + 'fuz_match_max'] = fuzzy_match_max(surf, names)
            feature_vector[feat_prefix + 'fuz_match_nwords'] = fuzzy_match(surf, names_words)

            # Fuzzy initial letters match
            feature_vector[feat_prefix + 'fuz_init_match'] = fuzzy_initial_match(surf, names)
            feature_vector[feat_prefix + 'fuz_init_match_max'] = fuzzy_initial_match_max(surf, names)
            feature_vector[feat_prefix + 'fuz_init_match_nwords'] = fuzzy_initial_match(surf, names_words)

            # Fuzzy phonetic match
            # feature_vector[feat_prefix + 'fuz_phon_match'] = fuzzy_phonetic_match(surf, names)
            # feature_vector[feat_prefix + 'fuz_phon_match_max'] = fuzzy_phonetic_match_max(surf, names)
            # feature_vector[feat_prefix + 'fuz_phon_match_nwords'] = fuzzy_phonetic_match(surf, names_words)

            # Match rating comparison
            # feature_vector[feat_prefix + 'fuz_mrc_match'] = fuzzy_mrc(surf, names)

            # Match acronym
            feature_vector[feat_prefix + 'acronym_match'] = acronym_match(surf, names)

            # Match abbreviations
            matched = re.match(ur'^([a-z]+)\.$', surf)
            if matched:
                s1 = matched.group(1)
                feature_vector[feat_prefix + 'abbr_match'] = abbreviation_match(s1, names)
                feature_vector[feat_prefix + 'abbr_match_nwords'] = abbreviation_match(s1, names_words)
                feature_vector[feat_prefix + 'abbr_match_nwords_sparse'] = abbreviation_sparse_match(s1, names_words)


        elif len(surf_words) == 2:
            feat_prefix = feat_prefix + '2w_'

            # Exact match
            feature_vector[feat_prefix + 'ex_match'] = exact_match(surf, names)
            feature_vector[feat_prefix + 'ex_match_nwords'] = exact_match_swords(surf_words, names_words)
            feature_vector[feat_prefix + 'ex_match_nwords_any'] = exact_match_swords_any(surf_words, names_words)

            # Fuzzy match
            feature_vector[feat_prefix + 'fuz_match'] = fuzzy_match(surf, names)
            feature_vector[feat_prefix + 'fuz_match_max'] = fuzzy_match_max(surf, names)
            feature_vector[feat_prefix + 'fuz_match_nwords'] = fuzzy_match_swords(surf_words, names_words)
            feature_vector[feat_prefix + 'fuz_match_nwords_any'] = fuzzy_match_swords_any(surf_words, names_words)

            # Fuzzy initial letters match in s_words vs n_words 'aaaz bbbz' match 'aaax bb cccc' and 'aaax', 'bbbx'
            feature_vector[feat_prefix + 'fuz_init_match'] = fuzzy_initial_match(surf, names)
            feature_vector[feat_prefix + 'fuz_init_match_max'] = fuzzy_initial_match_max(surf, names)
            feature_vector[feat_prefix + 'fuz_init_match_nwords'] = fuzzy_initial_match_swords(surf_words, names_words)

            # Fuzzy phonetic match in s_words vs n_words u'theophylakt simokates' match u'theophylactus simocatta'
            # feature_vector[feat_prefix + 'fuz_phon_match'] = fuzzy_phonetic_match(surf, names)
            # feature_vector[feat_prefix + 'fuz_phon_match_max'] = fuzzy_phonetic_match_max(surf, names)
            # feature_vector[feat_prefix + 'fuz_phon_match_nwords'] = fuzzy_phonetic_match_swords(surf_words, names_words)


        elif len(surf_words) == 3:
            feat_prefix = feat_prefix + '3w_'

            # Exact match
            feature_vector[feat_prefix + 'ex_match'] = exact_match(surf, names)
            feature_vector[feat_prefix + 'ex_match_nwords'] = exact_match_swords(surf_words, names_words)
            feature_vector[feat_prefix + 'ex_match_nwords_any'] = exact_match_swords_any(surf_words, names_words)

            # Fuzzy match
            feature_vector[feat_prefix + 'fuz_match'] = fuzzy_match(surf, names)
            feature_vector[feat_prefix + 'fuz_match_max'] = fuzzy_match_max(surf, names)
            feature_vector[feat_prefix + 'fuz_match_nwords'] = fuzzy_match_swords(surf_words, names_words)
            feature_vector[feat_prefix + 'fuz_match_nwords_any'] = fuzzy_match_swords_any(surf_words, names_words)

            # Fuzzy initial letters match
            feature_vector[feat_prefix + 'fuz_init_match'] = fuzzy_initial_match(surf, names)
            feature_vector[feat_prefix + 'fuz_init_match_max'] = fuzzy_initial_match_max(surf, names)
            feature_vector[feat_prefix + 'fuz_init_match_nwords'] = fuzzy_initial_match_swords(surf_words, names_words)

            # Fuzzy phonetic match
            # feature_vector[feat_prefix + 'fuz_phon_match'] = fuzzy_phonetic_match(surf, names)
            # feature_vector[feat_prefix + 'fuz_phon_match_max'] = fuzzy_phonetic_match_max(surf, names)
            # feature_vector[feat_prefix + 'fuz_phon_match_nwords'] = fuzzy_phonetic_match_swords(surf_words, names_words)


        elif len(surf_words) > 3:
            feat_prefix = feat_prefix + '3+w_'

            surf = self.remove_words_shorter_than(surf, 2)
            surf_words = surf.split()

            # Exact match
            feature_vector[feat_prefix + 'ex_match'] = exact_match(surf, names)
            feature_vector[feat_prefix + 'ex_match_nwords'] = exact_match_swords(surf_words, names_words)
            feature_vector[feat_prefix + 'ex_match_nwords_any'] = exact_match_swords_any(surf_words, names_words)

            # Fuzzy match
            feature_vector[feat_prefix + 'fuz_match'] = fuzzy_match(surf, names)
            feature_vector[feat_prefix + 'fuz_match_max'] = fuzzy_match_max(surf, names)
            feature_vector[feat_prefix + 'fuz_match_nwords'] = fuzzy_match_swords(surf_words, names_words)
            feature_vector[feat_prefix + 'fuz_match_nwords_any'] = fuzzy_match_swords_any(surf_words, names_words)

            # Fuzzy initial letters match in s_words vs n_words 'aaaz bbbz' match 'aaax bb cccc' and 'aaax', 'bbbx'
            feature_vector[feat_prefix + 'fuz_init_match'] = fuzzy_initial_match(surf, names)
            feature_vector[feat_prefix + 'fuz_init_match_max'] = fuzzy_initial_match_max(surf, names)
            feature_vector[feat_prefix + 'fuz_init_match_nwords'] = fuzzy_initial_match_swords(surf_words, names_words)

            # Fuzzy phonetic match in s_words vs n_words u'theophylakt simokates' match u'theophylactus simocatta'
            # feature_vector[feat_prefix + 'fuz_phon_match'] = fuzzy_phonetic_match(surf, names)
            # feature_vector[feat_prefix + 'fuz_phon_match_max'] = fuzzy_phonetic_match_max(surf, names)
            # feature_vector[feat_prefix + 'fuz_phon_match_nwords'] = fuzzy_phonetic_match_swords(surf_words, names_words)

    def add_other_mentions_string_similarities(self, feature_vector, feat_prefix, surf, scope, mtype, other_mentions,
                                               candidate_urn):

        # other_mentions = [(type, surface, scope), ...]
        anames, wnames = [], []
        if candidate_urn in kb_norm_authors.index:
            anames = kb_norm_authors.loc[candidate_urn, 'norm_names_clean']
            wnames = []
            for w in kb_norm_authors.loc[candidate_urn, 'works']:
                for wn in kb_norm_works.loc[w, 'norm_titles_clean']:
                    wnames.append(wn)

        elif candidate_urn in kb_norm_works.index:
            wnames = kb_norm_works.loc[candidate_urn, 'norm_titles_clean']
            aurn = kb_norm_works.loc[candidate_urn, 'author']
            anames = kb_norm_authors.loc[aurn, 'norm_names_clean']

        other_mentions = filter(lambda (typ, srf): srf != surf,
                                set(map(lambda (typ, srf, scp): (typ, srf), other_mentions)))
        amatch, wmatch, rmatch = [], [], []
        for om_type, om_surf in other_mentions:

            if om_type == AUTHOR_TYPE:
                tmp_vector = {}
                self.add_string_similarities(tmp_vector, 'tmp', om_surf, anames)
                amatch.append(self.dict_contains_match(tmp_vector))

            if om_type == WORK_TYPE:
                tmp_vector = {}
                self.add_string_similarities(tmp_vector, 'tmp', om_surf, wnames)
                wmatch.append(self.dict_contains_match(tmp_vector))

            if om_type == REFAUWORK_TYPE:
                tmp_vector = {}
                self.add_string_similarities(tmp_vector, 'tmp', om_surf, anames)
                self.add_string_similarities(tmp_vector, 'tmp', om_surf, wnames)
                self.add_mixed_string_similarities(tmp_vector, 'tmp', om_surf, wnames, anames)
                rmatch.append(self.dict_contains_match(tmp_vector))

        feature_vector[feat_prefix + 'author_match'] = any(amatch)
        feature_vector[feat_prefix + 'author_match_nb'] = float(sum(amatch)) / max(len(other_mentions), 1)
        feature_vector[feat_prefix + 'work_match'] = any(wmatch)
        feature_vector[feat_prefix + 'work_match_nb'] = float(sum(wmatch)) / max(len(other_mentions), 1)
        feature_vector[feat_prefix + 'refauwork_match'] = any(rmatch)
        feature_vector[feat_prefix + 'refauwork_match_nb'] = float(sum(rmatch)) / max(len(other_mentions), 1)

    def dict_contains_match(self, dictionary):
        return any(filter(lambda v: type(v) == bool, dictionary.values()))

    def add_title_similarities(self, feature_vector, feat_prefix, title_mentions, title, candidate_urn):

        stripped_title = self.remove_words_shorter_than(title, 3)

        if candidate_urn in kb_norm_authors.index:
            names = kb_norm_authors.loc[candidate_urn, 'norm_names_clean']
            wnames = []
            for w in kb_norm_authors.loc[candidate_urn, 'works']:
                for wn in kb_norm_works.loc[w, 'norm_titles_clean']:
                    wnames.append(wn)

            feature_vector[feat_prefix + 'auth_in_title'] = self.names_in_title(names, stripped_title)
            feature_vector[feat_prefix + 'auth_in_title_pw'] = self.names_in_title_perword(names, stripped_title)
            feature_vector[feat_prefix + 'authworks_in_title'] = self.names_in_title(wnames, stripped_title)
            feature_vector[feat_prefix + 'authworks_in_title_pw'] = self.names_in_title_perword(wnames, stripped_title)

            feature_vector[feat_prefix + 'auth_in_extr_title'] = self.names_in_extracted_title(names, title_mentions)
            feature_vector[feat_prefix + 'auth_in_extr_title_pw'] = self.names_in_extracted_title_perword(names,
                                                                                                          title_mentions)
            feature_vector[feat_prefix + 'authworks_in_extr_title'] = self.names_in_extracted_title(wnames,
                                                                                                    title_mentions)
            feature_vector[feat_prefix + 'authworks_in_extr_title_pw'] = self.names_in_extracted_title_perword(wnames,
                                                                                                               title_mentions)

        elif candidate_urn in kb_norm_works.index:
            names = kb_norm_works.loc[candidate_urn, 'norm_titles_clean']
            aurn = kb_norm_works.loc[candidate_urn, 'author']
            anames = kb_norm_authors.loc[aurn, 'norm_names_clean']

            feature_vector[feat_prefix + 'work_in_title'] = self.names_in_title(names, stripped_title)
            feature_vector[feat_prefix + 'work_in_title_pw'] = self.names_in_title_perword(names, stripped_title)
            feature_vector[feat_prefix + 'workauth_in_title'] = self.names_in_title(anames, stripped_title)
            feature_vector[feat_prefix + 'workauth_in_title_pw'] = self.names_in_title_perword(anames, stripped_title)

            feature_vector[feat_prefix + 'work_in_extr_title'] = self.names_in_extracted_title(names, title_mentions)
            feature_vector[feat_prefix + 'work_in_extr_title_pw'] = self.names_in_extracted_title_perword(names,
                                                                                                          title_mentions)
            feature_vector[feat_prefix + 'workauth_in_extr_title'] = self.names_in_extracted_title(anames,
                                                                                                   title_mentions)
            feature_vector[feat_prefix + 'workauth_in_extr_title_pw'] = self.names_in_extracted_title_perword(anames,
                                                                                                              title_mentions)

    def split_names(self, names):
        splitted_names = set(u' '.join(names).split())
        return list(splitted_names)

    def qexact_match(self, s1, s2):
        return s1 == s2 or levenshtein_distance_norm(s1, s2) >= 0.9

    def names_in_title(self, names, title):
        for name in names:
            for word in title.split():
                if self.qexact_match(name, word):
                    return True
        return False

    def names_in_title_perword(self, names, title):
        names = self.split_names(names)
        return self.names_in_title(names, title)

    def names_in_extracted_title(self, names, title_mentions):
        for m_type, m_surface in title_mentions:
            for name in names:
                if self.qexact_match(m_surface, name):
                    return True
        return False

    def names_in_extracted_title_perword(self, names, title_mentions):
        names = self.split_names(names)
        return self.names_in_extracted_title(names, title_mentions)

    def add_tfidf_similarity(self, feature_vector, feat_prefix, candidate_urn, doc_text):
        if self.tfidf is not None:
            urn_target = candidate_urn
            if urn_target in kb_norm_works.index:
                urn_target = kb_norm_works.loc[urn_target, 'author']

            tfidf_scores = []
            for lang in LANGUAGES:
                tfidf_scores.append(self.text_similarity(doc_text, urn_target, lang))

            # feature_vector[feat_prefix + 'max'] = max(tfidf_scores)
            feature_vector[feat_prefix + 'avg'] = avg(tfidf_scores)

    def add_prior_prob(self, feature_vector, feat_prefix, candidate_urn):
        if self.prior_prob is not None:
            feature_vector[feat_prefix] = self.prior_prob.loc[candidate_urn, 'prob']

    def add_me_prob(self, feature_vector, feat_prefix, surf, candidate_urn):
        if self.me_prob is not None and surf in self.me_prob.index:
            feature_vector[feat_prefix] = self.me_prob.loc[surf, candidate_urn]

    def add_em_prob(self, feature_vector, feat_prefix, surf, candidate_urn):
        if self.em_prob is not None and surf in self.em_prob.index:
            feature_vector[feat_prefix] = self.em_prob.loc[surf, candidate_urn]

    def text_similarity(self, text, urn, lang):
        if urn not in self.tfidf[lang]['urn_to_index'].keys():
            return 0.0

        text_vector = self.tfidf[lang]['vectorizer'].transform([text])
        urn_doc_index = self.tfidf[lang]['urn_to_index'][urn]
        urn_doc_vector = self.tfidf[lang]['matrix'][urn_doc_index]

        return cosine_similarity(text_vector, urn_doc_vector).flatten()[0]

    def remove_initial_word(self, word, name):
        pattern = u'^' + word + u' (.+)$'
        matched = re.match(pattern, name)
        if matched:
            return matched.group(1)
        return name

    def remove_words_shorter_than(self, name, k):
        filtered = filter(lambda w: len(w) > k, name.split())
        return u' '.join(filtered)

    def remove_preps(self, name):
        name_words = filter(lambda w: w not in PREPS, name.split())
        return u' '.join(name_words)

    def clean_surface(self, surface):
        if len(surface.split()) > 1:
            surface = self.remove_initial_word(u'de', surface)
        if len(surface.split()) > 1:
            surface = self.remove_initial_word(u'in', surface)
        if len(surface.split()) > 2:
            surface = self.remove_preps(surface)
        return surface


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
