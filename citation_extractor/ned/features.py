"""Code related to feature extraction for thr NED step."""

# -*- coding: utf-8 -*-

from __future__ import print_function
import pdb  # TODO remove when done with development
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from stop_words import safe_get_stop_words

from citation_extractor.Utils.strmatching import StringSimilarity
from citation_extractor.ned import AUTHOR_TYPE, WORK_TYPE, REFAUWORK_TYPE
from citation_extractor.ned import NIL_URN, LANGUAGES, PREPS
from citation_extractor.Utils.extra import avg

import pkg_resources
import pandas as pd
import logging
import os
import re

LOGGER = logging.getLogger(__name__)


# TODO: how to deal with parallel computing?

class FeatureExtractor(object):
    """TODO."""

    def __init__(self, kb=None, train_data=None, **kwargs):
        """Initialise an instance of FeatureExtractor.

        Optional kwargs:
            - `kb_author_urns`
            - `kb_work_urns`
            - `prior_prob`
            - `mention_entity_prob`
            - `entity_mention_prob`

        :param kb: instance of HuCit KnowledgeBase
        :param train_data:
        """
        LOGGER.info('Initializing Feature Extractor')

        if kb is not None:

            # get the list of author IDs (URNs)
            self._kb_author_urns = [
                str(a.get_urn())
                for a in kb.get_authors()
                if a.get_urn() is not None
                ]

            # get the list of work IDs (URNs)
            self._kb_work_urns = [
                str(w.get_urn())
                for a in kb.get_authors()
                for w in a.get_works()
                if w.get_urn() is not None
                ]

        elif 'kb_author_urns' in kwargs and 'kb_work_urns' in kwargs:
            self._kb_author_urns = kwargs['kb_author_urns']
            self._kb_work_urns = kwargs['kb_work_urns']

        else:
            raise Exception

        # TODO: pre-compute normalized authors/works (?)
        self._kb_norm_authors = None
        self._kb_norm_works = None

        # TODO: define how wiki data is referenced
        self._tfidf = self._compute_tfidf_matrix()

        if train_data is not None:

            self._prior_prob = self._compute_entity_probability(
                train_data
            )
            self._me_prob = self._compute_mention_entity_probability(
                train_data
            )
            self._em_prob = self._compute_entity_mention_probability(
                train_data
            )

        else:
            self._prior_prob = kwargs['prior_prob']
            self._me_prob = kwargs['mention_entity_prob']
            self._em_prob = kwargs['entity_mention_prob']

    def _compute_tfidf_matrix(self, base_dir=None):
        LOGGER.info('Computing TF-IDF matrix (base_dir={})'.format(base_dir))
        tfidf_data = {}

        # Compute tf-idf distribution for each language
        for lang in LANGUAGES:
            lang_data = {}

            if not base_dir:
                resources_dir = 'data/wikipages/text/authors/{}'.format(lang)
                text_authors_dir_lang = pkg_resources.resource_filename(
                    'citation_extractor',
                    resources_dir
                )
                text_authors_files = pkg_resources.resource_listdir(
                    'citation_extractor',
                    resources_dir
                )
            else:
                text_authors_dir_lang = os.path.join(base_dir, lang)
                text_authors_files = os.listdir(text_authors_dir_lang)

            LOGGER.info('Computing TF-IDF matrix: using %i document for \
                        language %s' % (len(text_authors_files), lang))

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
            LOGGER.info('Done computing TF-IDF matrix.')

        return tfidf_data

    def _compute_entity_probability(self, train_data):
        """Compute the probability of an entity to occur in the training data.

        :param train_data: a dataframe with the traning data
        :type train_data: outp of `citation_extractor.Utils.IO.load_brat_data`
        :rtype: a `pandas.Dataframe` with columns: ["count", "prob"] and
                indexed by URN of author/work/NIL entity.
        """
        LOGGER.info("Computing entity probability...")

        idx = pd.Index(self._kb_work_urns).append(
            pd.Index(self._kb_author_urns)
        )
        freqs = pd.DataFrame(
            index=idx.append(pd.Index([NIL_URN])),
            dtype='float64'
        )
        freqs['count'] = 0
        freqs['prob'] = 0.0
        M = train_data.shape[0]
        N = freqs.shape[0]
        MN = M + N

        # go through train data and update the frequency count table
        for mid, mrow in train_data.iterrows():
            urn = mrow.urn_clean
            freqs.loc[urn, 'count'] += 1

        # transform freq counts into probabilities
        for mid, mrow in freqs.iterrows():
            c = int(mrow['count'])
            p = float(c + 1) / MN
            freqs.loc[mid, 'prob'] = p

        LOGGER.info("Done computing entity probability.")
        return freqs

    def _compute_entity_mention_probability(self, train_data):
        """Probability of an entity to be referred to by a given mention.

        :param train_data: a dataframe with the traning data
        :type train_data: outp of `citation_extractor.Utils.IO.load_brat_data`
        :rtype: a `pandas.Dataframe` with as many columns as the entities in
                the training data, and as many rows as the mentions.
        """
        mentions = set(train_data.surface_norm_dots.tolist())
        entities = self._kb_author_urns + self._kb_work_urns + [NIL_URN]

        counts = pd.DataFrame(
            index=mentions,
            columns=entities,
            dtype='float64'
        ).fillna(0.0)

        for mid, mrow in train_data.iterrows():
            s = mrow.surface_norm_dots
            e = mrow.urn_clean
            counts.loc[s, e] += 1.0

        return counts.divide(counts.sum(axis=1), axis=0).fillna(0.0)

    def _compute_mention_entity_probability(self, train_data):
        """Probability of a mention to be connected to a given entity.

        :param train_data: a dataframe with the traning data
        :type train_data: outp of `citation_extractor.Utils.IO.load_brat_data`
        :rtype: a `pandas.Dataframe` with as many columns as the entities in
                the training data, and as many rows as the mentions.
        """
        mentions = set(train_data.surface_norm_dots.tolist())
        entities = self._kb_author_urns + self._kb_work_urns + [NIL_URN]

        counts = pd.DataFrame(
            index=mentions,
            columns=entities,
            dtype='float64'
        ).fillna(0.0)

        for mid, mrow in train_data.iterrows():
            s = mrow.surface_norm_dots
            e = mrow.urn_clean
            counts.loc[s, e] += 1.0

        return counts.divide(counts.sum(axis=0), axis=1).fillna(0.0)

    def extract_nil(self, m_type, m_scope, feature_dicts):
        """Extract NIL-related features from entity mention.

        :param m_type:
        :type m_type: str
        :param m_scope:
        :type m_scope: str
        :param feature_dicts:
        :type feature_dicts: dict
        :rtype: dict

        """
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

    def extract(
            self,
            m_surface,
            m_scope,
            m_type,
            m_title_mentions,
            m_title,
            m_doc_text,
            m_prev_entities,
            candidate_urn
            ):
        """TODO."""
        LOGGER.info('Extracting features for ...')

        feature_vector = {}

        string_sim = True
        context_sim = True
        prob = True

        # The mention is an author name and searching for an author
        if m_type == AUTHOR_TYPE and not m_scope:

            surf = m_surface
            names = self._kb_norm_authors.loc[
                candidate_urn,
                'norm_names_clean'
            ]
            abbr = self._kb_norm_authors.loc[
                candidate_urn,
                'norm_abbr'
            ]

            if string_sim:
                self.add_string_similarities(
                    feature_vector,
                    'ANS_ss_',
                    surf,
                    names
                )
                self.add_abbr_match(
                    feature_vector,
                    'ANS_ss_',
                    surf,
                    abbr
                )

            if context_sim:

                self.add_tfidf_similarity(
                    feature_vector,
                    'ANS_cxt_tfidf_',
                    candidate_urn,
                    m_doc_text
                )

                self.add_title_similarities(
                    feature_vector,
                    'ANS_cxt_title_',
                    m_title_mentions,
                    m_title,
                    candidate_urn
                )

                self.add_other_mentions_string_similarities(
                    feature_vector,
                    'ANS_cxt_om_',
                    m_surface,
                    m_scope,
                    m_type,
                    m_prev_entities,
                    candidate_urn
                )

            if prob:
                self.add_prior_prob(feature_vector, 'ANS_prob_entity_prior', candidate_urn)
                # self.add_me_prob(feature_vector, 'ANS_prob_m_given_e', surf, candidate_urn)
                # self.add_em_prob(feature_vector, 'ANS_prob_e_given_m', surf, candidate_urn)

        # The mention is an author name but searching for a work
        elif m_type == AUTHOR_TYPE and m_scope:

            surf = m_surface
            aurn = self._kb_norm_works.loc[candidate_urn, 'author']
            names = self._kb_norm_authors.loc[aurn, 'norm_names_clean']
            abbr = self._kb_norm_authors.loc[aurn, 'norm_abbr']

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
            names = self._kb_norm_works.loc[candidate_urn, 'norm_titles_clean']
            abbr = self._kb_norm_works.loc[candidate_urn, 'norm_abbr']

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
            names = self._kb_norm_works.loc[candidate_urn, 'norm_titles_clean']
            abbr = self._kb_norm_works.loc[candidate_urn, 'norm_abbr']
            aurn = self._kb_norm_works.loc[candidate_urn, 'author']
            anames = self._kb_norm_authors.loc[aurn, 'norm_names_clean']
            aabbr = self._kb_norm_authors.loc[aurn, 'norm_abbr']

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
        feature_vector[feat_prefix + 'kb_abbr_ex_match'] = StringSimilarity.exact_match(surf, abbr)

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
            feature_vector[feat_prefix + 'a_ex_match_and_w_ex_match'] = StringSimilarity.exact_match(s1,
                                                                                                     anames) and StringSimilarity.exact_match(
                s2, names)

            # ~author ~work
            feature_vector[feat_prefix + 'a_fuz_match_and_w_fuz_match'] = StringSimilarity.fuzzy_match(s1,
                                                                                                       anames) and StringSimilarity.fuzzy_match(
                s2,
                names)

        # s1 s2.
        matched = re.match(ur'^([a-z]+) ([a-z]+)\.$', surf)
        if matched:
            s1, s2 = matched.group(1), matched.group(2)

            # author work.
            feature_vector[feat_prefix + 'a_ex_match_and_w_abbr_match'] = StringSimilarity.exact_match(s1,
                                                                                                       anames_words) and StringSimilarity.abbreviation_match(
                s2, names_words)

            # author ?.
            feature_vector[feat_prefix + 'a_ex_match_and_unknown'] = StringSimilarity.exact_match(s1, anames_words)

            # replace u with v
            if u'u' in surf:
                s1, s2 = s1.replace(u'u', u'v'), s2.replace(u'u', u'v')
                feature_vector[feat_prefix + 'a_ex_match_and_w_abbr_match_ureplaced'] = StringSimilarity.exact_match(s1,
                                                                                                                     anames_words) and StringSimilarity.abbreviation_match(
                    s2, names_words)

        # s1. s2.
        matched = re.match(ur'^([a-z]+)\. ([a-z]+)\.$', surf)
        if matched:
            s1, s2 = matched.group(1), matched.group(2)

            # auth. work.
            feature_vector[feat_prefix + 'a_abbr_match_and_w_abbr_match'] = StringSimilarity.abbreviation_match(s1,
                                                                                                                anames_words) and StringSimilarity.abbreviation_match(
                s2, names_words)

            # work1. work2.
            feature_vector[feat_prefix + 'w_abbr_seq_match'] = StringSimilarity.abbreviation_sequence_match([s1, s2],
                                                                                                            names)

            # auth1. auth2.
            feature_vector[feat_prefix + 'a_abbr_seq_match'] = StringSimilarity.abbreviation_sequence_match([s1, s2],
                                                                                                            anames)

            # wrk1. wrk2.
            feature_vector[feat_prefix + 'w_abbr_sparse_match'] = StringSimilarity.abbreviation_sparse_match(s1 + s2,
                                                                                                             names)

            # ath1. ath2.
            feature_vector[feat_prefix + 'a_abbr_sparse_match'] = StringSimilarity.abbreviation_sparse_match(s1 + s2,
                                                                                                             anames)

            # ?. work.
            feature_vector[feat_prefix + 'unknown_and_w_abbr_match'] = StringSimilarity.abbreviation_match(s2,
                                                                                                           names_words)

            # ?. auth.
            feature_vector[feat_prefix + 'unknown_and_a_abbr_match'] = StringSimilarity.abbreviation_match(s2,
                                                                                                           anames_words)

            # work. ?
            feature_vector[feat_prefix + 'w_abbr_match_and_unknown'] = StringSimilarity.abbreviation_match(s1,
                                                                                                           names_words)

            # auth. ?
            feature_vector[feat_prefix + 'a_abbr_match_and_unknown'] = StringSimilarity.abbreviation_match(s1,
                                                                                                           anames_words)

        # s1. s2. s3.
        matched = re.match(ur'^([a-z]+)\. ([a-z]+)\. ([a-z]+)\.$', surf)
        if matched:
            s1, s2, s3 = matched.group(1), matched.group(2), matched.group(3)

            # auth. work1. work2.
            feature_vector[feat_prefix + 'a_abbr_match_and_w_abbr_seq_match'] = StringSimilarity.abbreviation_match(s1,
                                                                                                                    anames) and StringSimilarity.abbreviation_sequence_match(
                [s2, s3], names)

        # s1 s2. s3.
        matched = re.match(ur'^([a-z]+) ([a-z]+)\. ([a-z]+)\.$', surf)
        if matched:
            s1, s2, s3 = matched.group(1), matched.group(2), matched.group(3)

            # author work1. work2.
            feature_vector[feat_prefix + 'a_ex_match_and_w_abbr_seq_match'] = StringSimilarity.exact_match(s1,
                                                                                                           anames) and StringSimilarity.abbreviation_sequence_match(
                [s2, s3], names)

            # author work1. work2.
            feature_vector[feat_prefix + 'a_ex_match_and_w_abbr_seq_match'] = StringSimilarity.exact_match(s1,
                                                                                                           anames_words) and StringSimilarity.abbreviation_sequence_match(
                [s2, s3], names)

        # s1 s2 s3.
        matched = re.match(ur'^([a-z]+) ([a-z]+) ([a-z]+)\.$', surf)
        if matched:
            s1, s2, s3 = matched.group(1), matched.group(2), matched.group(3)

            # author1 author2 work.
            feature_vector[feat_prefix + 'aa_ex_match_and_w_abbr_match'] = StringSimilarity.exact_match(
                u' '.join([s1, s2]),
                anames) and StringSimilarity.abbreviation_match(
                s3, names)

            # author1 author2 work.
            feature_vector[feat_prefix + 'aa_fuz_match_and_w_abbr_match'] = StringSimilarity.fuzzy_match(
                u' '.join([s1, s2]),
                anames) and StringSimilarity.abbreviation_match(
                s3, names)

            # author work1 work2.
            feature_vector[feat_prefix + 'a_ex_match_and_w_ex_match_and_w_abbr_match'] = StringSimilarity.exact_match(
                s1,
                anames) and StringSimilarity.abbreviation_sequence_match(
                [s2, s3], names)

            # author work1 work2.
            feature_vector[
                feat_prefix + 'a_ex_match_nwords_and_w_ex_match_and_w_abbr_match'] = StringSimilarity.exact_match(s1,
                                                                                                                  anames_words) and StringSimilarity.abbreviation_sequence_match(
                [s2, s3], names)

        # s1 s2 s3
        matched = re.match(ur'^([a-z]+) ([a-z]+) ([a-z]+)$', surf)
        if matched:
            s1, s2, s3 = matched.group(1), matched.group(2), matched.group(3)

            # author work1 work2
            feature_vector[feat_prefix + 'a_ex_match_and_ww_ex_match'] = StringSimilarity.exact_match(s1,
                                                                                                      anames) and StringSimilarity.exact_match(
                u' '.join([s2, s3]), names)

    def add_string_similarities(self, feature_vector, feat_prefix, surf, names):

        surf = self.clean_surface(surf)
        surf_words = surf.split()
        names_words = set(u' '.join(names).split())

        if len(surf_words) == 1:
            feat_prefix = feat_prefix + '1w_'

            # Exact match
            feature_vector[feat_prefix + 'ex_match'] = StringSimilarity.exact_match(surf, names)
            feature_vector[feat_prefix + 'ex_match_nwords'] = StringSimilarity.exact_match(surf, names_words)

            # Fuzzy match
            feature_vector[feat_prefix + 'fuz_match'] = StringSimilarity.fuzzy_match(surf, names)
            feature_vector[feat_prefix + 'fuz_match_max'] = StringSimilarity.fuzzy_match_max(surf, names)
            feature_vector[feat_prefix + 'fuz_match_nwords'] = StringSimilarity.fuzzy_match(surf, names_words)

            # Fuzzy initial letters match
            feature_vector[feat_prefix + 'fuz_init_match'] = StringSimilarity.fuzzy_initial_match(surf, names)
            feature_vector[feat_prefix + 'fuz_init_match_max'] = StringSimilarity.fuzzy_initial_match_max(surf, names)
            feature_vector[feat_prefix + 'fuz_init_match_nwords'] = StringSimilarity.fuzzy_initial_match(surf,
                                                                                                         names_words)

            # Fuzzy phonetic match
            # feature_vector[feat_prefix + 'fuz_phon_match'] = StringSimilarity.fuzzy_phonetic_match(surf, names)
            # feature_vector[feat_prefix + 'fuz_phon_match_max'] = StringSimilarity.fuzzy_phonetic_match_max(surf, names)
            # feature_vector[feat_prefix + 'fuz_phon_match_nwords'] = StringSimilarity.fuzzy_phonetic_match(surf, names_words)

            # Match rating comparison
            # feature_vector[feat_prefix + 'fuz_mrc_match'] = fuzzy_mrc(surf, names)

            # Match acronym
            feature_vector[feat_prefix + 'acronym_match'] = StringSimilarity.acronym_match(surf, names)

            # Match abbreviations
            matched = re.match(ur'^([a-z]+)\.$', surf)
            if matched:
                s1 = matched.group(1)
                feature_vector[feat_prefix + 'abbr_match'] = StringSimilarity.abbreviation_match(s1, names)
                feature_vector[feat_prefix + 'abbr_match_nwords'] = StringSimilarity.abbreviation_match(s1, names_words)
                feature_vector[feat_prefix + 'abbr_match_nwords_sparse'] = StringSimilarity.abbreviation_sparse_match(
                    s1, names_words)


        elif len(surf_words) == 2:
            feat_prefix = feat_prefix + '2w_'

            # Exact match
            feature_vector[feat_prefix + 'ex_match'] = StringSimilarity.exact_match(surf, names)
            feature_vector[feat_prefix + 'ex_match_nwords'] = StringSimilarity.exact_match_swords(surf_words,
                                                                                                  names_words)
            feature_vector[feat_prefix + 'ex_match_nwords_any'] = StringSimilarity.exact_match_swords_any(surf_words,
                                                                                                          names_words)

            # Fuzzy match
            feature_vector[feat_prefix + 'fuz_match'] = StringSimilarity.fuzzy_match(surf, names)
            feature_vector[feat_prefix + 'fuz_match_max'] = StringSimilarity.fuzzy_match_max(surf, names)
            feature_vector[feat_prefix + 'fuz_match_nwords'] = StringSimilarity.fuzzy_match_swords(surf_words,
                                                                                                   names_words)
            feature_vector[feat_prefix + 'fuz_match_nwords_any'] = StringSimilarity.fuzzy_match_swords_any(surf_words,
                                                                                                           names_words)

            # Fuzzy initial letters match in s_words vs n_words 'aaaz bbbz' match 'aaax bb cccc' and 'aaax', 'bbbx'
            feature_vector[feat_prefix + 'fuz_init_match'] = StringSimilarity.fuzzy_initial_match(surf, names)
            feature_vector[feat_prefix + 'fuz_init_match_max'] = StringSimilarity.fuzzy_initial_match_max(surf, names)
            feature_vector[feat_prefix + 'fuz_init_match_nwords'] = StringSimilarity.fuzzy_initial_match_swords(
                surf_words, names_words)

            # Fuzzy phonetic match in s_words vs n_words u'theophylakt simokates' match u'theophylactus simocatta'
            # feature_vector[feat_prefix + 'fuz_phon_match'] = StringSimilarity.fuzzy_phonetic_match(surf, names)
            # feature_vector[feat_prefix + 'fuz_phon_match_max'] = StringSimilarity.fuzzy_phonetic_match_max(surf, names)
            # feature_vector[feat_prefix + 'fuz_phon_match_nwords'] = StringSimilarity.fuzzy_phonetic_match_swords(surf_words, names_words)


        elif len(surf_words) == 3:
            feat_prefix = feat_prefix + '3w_'

            # Exact match
            feature_vector[feat_prefix + 'ex_match'] = StringSimilarity.exact_match(surf, names)
            feature_vector[feat_prefix + 'ex_match_nwords'] = StringSimilarity.exact_match_swords(surf_words,
                                                                                                  names_words)
            feature_vector[feat_prefix + 'ex_match_nwords_any'] = StringSimilarity.exact_match_swords_any(surf_words,
                                                                                                          names_words)

            # Fuzzy match
            feature_vector[feat_prefix + 'fuz_match'] = StringSimilarity.fuzzy_match(surf, names)
            feature_vector[feat_prefix + 'fuz_match_max'] = StringSimilarity.fuzzy_match_max(surf, names)
            feature_vector[feat_prefix + 'fuz_match_nwords'] = StringSimilarity.fuzzy_match_swords(surf_words,
                                                                                                   names_words)
            feature_vector[feat_prefix + 'fuz_match_nwords_any'] = StringSimilarity.fuzzy_match_swords_any(surf_words,
                                                                                                           names_words)

            # Fuzzy initial letters match
            feature_vector[feat_prefix + 'fuz_init_match'] = StringSimilarity.fuzzy_initial_match(surf, names)
            feature_vector[feat_prefix + 'fuz_init_match_max'] = StringSimilarity.fuzzy_initial_match_max(surf, names)
            feature_vector[feat_prefix + 'fuz_init_match_nwords'] = StringSimilarity.fuzzy_initial_match_swords(
                surf_words, names_words)

            # Fuzzy phonetic match
            # feature_vector[feat_prefix + 'fuz_phon_match'] = StringSimilarity.fuzzy_phonetic_match(surf, names)
            # feature_vector[feat_prefix + 'fuz_phon_match_max'] = StringSimilarity.fuzzy_phonetic_match_max(surf, names)
            # feature_vector[feat_prefix + 'fuz_phon_match_nwords'] = StringSimilarity.fuzzy_phonetic_match_swords(surf_words, names_words)


        elif len(surf_words) > 3:
            feat_prefix = feat_prefix + '3+w_'

            surf = self.remove_words_shorter_than(surf, 2)
            surf_words = surf.split()

            # Exact match
            feature_vector[feat_prefix + 'ex_match'] = StringSimilarity.exact_match(surf, names)
            feature_vector[feat_prefix + 'ex_match_nwords'] = StringSimilarity.exact_match_swords(surf_words,
                                                                                                  names_words)
            feature_vector[feat_prefix + 'ex_match_nwords_any'] = StringSimilarity.exact_match_swords_any(surf_words,
                                                                                                          names_words)

            # Fuzzy match
            feature_vector[feat_prefix + 'fuz_match'] = StringSimilarity.fuzzy_match(surf, names)
            feature_vector[feat_prefix + 'fuz_match_max'] = StringSimilarity.fuzzy_match_max(surf, names)
            feature_vector[feat_prefix + 'fuz_match_nwords'] = StringSimilarity.fuzzy_match_swords(surf_words,
                                                                                                   names_words)
            feature_vector[feat_prefix + 'fuz_match_nwords_any'] = StringSimilarity.fuzzy_match_swords_any(surf_words,
                                                                                                           names_words)

            # Fuzzy initial letters match in s_words vs n_words 'aaaz bbbz' match 'aaax bb cccc' and 'aaax', 'bbbx'
            feature_vector[feat_prefix + 'fuz_init_match'] = StringSimilarity.fuzzy_initial_match(surf, names)
            feature_vector[feat_prefix + 'fuz_init_match_max'] = StringSimilarity.fuzzy_initial_match_max(surf, names)
            feature_vector[feat_prefix + 'fuz_init_match_nwords'] = StringSimilarity.fuzzy_initial_match_swords(
                surf_words, names_words)

            # Fuzzy phonetic match in s_words vs n_words u'theophylakt simokates' match u'theophylactus simocatta'
            # feature_vector[feat_prefix + 'fuz_phon_match'] = StringSimilarity.fuzzy_phonetic_match(surf, names)
            # feature_vector[feat_prefix + 'fuz_phon_match_max'] = StringSimilarity.fuzzy_phonetic_match_max(surf, names)
            # feature_vector[feat_prefix + 'fuz_phon_match_nwords'] = StringSimilarity.fuzzy_phonetic_match_swords(surf_words, names_words)

    def add_other_mentions_string_similarities(self, feature_vector, feat_prefix, surf, scope, mtype, other_mentions,
                                               candidate_urn):

        # other_mentions = [(type, surface, scope), ...]
        anames, wnames = [], []
        if candidate_urn in self._kb_norm_authors.index:
            anames = self._kb_norm_authors.loc[candidate_urn, 'norm_names_clean']
            wnames = []
            for w in self._kb_norm_authors.loc[candidate_urn, 'works']:
                for wn in self._kb_norm_works.loc[w, 'norm_titles_clean']:
                    wnames.append(wn)

        elif candidate_urn in self._kb_norm_works.index:
            wnames = self._kb_norm_works.loc[candidate_urn, 'norm_titles_clean']
            aurn = self._kb_norm_works.loc[candidate_urn, 'author']
            anames = self._kb_norm_authors.loc[aurn, 'norm_names_clean']

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

        if candidate_urn in self._kb_norm_authors.index:
            names = self._kb_norm_authors.loc[candidate_urn, 'norm_names_clean']
            wnames = []
            for w in self._kb_norm_authors.loc[candidate_urn, 'works']:
                for wn in self._kb_norm_works.loc[w, 'norm_titles_clean']:
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

        elif candidate_urn in self._kb_norm_works.index:
            names = self._kb_norm_works.loc[candidate_urn, 'norm_titles_clean']
            aurn = self._kb_norm_works.loc[candidate_urn, 'author']
            anames = self._kb_norm_authors.loc[aurn, 'norm_names_clean']

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
        return s1 == s2 or StringSimilarity.levenshtein_distance_norm(s1, s2) >= 0.9

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
        if self._tfidf is not None:
            urn_target = candidate_urn
            if urn_target in self._kb_norm_works.index:
                urn_target = self._kb_norm_works.loc[urn_target, 'author']

            tfidf_scores = []
            for lang in LANGUAGES:
                tfidf_scores.append(self.text_similarity(doc_text, urn_target, lang))

            # feature_vector[feat_prefix + 'max'] = max(tfidf_scores)
            feature_vector[feat_prefix + 'avg'] = avg(tfidf_scores)

    def add_prior_prob(self, feature_vector, feat_prefix, candidate_urn):
        if self._prior_prob is not None:
            feature_vector[feat_prefix] = self._prior_prob.loc[candidate_urn, 'prob']

    def add_me_prob(self, feature_vector, feat_prefix, surf, candidate_urn):
        if self._me_prob is not None and surf in self._me_prob.index:
            feature_vector[feat_prefix] = self._me_prob.loc[surf, candidate_urn]

    def add_em_prob(self, feature_vector, feat_prefix, surf, candidate_urn):
        if self._em_prob is not None and surf in self._em_prob.index:
            feature_vector[feat_prefix] = self._em_prob.loc[surf, candidate_urn]

    def text_similarity(self, text, urn, lang):
        if urn not in self._tfidf[lang]['urn_to_index'].keys():
            return 0.0

        text_vector = self._tfidf[lang]['vectorizer'].transform([text])
        urn_doc_index = self._tfidf[lang]['urn_to_index'][urn]
        urn_doc_vector = self._tfidf[lang]['matrix'][urn_doc_index]

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
