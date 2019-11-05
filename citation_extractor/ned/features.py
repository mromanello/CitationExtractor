# -*- coding: utf-8 -*-
# author: Matteo Filipponi

"""Code related to feature extraction for thr NED step."""

from __future__ import print_function
import pdb  # TODO remove when done with development
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from stop_words import safe_get_stop_words

from citation_extractor.Utils.strmatching import DictUtils
from citation_extractor.Utils.strmatching import StringSimilarity
from citation_extractor.Utils.strmatching import StringUtils
from citation_extractor.ned import AUTHOR_TYPE, WORK_TYPE, REFAUWORK_TYPE
from citation_extractor.ned import NIL_URN, LANGUAGES, PREPS
from citation_extractor.Utils.extra import avg

import time
import pkg_resources
import pandas as pd
import logging
import os
import re

LOGGER = logging.getLogger(__name__)


class FeatureExtractor(object):
    """Extract features for an <entity-mention, entity-candidate> couple."""

    def __init__(self, kb=None, train_data=None, **kwargs):
        """Initialize an instance of FeatureExtractor.

        Optional kwargs:
            - `kb_norm_authors`
            - `kb_norm_works`
            - `prior_prob`
            - `mention_entity_prob`
            - `entity_mention_prob`

        :param kb: an instance of HuCit KnowledgeBase
        :param train_data:
        """
        LOGGER.info('Initializing Feature Extractor')

        # TODO: if all necessary pickle files are available
        # just initialise using them

        if kb is not None:
            self._kb_norm_authors = self._normalize_kb_authors(kb)
            self._kb_norm_works = self._normalize_kb_works(kb)

        elif 'kb_norm_authors' in kwargs and 'kb_norm_works' in kwargs:
            self._kb_norm_authors = kwargs['kb_norm_authors']
            self._kb_norm_works = kwargs['kb_norm_works']

        else:
            raise Exception

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

    # TODO: remove method once moved to dask for parallelization
    def extract_unpack(self, extract_arguments):
        """Helper method to be used in parallel computation. It simply calls the FeatureExtractor.extract()
        method by unpacking the passed arguments.

        :param extract_arguments: a dictionary or a tuple containing all the necessary arguments to call FeatureExtractor.extract()
        :type extract_arguments: dict or tuple

        :return: the output of the FeatureExtractor.extract() method
        :rtype: dict
        """
        if type(extract_arguments) == tuple:
            return self.extract(*extract_arguments)
        elif type(extract_arguments) == dict:
            return self.extract(**extract_arguments)
        else:
            raise TypeError('extract_arguments type must be tuple or dict')

    def extract(self, m_surface, m_scope, m_type, m_title_mentions, m_title, m_doc_text, m_other_mentions, candidate_urn):
        """Extract features about the similarity between an entity-mention and a candidate entity.

        :param m_surface: the surface form of the mention
        :type m_surface: unicode
        :param m_scope: the scope of the mention (could be None)
        :type m_scope: unicode
        :param m_type: the type of the mention (AAUTHOR, AWORK, REFAUWORK)
        :type: m_type: str
        :param m_title_mentions: the mentions extracted from the title
        :type m_title_mentions: list of tuples [(m_type, m_surface), ...]
        :param m_title: the title of the document containing the mention
        :type m_title: unicode
        :param m_doc_text: the text of the document containing the mention
        :type m_doc_text: unicode
        :param m_other_mentions: the other mentions extracted from the same document
        :type m_other_mentions: list of triples [(m_type, m_surface, m_scope), ...]
        :param candidate_urn: the URN of the candidate entity
        :type candidate_urn: str

        :return: the features extracted from the mention and a candidate entity
        :rtype: dict
        """
        LOGGER.debug(
            u'Extracting features for {} {} ({})'.format(
                m_surface,
                m_scope,
                m_type
            )
        )

        feature_vector = {}

        # TODO: set from config file
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
                self._add_string_similarities(
                    feature_vector,
                    'ANS_ss_',
                    surf,
                    names
                )
                self._add_abbr_match(
                    feature_vector,
                    'ANS_ss_',
                    surf,
                    abbr
                )

            if context_sim:
                self._add_tfidf_similarity(
                    feature_vector,
                    'ANS_cxt_tfidf_',
                    candidate_urn,
                    m_doc_text
                )

                self._add_title_similarities(
                    feature_vector,
                    'ANS_cxt_title_',
                    m_title_mentions,
                    m_title,
                    candidate_urn
                )

                self._add_other_mentions_string_similarities(
                    feature_vector,
                    'ANS_cxt_om_',
                    m_surface,
                    m_scope,
                    m_type,
                    m_other_mentions,
                    candidate_urn
                )

            if prob:
                self._add_prior_prob(feature_vector, 'ANS_prob_entity_prior', candidate_urn)
                # self._add_me_prob(feature_vector, 'ANS_prob_m_given_e', surf, candidate_urn)
                # self._add_em_prob(feature_vector, 'ANS_prob_e_given_m', surf, candidate_urn)

        # The mention is an author name but searching for a work
        elif m_type == AUTHOR_TYPE and m_scope:

            surf = m_surface
            aurn = self._kb_norm_works.loc[candidate_urn, 'author']
            names = self._kb_norm_authors.loc[aurn, 'norm_names_clean']
            abbr = self._kb_norm_authors.loc[aurn, 'norm_abbr']

            if string_sim:
                self._add_string_similarities(
                    feature_vector,
                    'AS_ss_',
                    surf, names
                )
                self._add_abbr_match(feature_vector, 'AS_ss_', surf, abbr)

            if context_sim:
                self._add_tfidf_similarity(
                    feature_vector,
                    'AS_cxt_tfidf_',
                    candidate_urn,
                    m_doc_text
                )
                self._add_title_similarities(
                    feature_vector,
                    'AS_cxt_title_',
                    m_title_mentions,
                    m_title,
                    candidate_urn
                )
                self._add_other_mentions_string_similarities(
                    feature_vector,
                    'AS_cxt_om_',
                    m_surface,
                    m_scope,
                    m_type,
                    m_other_mentions,
                    candidate_urn
                )

            if prob:
                self._add_prior_prob(feature_vector, 'AS_prob_entity_prior', candidate_urn)
                # self._add_me_prob(feature_vector, 'AS_prob_m_given_e', surf, candidate_urn)
                self._add_em_prob(feature_vector, 'AS_prob_e_given_m', surf, candidate_urn)

        # The mention is a work name and searching for a work
        elif m_type == WORK_TYPE:

            surf = m_surface
            names = self._kb_norm_works.loc[candidate_urn, 'norm_titles_clean']
            abbr = self._kb_norm_works.loc[candidate_urn, 'norm_abbr']

            if string_sim:
                self._add_string_similarities(
                    feature_vector,
                    'W_ss_',
                    surf,
                    names
                )
                self._add_abbr_match(feature_vector, 'W_ss_', surf, abbr)

            if context_sim:
                self._add_tfidf_similarity(
                    feature_vector,
                    'W_cxt_tfidf_',
                    candidate_urn,
                    m_doc_text
                )
                self._add_title_similarities(
                    feature_vector,
                    'W_cxt_title_',
                    m_title_mentions,
                    m_title,
                    candidate_urn
                )
                self._add_other_mentions_string_similarities(
                    feature_vector,
                    'W_cxt_om_',
                    m_surface,
                    m_scope,
                    m_type,
                    m_other_mentions,
                    candidate_urn
                )

            if prob:
                self._add_prior_prob(
                    feature_vector,
                    'W_prob_entity_prior',
                    candidate_urn
                )
                # self._add_me_prob(feature_vector, 'W_prob_m_given_e', surf, candidate_urn)
                # self._add_em_prob(feature_vector, 'W_prob_e_given_m', surf, candidate_urn)

        # The mention is an author name, work name or mixed
        # and searching for a work
        elif m_type == REFAUWORK_TYPE:

            surf = m_surface
            names = self._kb_norm_works.loc[candidate_urn, 'norm_titles_clean']
            abbr = self._kb_norm_works.loc[candidate_urn, 'norm_abbr']
            aurn = self._kb_norm_works.loc[candidate_urn, 'author']
            anames = self._kb_norm_authors.loc[aurn, 'norm_names_clean']
            aabbr = self._kb_norm_authors.loc[aurn, 'norm_abbr']

            if string_sim:
                self._add_mixed_string_similarities(
                    feature_vector,
                    'R_ss_mix_',
                    surf,
                    names,
                    anames
                )
                self._add_string_similarities(
                    feature_vector,
                    'R_ss_w_',
                    surf,
                    names
                )
                self._add_abbr_match(feature_vector, 'R_ss_w_', surf, abbr)
                self._add_string_similarities(
                    feature_vector,
                    'R_ss_a_',
                    surf, anames
                )
                self._add_abbr_match(
                    feature_vector,
                    'R_ss_a_',
                    surf,
                    aabbr
                )

            if context_sim:
                self._add_tfidf_similarity(
                    feature_vector,
                    'R_cxt_tfidf_',
                    candidate_urn,
                    m_doc_text
                )
                self._add_title_similarities(
                    feature_vector,
                    'R_cxt_title_',
                    m_title_mentions,
                    m_title,
                    candidate_urn
                )
                self._add_other_mentions_string_similarities(
                    feature_vector,
                    'R_cxt_om_',
                    m_surface,
                    m_scope,
                    m_type,
                    m_other_mentions,
                    candidate_urn
                )

            if prob:
                self._add_prior_prob(
                    feature_vector,
                    'R_prob_entity_prior',
                    candidate_urn
                )
                # self._add_me_prob(feature_vector, 'R_prob_m_given_e', surf, candidate_urn)
                self._add_em_prob(
                    feature_vector,
                    'R_prob_e_given_m',
                    surf,
                    candidate_urn
                )

        else:
            LOGGER.error('Unknown mention type: {}'.format(m_type))

        return feature_vector

    def extract_nil(self, m_type, m_scope, feature_dicts):
        """Extract NIL-related features from entity mention.

        :param m_type: the type of the mention (AAUTHOR, AWORK, REFAUWORK)
        :type m_type: str
        :param m_scope: the scope of the mention (could be None)
        :type m_scope: unicode
        :param feature_dicts: the features of all the candidates for a given mention
        :type feature_dicts: list of dicts

        :return: the features for the NIL candidate entity extracted from the other candidates
        :rtype: dict
        """
        LOGGER.debug('Extracting NIL features for ...')
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

    ############################
    # Initialization functions #
    ############################

    def _normalize_kb_authors(self, knowledge_base):
        """Pre-compute cleaning and normalization of author names/abbr.

        :param knowledge_base: an instance of HuCit KnowledgeBase
        :type knowledge_base: `knowledge_base.KnowledgeBase`
        :rtype: `pd.DataFrame`
        """
        cols = [
            'names',
            'norm_names',
            'norm_names_clean',
            'abbr',
            'norm_abbr',
            'works'
        ]
        LOGGER.info("Normalizing KB authors...")
        df_norm_authors = pd.DataFrame(dtype='object', columns=cols)

        i = 0
        t = len(knowledge_base.get_authors())

        for author in knowledge_base.get_authors():
            i += 1
            print('{}/{}'.format(i, t), end='\r')

            a_urn = str(author.get_urn())
            if a_urn == 'None':
                continue

            a_names = author.get_names()
            a_norm_names = []
            a_norm_names_clean = []
            for lang, name in a_names:
                if type(name) != unicode:
                    print('!!!', a_urn, repr(name), type(name))
                norm_name = StringUtils.normalize(name, lang=lang)
                a_norm_names.append((lang, norm_name))
                a_norm_names_clean.append(
                    StringUtils.remove_words_shorter_than(norm_name, 2)
                )
            df_norm_authors.loc[a_urn, 'names'] = a_names
            df_norm_authors.loc[a_urn, 'norm_names'] = a_norm_names
            df_norm_authors.loc[a_urn, 'norm_names_clean'] = a_norm_names_clean

            a_abbr = author.get_abbreviations()
            a_norm_abbr = []
            for abbr in a_abbr:
                if type(abbr) != unicode:
                    print('!!!', a_urn, repr(abbr), type(abbr))
                a_norm_abbr.append(StringUtils.normalize(abbr))
            df_norm_authors.loc[a_urn, 'abbr'] = a_abbr
            df_norm_authors.loc[a_urn, 'norm_abbr'] = a_norm_abbr

            a_works = []
            for w in author.get_works():
                a_works.append(str(w.get_urn()))
            df_norm_authors.loc[a_urn, 'works'] = a_works

        return df_norm_authors

    def _normalize_kb_works(self, knowledge_base):
        """Pre-compute cleaning and normalization of work titles/abbr.

        :param knowledge_base: an instance of HuCit KnowledgeBase
        :type knowledge_base: `knowledge_base.KnowledgeBase`
        :rtype: `pd.DataFrame`
        """
        cols = [
            'titles',
            'norm_titles',
            'norm_titles_clean',
            'abbr',
            'norm_abbr',
            'author'
        ]
        LOGGER.info("Normalizing KB works...")
        df_norm_works = pd.DataFrame(dtype='object', columns=cols)

        i = 0
        t = len(knowledge_base.get_works())
        for work in knowledge_base.get_works():
            i += 1
            print('{}/{}'.format(i, t), end='\r')

            w_urn = str(work.get_urn())
            if w_urn == 'None':
                continue

            w_titles = work.get_titles()
            w_norm_titles = []
            w_norm_titles_clean = []
            for lang, title in w_titles:
                if type(title) != unicode:
                    print('!!!', w_urn, repr(title), type(title))
                norm_name = StringUtils.normalize(title)
                w_norm_titles.append((lang, norm_name))
                w_norm_titles_clean.append(
                    StringUtils.remove_words_shorter_than(norm_name, 2)
                )
            df_norm_works.loc[w_urn, 'titles'] = w_titles
            df_norm_works.loc[w_urn, 'norm_titles'] = w_norm_titles
            df_norm_works.loc[w_urn, 'norm_titles_clean'] = w_norm_titles_clean

            w_abbr = work.get_abbreviations()
            w_norm_abbr = []
            for abbr in w_abbr:
                if type(abbr) != unicode:
                    print('!!!', w_urn, repr(abbr), type(abbr))
                w_norm_abbr.append(
                    StringUtils.normalize(abbr)
                )
            df_norm_works.loc[w_urn, 'abbr'] = w_abbr
            df_norm_works.loc[w_urn, 'norm_abbr'] = w_norm_abbr

            w_author = str(work.author.get_urn())
            df_norm_works.loc[w_urn, 'author'] = w_author

        return df_norm_works

    def _compute_tfidf_matrix(self, base_dir=None):
        """Compute the TF-IDF matrix for a set of documents divided per-language.

        :param base_dir: the directory containing the plain text of Wikipedia
            related to the ancient authors in the KB.
        :type base_dir: str
        :rtype: a dictionary of dictionaries. The first dictionary has
            languages as keys; second dictionary has two keys ('matrix'
            and `vectorizer`), and as values the raw tf-idf matrix and
            the TfidfVectorizer used.
        """
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

        idx = pd.Index(self._kb_norm_works.index).append(
            pd.Index(self._kb_norm_authors.index)
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
        entities = list(self._kb_norm_authors.index) + \
                   list(self._kb_norm_works.index) + [NIL_URN]

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
        entities = list(self._kb_norm_authors.index) + \
                   list(self._kb_norm_works.index) + [NIL_URN]

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

    ########################################
    # String similarity features functions #
    ########################################

    def _add_string_similarities(
        self,
        feature_vector,
        feat_prefix,
        surf,
        names
    ):
        """Add several string similarity features.

        Features represent string similarities between the surface form and the
        names of the candidate work/author

        :param feature_vector: the feature vector
        :type feature_vector: dict
        :param feat_prefix: the prefix of the name of this feature
        :type feat_prefix: str
        :param surf: the surface form of the mention
        :type surf: unicode
        :param names: the names of the candidate
        :type names: list of unicode
        """

        surf = StringUtils.clean_surface(surf)
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

            surf = StringUtils.remove_words_shorter_than(surf, 2)
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

    def _add_mixed_string_similarities(self, feature_vector, feat_prefix, surf, names, anames):
        """Add several features: string similarities between mixed combinations of author and work names

        :param feature_vector: the feature vector
        :type feature_vector: dict
        :param feat_prefix: the prefix of the name of this feature
        :type feat_prefix: str
        :param surf: the surface form of the mention
        :type surf: unicode
        :param names: the names of the candidate work
        :type names: list of unicode
        :param anames: the names of the author of the candidate work
        :type anames: list of unicode
        """
        surf = StringUtils.clean_surface(surf)
        surf_words = surf.split()
        names_words = set(u' '.join(names).split())
        anames_words = set(u' '.join(anames).split())

        # s1 s2
        matched = re.match(ur'^([a-z]+) ([a-z]+)$', surf)
        if matched:
            s1, s2 = matched.group(1), matched.group(2)

            # author work
            exact_match_s1 = StringSimilarity.exact_match(s1, anames)
            exact_match_s2 = StringSimilarity.exact_match(s2, names)
            feat_name = 'a_ex_match_and_w_ex_match'
            feature_vector[
                feat_prefix + feat_name
                ] = exact_match_s1 and exact_match_s2

            # ~author ~work
            fuzzy_match_s1 = StringSimilarity.fuzzy_match(s1, anames)
            fuzzy_match_s2 = StringSimilarity.fuzzy_match(s2, names)
            feat_name = 'a_fuz_match_and_w_fuz_match'
            feature_vector[
                feat_prefix + feat_name
                ] = fuzzy_match_s1 and fuzzy_match_s2

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

    def _add_abbr_match(self, feature_vector, feat_prefix, surf, abbr):
        """Add the feature: the surface exactly match the abbreviation

        :param feature_vector: the feature vector
        :type feature_vector: dict
        :param feat_prefix: the prefix of the name of this feature
        :type feat_prefix: str
        :param surf: the surface form of the mention
        :type surf: unicode
        :param abbr: a potential abbreviation of the surface
        :type abbr: unicode
        """
        surf = surf.replace(u'.', u'')
        feature_vector[feat_prefix + 'kb_abbr_ex_match'] = StringSimilarity.exact_match(surf, abbr)

    #########################################
    # Context similarity features functions #
    #########################################

    def _add_title_similarities(self, feature_vector, feat_prefix, title_mentions, title, candidate_urn):
        """Add several features: matching presence of the candidate in the title

        If the candidate is an author, then its names and the names of its works are searched in the title.
        If the candidate is a work, then its names and the names of its author are searched in the title.

        :param feature_vector: the feature vector
        :type dict
        :param feat_prefix: the prefix of the name of this feature
        :type feat_prefix: str
        :param title_mentions: the mentions extracted from the title
        :type title_mentions: list of tuples [(m_type, m_surface), ...]
        :param title: the title of the document containing the mention
        :type title: unicode
        :param candidate_urn: the URN of the candidate entity
        :type candidate_urn: str
        """

        if title is None:
            return

        stripped_title = StringUtils.remove_words_shorter_than(title, 3)

        if candidate_urn in self._kb_norm_authors.index:
            names = self._kb_norm_authors.loc[candidate_urn, 'norm_names_clean']
            wnames = []
            for w in self._kb_norm_authors.loc[candidate_urn, 'works']:
                for wn in self._kb_norm_works.loc[w, 'norm_titles_clean']:
                    wnames.append(wn)

            feature_vector[feat_prefix + 'auth_in_title'] = self._names_in_title(names, stripped_title)
            feature_vector[feat_prefix + 'auth_in_title_pw'] = self._names_in_title_perword(names, stripped_title)
            feature_vector[feat_prefix + 'authworks_in_title'] = self._names_in_title(wnames, stripped_title)
            feature_vector[feat_prefix + 'authworks_in_title_pw'] = self._names_in_title_perword(wnames, stripped_title)

            feature_vector[feat_prefix + 'auth_in_extr_title'] = self._names_in_extracted_title(names, title_mentions)
            feature_vector[feat_prefix + 'auth_in_extr_title_pw'] = self._names_in_extracted_title_perword(names, title_mentions)
            feature_vector[feat_prefix + 'authworks_in_extr_title'] = self._names_in_extracted_title(wnames, title_mentions)
            feature_vector[feat_prefix + 'authworks_in_extr_title_pw'] = self._names_in_extracted_title_perword(wnames, title_mentions)

        elif candidate_urn in self._kb_norm_works.index:
            names = self._kb_norm_works.loc[candidate_urn, 'norm_titles_clean']
            aurn = self._kb_norm_works.loc[candidate_urn, 'author']
            anames = self._kb_norm_authors.loc[aurn, 'norm_names_clean']

            feature_vector[feat_prefix + 'work_in_title'] = self._names_in_title(names, stripped_title)
            feature_vector[feat_prefix + 'work_in_title_pw'] = self._names_in_title_perword(names, stripped_title)
            feature_vector[feat_prefix + 'workauth_in_title'] = self._names_in_title(anames, stripped_title)
            feature_vector[feat_prefix + 'workauth_in_title_pw'] = self._names_in_title_perword(anames, stripped_title)

            feature_vector[feat_prefix + 'work_in_extr_title'] = self._names_in_extracted_title(names, title_mentions)
            feature_vector[feat_prefix + 'work_in_extr_title_pw'] = self._names_in_extracted_title_perword(names, title_mentions)
            feature_vector[feat_prefix + 'workauth_in_extr_title'] = self._names_in_extracted_title(anames, title_mentions)
            feature_vector[feat_prefix + 'workauth_in_extr_title_pw'] = self._names_in_extracted_title_perword(anames, title_mentions)

    def _add_tfidf_similarity(self, feature_vector, feat_prefix, candidate_urn, doc_text):
        """ Add the feature: text similarity between the mention's document and the candidate's document(s).

        :param feature_vector: the feature vector
        :type feature_vector: dict
        :param feat_prefix: the prefix of the name of this feature
        :type feat_prefix: str
        :param candidate_urn: the URN of the candidate entity
        :type candidate_urn: str
        :param doc_text: the text of the document containing the mention
        :type doc_text: unicode
        """
        if self._tfidf is not None:
            urn_target = candidate_urn
            if urn_target in self._kb_norm_works.index:
                urn_target = self._kb_norm_works.loc[urn_target, 'author']

            tfidf_scores = []
            for lang in LANGUAGES:
                tfidf_scores.append(self._text_similarity(doc_text, urn_target, lang))

            # feature_vector[feat_prefix + 'max'] = max(tfidf_scores)
            feature_vector[feat_prefix + 'avg'] = avg(tfidf_scores)

    def _add_other_mentions_string_similarities(self, feature_vector, feat_prefix, surf, scope, mtype, other_mentions, candidate_urn):
        """Add several features: string similarities between the surface form of the other mentions present in the same
        context (but different from the target mention) and the names of the candidate work/author

        :param feature_vector: the feature vector
        :type dict
        :param feat_prefix: the prefix of the name of this feature
        :type feat_prefix: str
        :param surf: the surface form of the mention
        :type surf: unicode
        :param scope: the scope of the meniton
        :type scope: unicode
        :param mtype: the type of the mention
        :type mtype: str
        :param other_mentions: the other mentions extracted from the same document
        :type other_mentions: list of triples [(m_type, m_surface, m_scope), ...]
        :param candidate_urn:
        :type candidate_urn: str
        """

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

        other_mentions = filter(lambda (typ, srf): srf != surf, set(map(lambda (typ, srf, scp): (typ, srf), other_mentions)))
        amatch, wmatch, rmatch = [], [], []
        for om_type, om_surf in other_mentions:

            if om_type == AUTHOR_TYPE:
                tmp_vector = {}
                self._add_string_similarities(tmp_vector, 'tmp', om_surf, anames)
                amatch.append(DictUtils._dict_contains_match(tmp_vector))

            if om_type == WORK_TYPE:
                tmp_vector = {}
                self._add_string_similarities(tmp_vector, 'tmp', om_surf, wnames)
                wmatch.append(DictUtils._dict_contains_match(tmp_vector))

            if om_type == REFAUWORK_TYPE:
                tmp_vector = {}
                self._add_string_similarities(tmp_vector, 'tmp', om_surf, anames)
                self._add_string_similarities(tmp_vector, 'tmp', om_surf, wnames)
                self._add_mixed_string_similarities(tmp_vector, 'tmp', om_surf, wnames, anames)
                rmatch.append(DictUtils._dict_contains_match(tmp_vector))

        feature_vector[feat_prefix + 'author_match'] = any(amatch)
        feature_vector[feat_prefix + 'author_match_nb'] = float(sum(amatch)) / max(len(other_mentions), 1)
        feature_vector[feat_prefix + 'work_match'] = any(wmatch)
        feature_vector[feat_prefix + 'work_match_nb'] = float(sum(wmatch)) / max(len(other_mentions), 1)
        feature_vector[feat_prefix + 'refauwork_match'] = any(rmatch)
        feature_vector[feat_prefix + 'refauwork_match_nb'] = float(sum(rmatch)) / max(len(other_mentions), 1)

    ##################################
    # Probability features functions #
    ##################################

    def _add_prior_prob(self, feature_vector, feat_prefix, candidate_urn):
        """Add the feature: probability of the candidate occurring in the train set

        :param feature_vector: the feature vector
        :type feature_vector: dict
        :param feat_prefix: the prefix of the name of this feature
        :type feat_prefix: str
        :param candidate_urn: the URN of the candidate entity
        :type candidate_urn: str
        """
        if self._prior_prob is not None:
            feature_vector[feat_prefix] = self._prior_prob.loc[candidate_urn, 'prob']

    def _add_me_prob(self, feature_vector, feat_prefix, surf, candidate_urn):
        """Add the feature: probability of the mention to be connected to the candidate

        :param feature_vector: the feature vector
        :type feature_vector: dict
        :param feat_prefix: the prefix of the name of this feature
        :type feat_prefix: str
        :param surf: the surface form of the mention
        :type surf: unicode
        :param candidate_urn: the URN of the candidate entity
        :type candidate_urn: str
        """
        if self._me_prob is not None and surf in self._me_prob.index:
            feature_vector[feat_prefix] = self._me_prob.loc[surf, candidate_urn]

    def _add_em_prob(self, feature_vector, feat_prefix, surf, candidate_urn):
        """Add the feature: probability of the candidate to be referred to by the mention

        :param feature_vector: the feature vector
        :type feature_vector: dict
        :param feat_prefix: the prefix of the name of this feature
        :type feat_prefix: str
        :param surf: the surface form of the mention
        :type surf: unicode
        :param candidate_urn: the URN of the candidate entity
        :type candidate_urn: str
        """
        if self._em_prob is not None and surf in self._em_prob.index:
            feature_vector[feat_prefix] = self._em_prob.loc[surf, candidate_urn]

    ####################
    # Helper functions #
    ####################

    def _names_in_title(self, names, title):
        """Find a name in a title.

        :param names: a list of names
        :type names: list of unicode
        :param title: a title of a document
        :type title: unicode

        :return: True if a name is matched in the title, False otherwise
        :rtype: bool
        """
        for name in names:
            for word in title.split():
                if StringSimilarity.qexact_match(name, word):
                    return True
        return False

    def _names_in_title_perword(self, names, title):
        """Find a word of a name in a title

        :param names: a list of names
        :type names: list of unicode
        :param title: a title of a document
        :type title: unicode

        :return: True if a word of a name is matched in the title, False otherwise
        :rtype: bool
        """
        names = StringUtils.split_names(names)
        return self._names_in_title(names, title)

    def _names_in_extracted_title(self, names, title_mentions):
        """Find a name in the mentions extracted from a title

        :param names: a list of names
        :type names: list of unicode
        :param title_mentions: the mentions extracted from the title
        :type title_mentions: list of tuples [(m_type, m_surface), ...]

        :return: True if a name is matched against a mention extracted from the title, False otherwise
        :rtype: bool
        """
        for m_type, m_surface in title_mentions:
            for name in names:
                if StringSimilarity.qexact_match(m_surface, name):
                    return True
        return False

    def _names_in_extracted_title_perword(self, names, title_mentions):
        """Find a word of a name in the mentions extracted from a title

        :param names: a list of names
        :type names: list of unicode
        :param title_mentions: the mentions extracted from the title
        :type title_mentions: list of tuples [(m_type, m_surface), ...]

        :return: True if a word of a name is matched against a mention extracted from the title, False otherwise
        :rtype: bool
        """
        names = StringUtils.split_names(names)
        return self._names_in_extracted_title(names, title_mentions)

    def _text_similarity(self, text, urn, lang):
        """Return the cosine similarity between a text and the document describing an entity in a given language

        :param text: the target text
        :type text: unicode
        :param urn: the URN of an entity
        :type urn: str
        :param lang: the language of the text (en, es, it, fr, de)
        :type lang: str

        :return: the cosine similarity between the input text and the entity document
        :rtype: float
        """
        if urn not in self._tfidf[lang]['urn_to_index'].keys():
            return 0.0

        text_vector = self._tfidf[lang]['vectorizer'].transform([text])
        urn_doc_index = self._tfidf[lang]['urn_to_index'][urn]
        urn_doc_vector = self._tfidf[lang]['matrix'][urn_doc_index]

        return cosine_similarity(text_vector, urn_doc_vector).flatten()[0]
