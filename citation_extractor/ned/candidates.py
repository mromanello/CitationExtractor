"""Candidates Generation code related to the NED step."""

# -*- coding: utf-8 -*-


from __future__ import print_function
import logging

from citation_extractor.Utils.strmatching import StringSimilarity, StringUtils
from citation_extractor.ned import AUTHOR_TYPE, WORK_TYPE, REFAUWORK_TYPE
import pandas as pd
import multiprocessing

LOGGER = logging.getLogger(__name__)


# TODO: we should define precise data-structures,
# if we use pandas dataframes we should also enforce a schema and also define
# column names as variables

class CandidatesGenerator(object):
    def __init__(self, kb, mention_surface_is_normalized=True, fuzzy_threshold=0.7):

        # TODO: compute kb norm authors/works (get Dataframe schema)
        self._kb_norm_authors = None
        self._kb_norm_works = None

        self._surf_is_norm = mention_surface_is_normalized
        self._fuzzy_threshold = fuzzy_threshold

        self._authors_dict_names = None
        self._authors_dict_abbr = None
        self._works_dict_names = None
        self._works_dict_abbr = None

        self._build_name_abbr_dict()

    def _build_name_abbr_dict(self):

        def update_df_list(df, row, col, value):
            if row not in df.index:
                df.loc[row, col] = set()
            df.loc[row, col].add(value)

        authors_dict_names = pd.DataFrame(dtype='object', columns=['urns', 'len'])
        authors_dict_abbr = pd.DataFrame(dtype='object', columns=['urns', 'len'])
        for aid, arow in self._kb_norm_authors.iterrows():
            for n in arow.norm_names_clean:
                if n == u'': continue
                update_df_list(authors_dict_names, n, 'urns', aid)
            for a in arow.norm_abbr:
                if a == u'': continue
                update_df_list(authors_dict_abbr, a, 'urns', aid)
        for aid, arow in authors_dict_names.iterrows():
            arow.len = len(arow.urns)
        for aid, arow in authors_dict_abbr.iterrows():
            arow.len = len(arow.urns)
        authors_dict_names.sort_values('len', ascending=False, inplace=True)
        authors_dict_abbr.sort_values('len', ascending=False, inplace=True)

        works_dict_names = pd.DataFrame(dtype='object', columns=['urns', 'len'])
        works_dict_abbr = pd.DataFrame(dtype='object', columns=['urns', 'len'])
        for aid, arow in self._kb_norm_works.iterrows():
            for n in arow.norm_titles_clean:
                if n == u'': continue
                update_df_list(works_dict_names, n, 'urns', aid)
            for a in arow.norm_abbr:
                if a == u'': continue
                update_df_list(works_dict_abbr, a, 'urns', aid)
        for aid, arow in works_dict_names.iterrows():
            arow.len = len(arow.urns)
        for aid, arow in works_dict_abbr.iterrows():
            arow.len = len(arow.urns)
        works_dict_names.sort_values('len', ascending=False, inplace=True)
        works_dict_abbr.sort_values('len', ascending=False, inplace=True)

        self._authors_dict_names = authors_dict_names
        self._authors_dict_abbr = authors_dict_abbr
        self._works_dict_names = works_dict_names
        self._works_dict_abbr = works_dict_abbr

    def generate_candidates(self, mention_surface, mention_type, mention_scope):
        """
        Generate the candidates for a mention
        :param mention_surface: A unicode string
        :param mention_type: A string representing the type of the mention
        :param mention_scope: A string representing the scope of the mention
        :return: A set of strings representing the ids of the canidates
        """

        def many_to_many_exact_match(surf, name):
            return len(set(surf.split()) & set(name.split())) > 0

        def many_to_many_startswith(surf, name):
            for s in surf.split():
                for n in name.split():
                    if n.startswith(s):
                        return True
            return False

        def is_exact_acronym(acronym, name):
            return acronym == u''.join(map(lambda w: w[0], name.split()))

        def search_names(names_dict, surf):
            results = set()
            for n, row in names_dict.iterrows():
                if n == surf:
                    results.update(row.urns)
                elif StringSimilarity.levenshtein_distance_norm(n, surf) >= 0.7:
                    results.update(row.urns)
                elif many_to_many_exact_match(surf, n):
                    results.update(row.urns)
                elif many_to_many_startswith(surf, n):
                    results.update(row.urns)
                elif is_exact_acronym(surf, n):
                    results.update(row.urns)

            return results

        def search_abbr(abbr_dict, surf):
            results = set()
            for n, row in abbr_dict.iterrows():
                if n == surf:
                    results.update(row.urns)
                elif many_to_many_exact_match(surf, n):
                    results.update(row.urns)
            return results

        candidates = set()
        norm_surface = mention_surface
        if not self._surf_is_norm:
            norm_surface = StringUtils.normalize(mention_surface)

        if mention_type == AUTHOR_TYPE and mention_scope:
            for aurn in search_names(self._authors_dict_names, norm_surface):
                candidates.update(self._kb_norm_authors.loc[aurn, 'works'])
            for aurn in search_abbr(self._authors_dict_abbr, norm_surface):
                candidates.update(self._kb_norm_authors.loc[aurn, 'works'])

        if mention_type == AUTHOR_TYPE and not mention_scope:
            candidates.update(search_names(self._authors_dict_names, norm_surface))
            candidates.update(search_abbr(self._authors_dict_abbr, norm_surface))

        if mention_type == WORK_TYPE:
            candidates.update(search_names(self._works_dict_names, norm_surface))
            candidates.update(search_abbr(self._works_dict_abbr, norm_surface))

        if mention_type == REFAUWORK_TYPE:
            for aurn in search_names(self._authors_dict_names, norm_surface):
                candidates.update(self._kb_norm_authors.loc[aurn, 'works'])
            for aurn in search_abbr(self._authors_dict_abbr, norm_surface):
                candidates.update(self._kb_norm_authors.loc[aurn, 'works'])
            candidates.update(search_names(self._works_dict_names, norm_surface))
            candidates.update(search_abbr(self._works_dict_abbr, norm_surface))

        return candidates

    def generate_candidates_parallel(self, mentions, nb_processes=10):
        """
        Generate the candidates for a list of mentions by using parallel computation.
        :param mentions: A pandas Dataframe containing the mentions.
        :param nb_processes: The number of processes to be used for the computation.
        :return: A list of tuples (mention id, set of candidates)
        """

        def dispatch_per_process(params):
            mention_id = params[0]
            arguments = params[1]
            candidates = self.generate_candidates(*arguments)
            return mention_id, candidates

        params = []
        for m_id, row in mentions.iterrows():
            args = [row['surface_norm'], row['type'], row['scope']]
            params.append((m_id, args))

        pool = multiprocessing.Pool(processes=nb_processes)
        candidates = pool.map(dispatch_per_process, params)
        pool.terminate()

        return candidates
