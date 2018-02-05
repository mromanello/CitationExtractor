# -*- coding: utf-8 -*-
# author: Matteo Filipponi

"""Candidates Generation code related to the NED step."""

from __future__ import print_function
import logging

from citation_extractor.Utils.strmatching import StringSimilarity, StringUtils
from citation_extractor.ned import AUTHOR_TYPE, WORK_TYPE, REFAUWORK_TYPE
import pandas as pd
# import multiprocessing
from dask import compute, delayed
from dask.diagnostics import ProgressBar

LOGGER = logging.getLogger(__name__)


# TODO: we should define precise data-structures,
# if we use pandas dataframes we should also enforce a schema and also define
# column names as variables

class CandidatesGenerator(object):
    """Generate entity candidates for a given mention."""

    def __init__(
        self,
        kb,
        mention_surface_is_normalized=True,
        fuzzy_threshold=0.7,
        **kwargs
    ):
        """Initialize an instance of CandidatesGenerator.

        :param kb: an instance of HuCit KnowledgeBase
        :type kb: knowledge_base.KnowledgeBase
        :param mention_surface_is_normalized: specify whether mention surfaces
            are already normalized (default is True)
        :type mention_surface_is_normalized: bool
        :param fuzzy_threshold: specify the threshold to be used in fuzzy
            string matching (default is 0.7)
        :type fuzzy_threshold: float
        """

        if 'kb_norm_authors' in kwargs and 'kb_norm_works' in kwargs:
            self._kb_norm_authors = kwargs['kb_norm_authors']
            self._kb_norm_works = kwargs['kb_norm_works']
        else:
            # TODO: make static the normalization methods of FeatureExtractor
            raise Exception

        self._surf_is_norm = mention_surface_is_normalized
        self._fuzzy_threshold = fuzzy_threshold

        self._authors_dict_names = None
        self._authors_dict_abbr = None
        self._works_dict_names = None
        self._works_dict_abbr = None

        self._build_name_abbr_dict()

    def _build_name_abbr_dict(self):
        """Map names/abbrev. to sets of entity URNs with those names/abbrev.

        This function initializes (in-place) the dictionaries that map author
        names to their URNs, author abbreviations to their URN, work names to
        their URNs and work abbreviations to their URNs.
        For example if two author entities share the same abbreviation.
        then the URNs of the two entities will be both present in the set
        mapped by that abbreviation key.
        """
        LOGGER.info("Starting to build name/abbreviation indexes...")

        # Helper function: update a dataframe entry set with a new value
        def update_df_list(df, row, col, value):
            if row not in df.index:
                df.loc[row, col] = set()
            df.loc[row, col].add(value)

        authors_dict_names = pd.DataFrame(
            dtype='object', columns=['urns', 'len']
        )
        authors_dict_abbr = pd.DataFrame(
            dtype='object', columns=['urns', 'len']
        )

        # first, process all authors
        for aid, arow in self._kb_norm_authors.iterrows():
            for n in arow.norm_names_clean:
                if n == u'':
                    continue
                update_df_list(authors_dict_names, n, 'urns', aid)
            for a in arow.norm_abbr:
                if a == u'':
                    continue
                update_df_list(authors_dict_abbr, a, 'urns', aid)
        for aid, arow in authors_dict_names.iterrows():
            arow.len = len(arow.urns)
        for aid, arow in authors_dict_abbr.iterrows():
            arow.len = len(arow.urns)
        authors_dict_names.sort_values('len', ascending=False, inplace=True)
        authors_dict_abbr.sort_values('len', ascending=False, inplace=True)

        works_dict_names = pd.DataFrame(
            dtype='object', columns=['urns', 'len']
        )
        works_dict_abbr = pd.DataFrame(
            dtype='object', columns=['urns', 'len']
        )

        # second, process all works
        for aid, arow in self._kb_norm_works.iterrows():
            for n in arow.norm_titles_clean:
                if n == u'':
                    continue
                update_df_list(works_dict_names, n, 'urns', aid)
            for a in arow.norm_abbr:
                if a == u'':
                    continue
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

        LOGGER.info("Done with creating name/abbreviation indexes.")

    def generate_candidates(self, mention_surface, mention_type, mention_scope):
        """Generate a set of entity candidates for a given mention.

        :param mention_surface: surface form of the mention
        :type mention_surface: unicode
        :param mention_type: type of the mention (AAUTHOR, AWORK, REFAUWORK)
        :type mention_type: str
        :param mention_scope: the scope of the mention (could be None)
        :type mention_scope: unicode

        :return: the URNs of the candidate entities
        :rtype: list of str
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
            candidates.update(
                search_names(self._authors_dict_names, norm_surface)
            )
            candidates.update(
                search_abbr(self._authors_dict_abbr, norm_surface)
            )

        if mention_type == WORK_TYPE:
            candidates.update(
                search_names(self._works_dict_names, norm_surface)
            )
            candidates.update(search_abbr(self._works_dict_abbr, norm_surface))

        if mention_type == REFAUWORK_TYPE:
            for aurn in search_names(self._authors_dict_names, norm_surface):
                candidates.update(self._kb_norm_authors.loc[aurn, 'works'])
            for aurn in search_abbr(self._authors_dict_abbr, norm_surface):
                candidates.update(self._kb_norm_authors.loc[aurn, 'works'])
            candidates.update(
                search_names(self._works_dict_names, norm_surface)
            )
            candidates.update(search_abbr(self._works_dict_abbr, norm_surface))

        return list(candidates)

    def generate_candidates_parallel(self, mentions, nb_processes=10):
        # TODO: ask MatteoF is should be removed?
        """Generate the entity candidates for a mention by using parallel computation.

        :param mentions: a pandas Dataframe containing the mentions (schema: surface_norm, type, scope)
        :type mentions: pandas.DataFrame
        :param nb_processes: The number of processes to be used for the computation (default is 10)
        :type nb_processes: int

        :return: the URNs of the candidate entities for each mention [(mention_id, set_of_candidates), ...]
        :rtype: list of (str, list of str)
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
