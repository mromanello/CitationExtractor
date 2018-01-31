# -*- coding: utf-8 -*-
# author: Matteo Romanello, matteo.romanello@gmail.com

"""
# TODO: remove after development

%load_ext autoreload
%autoreload 2

import logging
import tabulate
from citation_extractor.Utils.IO import init_logger
from citation_extractor.Utils.strmatching import StringUtils
init_logger(loglevel=logging.DEBUG)

import codecs, pickle
with codecs.open("citation_extractor/data/pickles/kb_data.pkl","rb") as pickle_file:
    kb_data = pickle.load(pickle_file)

#with codecs.open("citation_extractor/data/pickles/kb_data.pkl","wb") as pickle_file:
#    pickle.dump(kb_data, pickle_file)

# replace this with a fresh dataframe
with codecs.open("citation_extractor/data/pickles/testset_dataframe.pkl","rb") as pickle_file:
    testset_gold_df = pickle.load(pickle_file)

from knowledge_base import KnowledgeBase
#kb = KnowledgeBase("../KnowledgeBase/knowledge_base/config/virtuoso.ini")
kb = KnowledgeBase("/Users/rromanello/Documents/ClassicsCitations/hucit_kb/knowledge_base/config/virtuoso_local.ini")


author_names = kb.author_names
author_abbreviations = kb.author_abbreviations
work_titles = kb.work_titles
work_abbreviations = kb.work_abbreviations

kb_data = {
        "author_names": {key: StringUtils.normalize(author_names[key]) for key in author_names}
        , "author_abbreviations": {key: StringUtils.normalize(author_abbreviations[key]) for key in author_abbreviations}
        , "work_titles": {key: StringUtils.normalize(work_titles[key]) for key in work_titles}
        , "work_abbreviations": {key: StringUtils.normalize(work_abbreviations[key]) for key in work_abbreviations}
}

with codecs.open("citation_extractor/data/pickles/kb_data.pkl","rb") as pickle_file:
    kb_data = pickle.load(pickle_file)

from citation_extractor.ned import CitationMatcher
cm = CitationMatcher(kb
                        , fuzzy_matching_entities=True
                        , fuzzy_matching_relations=True
                        , min_distance_entities=4
                        , max_distance_entities=7
                        , distance_relations=4
                        , **kb_data)
cm._disambiguate_entity("Iliad's", "AWORK")
%time y = cm._disambiguate_entity("Ovid", "AAUTHOR")

for i, instance in testset_gold_df.iterrows():
    result = cm.disambiguate(instance['surface'], instance["type"], instance["scope"])
    testset_gold_df.loc[i]["predicted_urn"] = result.urn

print(tabulate.tabulate(testset_gold_df.head(20)[["urn_clean","predicted_urn"]]))

"""

from __future__ import print_function
import re
import sys
import pdb
import logging
from operator import itemgetter
from collections import namedtuple
from nltk.metrics import edit_distance
from pyCTS import CTS_URN
from citation_extractor.extra.pysuffix.suffixIndexers import DictValuesIndexer
from citation_parser import CitationParser
from citation_extractor.pipeline import NIL_URN
from citation_extractor.Utils.strmatching import *

global logger
logger = logging.getLogger(__name__)


# TODO: not sure about `scope`
# also not sure what's the best place where to put it (as other submodules will need to import it)
# perhaps `ned.__init__.py` ?
Result = namedtuple('DisambiguationResult','mention, entity_type, scope, urn')

"""
class DisambiguationNotFound(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)
    def __str__(self):
        return repr(self.message)
"""


def longest_common_substring(s1, s2):
    """
    Taken from https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Longest_common_substring#Python
    """
    m = [[0] * (1 + len(s2)) for i in xrange(1 + len(s1))]
    longest, x_longest = 0, 0
    for x in xrange(1, 1 + len(s1)):
       for y in xrange(1, 1 + len(s2)):
           if s1[x - 1] == s2[y - 1]:
               m[x][y] = m[x - 1][y - 1] + 1
               if m[x][y] > longest:
                   longest = m[x][y]
                   x_longest = x
           else:
               m[x][y] = 0
    return s1[x_longest - longest: x_longest]


def select_lcs_match(citation_string, matches, n_guess=1):
    """TODO."""
    # iterate and get what's the lowest ed_score
    # then keep only the matches with lowest (best) score
    # then keep the one with longest common string
    lowest_score = 1000

    for m in matches:
        score = m[2]
        if score < lowest_score:
            lowest_score = score

    filtered_matches = [m for m in matches if m[2] == lowest_score]

    best_match = ("", None)

    if(lowest_score > 0):
        for match in filtered_matches:
            lcs = longest_common_substring(match[1], citation_string)
            if(len(lcs) > len(best_match[0])):
                best_match = (lcs, match)
        match = [best_match[1]]
        logger.debug("Longest_common_substring selected %s out of %s" % (
            match,
            filtered_matches
        ))
    else:
        # TODO: use context here to disambiguate
        match = matches[:n_guess]
    return match


class CitationMatcher(object): #TODO: rename => FuzzyCitationMatcher
    """
    TODO
    docstring for CitationMatcher

    """
    def __init__(self, knowledge_base=None, fuzzy_matching_entities=False, fuzzy_matching_relations=False,
                min_distance_entities=1, max_distance_entities=3, distance_relations=3, **kwargs):

        self.fuzzy_match_entities = fuzzy_matching_entities
        self.fuzzy_match_relations = fuzzy_matching_relations

        self.min_distance_entities = min_distance_entities if fuzzy_matching_entities else None
        self.max_distance_entities = max_distance_entities if fuzzy_matching_entities else None
        self.distance_relations = distance_relations if fuzzy_matching_relations else None

        self._kb = knowledge_base

        if 'author_names' in kwargs and 'work_titles' in kwargs \
            and 'work_abbreviations' in kwargs and 'author_abbreviations' in kwargs:

            self._author_names = kwargs["author_names"]
            self._author_abbreviations = kwargs["author_abbreviations"]
            self._work_titles = kwargs["work_titles"]
            self._work_abbreviations = kwargs["work_abbreviations"]

        else:

            logger.info("Initialising CitationMatcher...")
            self._citation_parser = CitationParser()

            logger.info("Fetching author names from the KB...")
            author_names = knowledge_base.author_names
            self._author_names = {key: StringUtils.normalize(author_names[key]) for key in author_names}

            logger.info("Done. Fetching work titles from the KB...")
            work_titles = knowledge_base.work_titles
            self._work_titles = {key:StringUtils.normalize(work_titles[key]) for key in work_titles}

            logger.info("Done. Fetching author abbreviations from the KB...")
            author_abbreviations = knowledge_base.author_abbreviations
            self._author_abbreviations = {key:StringUtils.normalize(author_abbreviations[key]) for key in author_abbreviations}

            logger.info("Done. Fetching work abbreviations from the KB...")
            work_abbreviations = knowledge_base.work_abbreviations
            self._work_abbreviations = {key:StringUtils.normalize(work_abbreviations[key]) for key in work_abbreviations}

            logger.info("Done. Now let's index all this information.")

        self._author_idx, self._author_abbr_idx, self._work_idx, self._work_abbr_idx = self._initialise_indexes()
        logger.info(self.settings)

    def _initialise_indexes(self):
        """
        Creates suffix arrays for efficient retrieval.

        TODO: convert to lowercase before indexing

        """
        try:
            logger.info("Start indexing author names...")
            author_idx = DictValuesIndexer(self._author_names)
            logger.info("Done. Start indexing author abbreviations...")
            author_abbr_idx = DictValuesIndexer(self._author_abbreviations)
            logger.info("Done. Start indexing work titles...")
            work_idx = DictValuesIndexer(self._work_titles)
            logger.info("Done. Start indexing work abbreviations...")
            work_abbr_idx = DictValuesIndexer(self._work_abbreviations)
            logger.info("Done with indexing.")
            return author_idx, author_abbr_idx, work_idx, work_abbr_idx
        except Exception, e:
            raise e

    @property
    def settings(self):
        """
        Prints to the stdout the settings of the CitationMatcher.
        """

        prolog = "%s initialisation settings:" % self.__class__

        if self.fuzzy_match_entities:
            entity_matching_settings = "\t-Entity matching: fuzzy matching=%s; min distance threshold=%i; max distance threshold=%i" % \
                                        (self.fuzzy_match_entities, self.min_distance_entities, self.max_distance_entities)
        else:
            entity_matching_settings = "\t-Entity matching: fuzzy matching=%s" % self.fuzzy_match_entities

        if self.fuzzy_match_relations:
            relation_matching_settings = "\t-Relation matching: fuzzy matching=%s; edit distance threshold=%i" % \
                                        (self.fuzzy_match_relations, self.distance_relations)
        else:
            relation_matching_settings = "\t-Relation matching: fuzzy matching=%s" % self.fuzzy_match_relations

        knowledge_base_extent = "\t-Extent of the KnowledgeBase: %i author abbreviations, %i author_names, %i work abbreviations, %i work titles." % \
                                (len(self._author_abbreviations)
                                , len(self._author_names)
                                , len(self._work_abbreviations)
                                , len(self._work_titles))

        return "\n".join((prolog, entity_matching_settings, relation_matching_settings, knowledge_base_extent))

    # TODO: remove and add to the `citation_parser`
    def _format_scope(self, scope_dictionary):
        """
        Args:
            scope_dictionary:
                {u'start': [u'1', u'100']}
        returns:
            string
        """
        if(scope_dictionary.has_key("end")):
            #is range
            return "%s-%s"%(".".join(scope_dictionary["start"]),".".join(scope_dictionary["end"]))
        else:
            #is not range
            return ".".join(scope_dictionary["start"])

    def _consolidate_result(
        self,
        urn_string,
        citation_string,
        entity_type,
        scope
    ):
        urn = CTS_URN(urn_string)

        # check: does the URN have a scope but is missing the work element
        if(urn.work is None):
            # if so, try to get the opus maximum from the KB
            opmax = self._kb.get_opus_maximum_of(urn)

            if(opmax is not None):
                logger.debug("%s is opus maximum of %s" % (opmax, urn))
                urn = CTS_URN("{}".format(opmax.get_urn()))

        return Result(citation_string, entity_type, scope, urn)

    def _disambiguate_relation(
        self,
        citation_string,
        entity_type,
        scope,
        n_guess=1
    ):
        """Disambiguate a relation.

        :citation_string: e.g. "Hom. Il.
        :scope: e.g. "1,100"
        :return: a named tuple  (see `Result`)
        """
        match = None

        # citation string has one single token
        if len(citation_string.split(" ")) == 1:

            match = self.matches_work(citation_string, self.fuzzy_match_relations, self.distance_relations)

            # TODO this is problematic
            # should be: match is None or match does not contain at least one entry with distance=0
            zero_distance_match = False
            if match is not None:
                for m in match:
                    if m[2] == 0:
                        zero_distance_match = True

            logger.debug("[%s %s] zero distance match is %s, match = %s" % (citation_string, scope, zero_distance_match, match))

            if match is None or not zero_distance_match:
                match = self.matches_author(citation_string, self.fuzzy_match_relations, self.distance_relations)
            if match is not None:
                if(len(match) <= n_guess):
                    match = match[:n_guess]
                else:
                    match = select_lcs_match(citation_string, match, n_guess)

                for urn_string, label, score in match:
                    result = self._consolidate_result(
                        urn_string,
                        citation_string,
                        entity_type,
                        scope
                    )
                    return result

        # citation string has two tokens
        elif(len(citation_string.split(" ")) == 2):
            tok1, tok2 = citation_string.split(" ")

            # case 2: tok1 and tok2 are author
            match = self.matches_author(citation_string, self.fuzzy_match_relations, self.distance_relations)

            if match is not None:
                if(len(match) <= n_guess):
                    match = match[:n_guess]
                else:
                    match = select_lcs_match(citation_string, match, n_guess)

                for urn_string, label, score in match:
                    result = self._consolidate_result(
                        urn_string,
                        citation_string,
                        entity_type,
                        scope
                    )
                    return result
            else:
                # case 3: tok1 and tok2 are work
                match = self.matches_work(
                    citation_string,
                    self.fuzzy_match_relations,
                    self.distance_relations
                )
                if match is not None:
                    if(len(match) <= n_guess):
                        match = match[:n_guess]
                    else:
                        match = select_lcs_match(citation_string, match, n_guess)

                    for urn_string, label, score in match:
                        result = self._consolidate_result(
                            urn_string,
                            citation_string,
                            entity_type,
                            scope
                        )
                        return result

            # case 1: tok1 is author and tok2 is work
            match_tok1 = self.matches_author(tok1, self.fuzzy_match_relations, self.distance_relations)
            match_tok2 = self.matches_work(tok2, self.fuzzy_match_relations, self.distance_relations)

            if(match_tok1 is not None and match_tok2 is not None):

                for id1, label1, score1 in match_tok1:
                    for id2, label2, score2 in match_tok2:
                        work = self._kb.get_resource_by_urn(id2)

                        if id1 == str(work.author.get_urn()):
                            match = [(id2, label2, score2)]
                            return Result(citation_string, entity_type, scope, CTS_URN(id2))
                        else:
                            logger.debug("The combination: {} and {} was ruled out".format(id1, id2))

        # citation string has more than two tokens
        elif(len(citation_string.split(" ")) > 2):
            match = self.matches_author(citation_string, self.fuzzy_match_relations, self.distance_relations)
        else:
            logger.error("This case is not handled properly: {}".format(
                citation_string
            ))
            raise

        # return only n_guess results
        if match is None or len(match) == 0:
            logger.debug("\'%s %s\': no disambiguation candidates were found." % (citation_string, scope))
            return Result(citation_string, entity_type, scope, NIL_URN)

        elif len(match) <= n_guess:
            logger.debug("There are %i matches and `n_guess`==%i. Nothing to cut." % (len(match), n_guess))

        elif len(match) > n_guess:
            logger.debug("There are %i matches: selecting based on LCS" % len(
                match
            ))
            match = select_lcs_match(citation_string, match, n_guess)

        for urn_string, label, score in match:
            result = self._consolidate_result(
                urn_string,
                citation_string,
                entity_type,
                scope
            )
            return result

    def _disambiguate_entity(self, mention, entity_type):
        """

        When no match is found it's better not to fill with a bogus URN. The
        reason is that in some cases it's perfectly ok that no match is found. An entity
        can be valid entity also without having disambiguation information in the groundtruth.

        :param mention:
        :param entity_type:
        :return: a named tuple  (see `Result`)

        """
        result = []
        matches = []

        distance_threshold = self.min_distance_entities
        max_distance_threshold = self.max_distance_entities
        """
        string = mention.encode("utf-8") # TODO: add a type check

        regex_clean_string = r'(« )|( »)|\(|\)|\,'
        cleaned_string = re.sub(regex_clean_string,"",string)
        string = cleaned_string
        """
        string = mention

        if entity_type == "AAUTHOR":

            if self.fuzzy_match_entities:

                matches = self.matches_author(string, True, distance_threshold)
                while(matches is None and distance_threshold <= max_distance_threshold):
                    distance_threshold+=1
                    matches = self.matches_author(string, True, distance_threshold)

            else:
                matches = self.matches_author(string, False)

        elif(entity_type == "AWORK"):

            if self.fuzzy_match_entities:

                matches = self.matches_work(string,True,distance_threshold)

                while(matches is None and distance_threshold <= max_distance_threshold):
                    distance_threshold+=1
                    matches = self.matches_work(string, True, distance_threshold)
            else:
                matches = self.matches_work(string, False)

        else:
            # TODO: raise exception
            logger.warning("unknown entity type: %s" % entity_type)

        if(matches is not None and len(matches)>0):
            lowest_score = 1000

            for match in matches:
                score = match[2]
                if(score < lowest_score):
                    lowest_score = score

            filtered_matches = [match for match in matches if match[2]==lowest_score]
            filtered_matches = sorted(filtered_matches, key =itemgetter(2))
            best_match = ("",None)

            if(lowest_score > 0):
                for match in filtered_matches:
                    lcs = longest_common_substring(match[1],string)
                    if(len(lcs)>len(best_match[0])):
                        best_match = (lcs,match)

                if(best_match[1] is not None):
                    return Result(mention, entity_type, None, best_match[1][0])
                else:
                    # TODO: perhaps log some message
                    return Result(mention, entity_type, None, filtered_matches[0][0])
            else:
                return Result(mention, entity_type, None, filtered_matches[0][0])

        else:
            return Result(mention, entity_type, None, NIL_URN)

    def matches_author(self, string, fuzzy=False, distance_threshold=3):
        """
        This function retrieves from the KnowledgeBase possible authors that match the search string.
        None is returned if no matches are found.

        :param string: the string to be matched

        :param fuzzy: whether exact or fuzzy string matching should be applied

        :distance_threshold: the maximum edit distance threshold (ignored if `fuzzy==False`)

        :return: a list of tuples, ordered by distance between the seach and the matching string, where:
                tuple[0] contains the id (i.e. CTS URN) of the matching author
                tuple[1] contains a label of the matching author
                tuple[2] is the distance, measured in characters, between the search string and the matching string
                or None if no match is found.
        """
        #string = string.lower()
        author_matches, abbr_matches = [],[]

        if(not fuzzy):

            author_matches = [(id.split("$$")[0]
                            , self._author_names[id]
                            , len(self._author_names[id])-len(string))
                             for id in self._author_idx.searchAllWords(string)]

            abbr_matches = [(id.split("$$")[0]
                            , self._author_abbreviations[id]
                            , len(self._author_abbreviations[id])-len(string))
                            for id in self._author_abbr_idx.searchAllWords(string)]
        else:
            abbr_matches = [(id.split("$$")[0]
                            , self._author_abbreviations[id]
                            , edit_distance(string,self._author_abbreviations[id]))
                            for id in self._author_abbreviations
                            if edit_distance(string,self._author_abbreviations[id]) <= distance_threshold]

            abbr_matches = sorted(abbr_matches, key =itemgetter(2))
            author_matches = []

            for id in self._author_names:
                if(string.endswith(".")):
                    if string.replace(".","") in self._author_names[id]:
                        if(len(string) > (len(self._author_names[id]) / 2)):
                            try:
                                assert abbr_matches[0][2] == 0
                                distance = len(self._author_names[id]) - len(string)
                                if distance < 0:
                                    distance = 1
                                author_matches.append((id.split("$$")[0], self._author_names[id],distance))
                            except Exception, e:
                                author_matches.append((id.split("$$")[0], self._author_names[id],0))
                        else:
                            if(edit_distance(string,self._author_names[id]) <= distance_threshold):
                                author_matches.append((id.split("$$")[0], self._author_names[id], edit_distance(string,self._author_names[id])))
                else:
                    if(edit_distance(string,self._author_names[id]) <= distance_threshold):
                        author_matches.append((id.split("$$")[0], self._author_names[id], edit_distance(string,self._author_names[id])))

        if(len(author_matches)>0 or len(abbr_matches)>0):
            return sorted(author_matches + abbr_matches, key =itemgetter(2))
        else:
            return None

    def matches_work(self, string, fuzzy=False, distance_threshold=3):
        """
        This function retrieves from the KnowledgeBase possible works that match the search string.
        None is returned if no matches are found.

        :param string: the string to be matched

        :param fuzzy: whether exact or fuzzy string matching should be applied

        :distance_threshold: the maximum edit distance threshold (ignored if `fuzzy==False`)

        :return: a list of tuples, ordered by distance between the seach and the matching string, where:
                tuple[0] contains the id (i.e. CTS URN) of the matching work
                tuple[1] contains a label of the matching work
                tuple[2] is the distance, measured in characters, between the search string and the matching string
                or None if no match is found.
        """
        #string = string.lower()
        work_matches, work_abbr_matches = [],[]

        if(not fuzzy):

            work_matches = [(id.split("$$")[0]
                            , self._work_titles[id]
                            , len(self._work_titles[id])-len(string))
                            for id
                            in self._work_idx.searchAllWords(string)]

            work_abbr_matches = [(id.split("$$")[0]
                                , self._work_abbreviations[id]
                                , len(self._work_abbreviations[id])-len(string))
                                for id
                                in self._work_abbr_idx.searchAllWords(string)]

            logger.debug("Matching works: %s (fuzzy matching=%s)" % (work_matches, fuzzy))
            logger.debug("Matching work abbreviations: %s (fuzzy matching=%s)" % (work_abbr_matches, fuzzy))

        else:
            string = string.lower()
            work_matches = []

            for id in self._work_titles:
                distance = edit_distance(string,self._work_titles[id])
                if distance <= distance_threshold:
                    work_matches.append(
                            (id.split("$$")[0]
                            , self._work_titles[id]
                            , distance)
                            )

            work_abbr_matches = [(id.split("$$")[0]
                                , self._work_abbreviations[id]
                                , edit_distance(string, self._work_abbreviations[id].lower()))
                                for id in self._work_abbreviations
                                if edit_distance(string, self._work_abbreviations[id].lower()) <= distance_threshold]

            logger.debug("Matching works: %s (fuzzy matching=%s; edit_distance_threshold=%i)" % (work_matches
                                                                                                , fuzzy
                                                                                                , distance_threshold))

            logger.debug("Matching work abbreviations: %s (fuzzy matching=%s; edit_distance_threshold=%i)" % (work_abbr_matches
                                                                                                            , fuzzy
                                                                                                            , distance_threshold))

        if(len(work_matches)>0 or len(work_abbr_matches)>0):
            return sorted(work_matches + work_abbr_matches, key=itemgetter(2))
        else:
            return None

    def disambiguate(self, surface, entity_type, scope=None, n_results=1, **kwargs):
        """
        :param surface:
        :param type:
        :param scope:
        :param n_results:

        """

        assert surface is not None

        text = StringUtils.remove_symbols(surface)
        text = StringUtils.remove_numbers(text)
        text = StringUtils.strip_accents(text)
        text = StringUtils.remove_punctuation(text, keep_dots=False)
        text = StringUtils.trim_spaces(text)
        cleaned_surface = text.lower()
        logger.debug("Citation string before and after cleaning: \"%s\" => \"%s\"" % (surface, cleaned_surface))

        # TODO: log the result
        if scope is None:
            return self._disambiguate_entity(cleaned_surface, entity_type)

        elif scope is not None:
            return self._disambiguate_relation(cleaned_surface, entity_type, scope, n_results)
