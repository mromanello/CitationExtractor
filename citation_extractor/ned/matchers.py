# -*- coding: utf-8 -*-
# author: Matteo Romanello, matteo.romanello@gmail.com

"""
# TODO: remove after development

import logging
from citation_extractor.Utils.IO import init_logger
init_logger(loglevel=logging.DEBUG)

%load_ext autoreload
%autoreload 2

import codecs, pickle
with codecs.open("citation_extractor/data/pickles/kb_data.pkl","rb") as pickle_file:
    kb_data = pickle.load(pickle_file)

from knowledge_base import KnowledgeBase
from citation_extractor.ned import CitationMatcher
cm = FuzzyCitationMatcher(fuzzy_matching_entities=True, fuzzy_matching_relations=True, **kb_data)
cm._disambiguate_entity("Iliad's", "AWORK")
%time y = cm._disambiguate_entity("Ovid", "AAUTHOR")
"""

import re
import sys
import logging
from operator import itemgetter
from collections import namedtuple
from nltk.metrics import edit_distance
from pyCTS import CTS_URN
from pysuffix import suffixIndexers
from pysuffix.suffixIndexers import DictValuesIndexer
from citation_parser import CitationParser
from citation_extractor.pipeline import NIL_URN

global logger
logger = logging.getLogger(__name__)


Result = namedtuple('DisambiguationResult','mention, cleaned_mention, entity_type, scope, urn') # TODO: not sure about `scope`

class DisambiguationNotFound(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)
    def __str__(self):
        return repr(self.message)

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

# deprecated! remove asap and replace with the above
def longestSubstringFinder(string1, string2):
    """
    solution taken from http://stackoverflow.com/questions/18715688/find-common-substring-between-two-strings
    """
    answer = ""
    len1, len2 = len(string1), len(string2)
    for i in range(len1):
        match = ""
        for j in range(len2):
            if (i + j < len1 and string1[i + j] == string2[j]):
                match += string2[j]
            else:
                if (len(match) > len(answer)): answer = match
                match = ""
    return answer

class FuzzyCitationMatcher(object):
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

        if knowledge_base is None and 'author_names' in kwargs and 'work_titles' in kwargs \
            and 'work_abbreviations' in kwargs and 'author_abbreviations' in kwargs:

            self._author_names = kwargs["author_names"]
            self._author_abbreviations = kwargs["author_abbreviations"]
            self._work_titles = kwargs["work_titles"]
            self._work_abbreviations = kwargs["work_abbreviations"]

        elif knowledge_base is not None:

            logger.info("Initialising CitationMatcher...")
            self._citation_parser = CitationParser()
            self._kb = knowledge_base
            
            logger.info("Fetching author names from the KB...")
            self._author_names = knowledge_base.author_names
            
            logger.info("Done. Fetching work titles from the KB...")
            self._work_titles = knowledge_base.work_titles
            
            logger.info("Done. Fetching author abbreviations from the KB...")
            self._author_abbreviations = knowledge_base.author_abbreviations
            
            logger.info("Done. Fetching work abbreviations from the KB...")
            self._work_abbreviations = knowledge_base.work_abbreviations
            
            logger.info("Done. Now let's index all this information.")
            
        else:
            raise

        self._author_idx, self._author_abbr_idx, self._work_idx, self._work_abbr_idx = self._initialise_indexes()
        logger.info("CitationMatcher initialised with %i author abbreviations, %i author_names, %i work abbreviations, %i work titles" 
                                                                                    % (len(self._author_abbreviations)
                                                                                    , len(self._author_names)
                                                                                    , len(self._work_abbreviations)
                                                                                    , len(self._work_titles) 
                                                                                    ))
    
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
    
    def _disambiguate_relation(self, citation_string, scope):
        """
        Returns:
             [(u'R5', u'[ Verg. ] catal. 47s', u'urn:cts:TODO:47s')]
        """
        match = []
        try:
            normalized_scope = self._citation_parser.parse(scope)
        except Exception, e:
            print >> sys.stderr, "Got exception %s while parsing \"%s\""%(e,scope)
            normalized_scope = scope
        
        # citation string has one single token
        if(len(citation_string.split(" "))==1):
            match = self.matches_work(citation_string,fuzzy,distance_threshold)
            # TODO this is problematic
            # should be: match is None or match does not contain at least one entry with distance=0
            zero_distance_match = False
            if(match is not None):
                for m in match:
                    if(m[2]==0):
                        zero_distance_match = True
            print fuzzy
            print citation_string
            print "zero distance match is %s"%zero_distance_match
            if match is None or not zero_distance_match:
            #if match is None:
                try:
                    match = self.matches_author(citation_string,fuzzy,distance_threshold)
                except Exception, e:
                    raise e
            if match is not None:
                #match = [(id,name,diff) for id, name, diff in match if diff == 0][:n_guess] # this has to be removed
                pass
            else:
                # fuzzy matching as author
                # then fuzzy matching as work
                # ad the end take the matching with lowest score
                pass
        
        # citation string has two tokens
        elif(len(citation_string.split(" "))==2):
            tok1 = citation_string.split(" ")[0]
            tok2 = citation_string.split(" ")[1]
            # case 1: tok1 is author and tok2 is work
            match_tok1 = self.matches_author(tok1,fuzzy,distance_threshold)
            match_tok2 = self.matches_work(tok2,fuzzy,distance_threshold)
            #print >> sys.stderr, match_tok1
            #print >> sys.stderr, match_tok2
            if(match_tok1 is not None and match_tok2 is not None):
                # take this
                for id1,label1,score1 in match_tok1:
                    for id2,label2,score2 in match_tok2:
                        if(id1 in id2):
                            match = [(id2,label2,score2)]
                            break
            else:
                # case 2: tok1 and tok2 are author
                match = self.matches_author(citation_string,fuzzy,distance_threshold)
                if match is None:
                    # case 3: tok1 and tok2 are work
                    match = self.matches_work(citation_string,fuzzy,distance_threshold)
                else:
                    # take this
                    pass
        
        # citation string has more than two tokens
        elif(len(citation_string.split(" "))>2):
            match = self.matches_author(citation_string, fuzzy, distance_threshold)
        
        # return only n_guess results
        if(match is None or len(match)==0):
            raise DisambiguationNotFound("For the string \'%s\' no candidates for disambiguation were found!"%citation_string)
        elif(len(match)<= n_guess):
            print >> sys.stderr, "There are %i results and `n_guess`==%i. Nothing to cut."%(len(match),n_guess)
        elif(len(match)> n_guess):
            # iterate and get what's the lowest ed_score
            # then keep only the matches with lowest (best) score
            # then keep the one with longest common string
            lowest_score = 1000
            for m in match:
                score = m[2]
                if(score < lowest_score):
                    lowest_score = score
            filtered_matches = [m for m in match if m[2]==lowest_score]
            best_match = ("",None)
            if(lowest_score > 0):
                for match in filtered_matches:
                    lcs = longestSubstringFinder(match[1],citation_string)
                    if(len(lcs)>len(best_match[0])):
                        best_match = (lcs,match)
                match = [best_match[1]]
                print match
            else:
                # TODO: use context here to disambiguate
                match = match[:n_guess]
        
        results = []
        for id, label, score in match:
            formatted_scope = self._format_scope(normalized_scope[0]['scp'])
            urn = CTS_URN("%s:%s"%(id, formatted_scope))
            # check: does the URN has a scope but is missing the work element (not possible)?
            if(urn.work is None):
                # if so, try to get the opus maximum from the KB
                opmax = self._kb.get_opus_maximum_of(urn.get_urn_without_passage())
                if(opmax is not None):
                    print >> sys.stderr, "%s is opus maximum of %s"%(opmax,urn)
                    urn = CTS_URN("%s:%s"%(opmax,formatted_scope))
            results.append(urn)
        print >> sys.stderr,results
        return results
    
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
        string = mention.encode("utf-8") # TODO: add a type check
        
        regex_clean_string = r'(« )|( »)|\(|\)|\,'
        cleaned_string = re.sub(regex_clean_string,"",string)
        string = cleaned_string

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
            print "unknown entity type"
        
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
                    return Result(mention, string, entity_type, best_match[1][0])
                else:
                    # TODO: perhaps log some message
                    return Result(mention, string, entity_type, filtered_matches[0][0])
            else:
                return Result(mention, string, entity_type, filtered_matches[0][0])
        
        else:
            return Result(mention, string, entity_type, NIL_URN)
    
    def matches_author(self, string, fuzzy=False, distance_threshold=3):
        """
        This function retrieves from the KnowledgeBase possible authors that match the search string.
        None is returned if no matches are found.

        Args:
            string: the search string

        Returns:
            a list of tuples, ordered by distance between the seach and the matching string, where:
                tuple[0] contains the id (i.e. CTS URN) of the matching author
                tuple[1] contains a label of the matching author
                tuple[2] is the distance, measured in characters, between the search string and the matching string
            or None if no match is found.
        """
        #string = string.lower()
        author_matches, abbr_matches = [],[]
        if(not fuzzy):
            author_matches = [(id.split("$$")[0], self._author_names[id], len(self._author_names[id])-len(string)) for id in self._author_idx.searchAllWords(string)]
            abbr_matches = [(id.split("$$")[0], self._author_abbreviations[id], len(self._author_abbreviations[id])-len(string)) for id in self._author_abbr_idx.searchAllWords(string)]
        else:
            abbr_matches = [(id.split("$$")[0], self._author_abbreviations[id], edit_distance(string,self._author_abbreviations[id])) for id in self._author_abbreviations if edit_distance(string,self._author_abbreviations[id]) <= distance_threshold]
            abbr_matches = sorted(abbr_matches, key =itemgetter(2))
            author_matches = []
            for id in self._author_names:
                if(string.endswith(".")):
                    if string.replace(".","") in self._author_names[id]:
                        if(len(string)>(len(self._author_names[id])/2)):
                            try:
                                assert abbr_matches[0][2]==0
                                distance = len(self._author_names[id])-len(string)
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

        Args:
            string: the search string

        Returns:
            a list of tuples, ordered by distance between the seach and the matching string, where:
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

            logger.debug("Matching works: %s (fuzzy matching=%s; edit_distance_threshold=%i)" % (work_matches, fuzzy, distance_threshold))
            logger.debug("Matching work abbreviations: %s (fuzzy matching=%s; edit_distance_threshold=%i)" % (work_abbr_matches, fuzzy, distance_threshold))

        if(len(work_matches)>0 or len(work_abbr_matches)>0):
            return sorted(work_matches + work_abbr_matches, key=itemgetter(2))
        else:
            return None
    
    def print_settings(self):
        pass

    def disambiguate(self, surface, scope, type, **kwargs): #TODO: implement
        """
        TODO
        """
        pass

    def old_disambiguate(self
                    , citation_string
                    , scope
                    , n_guess=1
                    , validate = False
                    , fuzzy = False
                    , distance_threshold=3
                    , use_context = False
                    , entities_before = None
                    , entities_after = None
                    , cleanup = False
                    ):
        """

        NB: the string cleaning should happen here, not outside!

        Args:
            citation_string:
                e.g. "Hom. Il."
            scope:
                e.g. "10.1", "1.204", "X 345"
            n_guess:
                number of guesses that should be returned
                if n_guess > 1, they are returned as ordered list, with
                the most likely candidate first and the least likely last.
        Returns:
            a list of pyCTS.CTS_URN objects.

        Example:
            >>> cm.disambiguate("Hom. Il.","1.100")
        """
        def longestSubstringFinder(string1, string2):
            """
            solution taken from http://stackoverflow.com/questions/18715688/find-common-substring-between-two-strings
            """
            answer = ""
            len1, len2 = len(string1), len(string2)
            for i in range(len1):
                match = ""
                for j in range(len2):
                    if (i + j < len1 and string1[i + j] == string2[j]):
                        match += string2[j]
                    else:
                        if (len(match) > len(answer)): answer = match
                        match = ""
            return answer

        match = []
        try:
            normalized_scope = self._citation_parser.parse(scope)
        except Exception, e:
            print >> sys.stderr, "Got exception %s while parsing \"%s\""%(e,scope)
            normalized_scope = scope
        
        # citation string has one single token
        if(len(citation_string.split(" "))==1):
            match = self.matches_work(citation_string,fuzzy,distance_threshold)
            # TODO this is problematic
            # should be: match is None or match does not contain at least one entry with distance=0
            zero_distance_match = False
            if(match is not None):
                for m in match:
                    if(m[2]==0):
                        zero_distance_match = True
            print fuzzy
            print citation_string
            print "zero distance match is %s"%zero_distance_match
            if match is None or not zero_distance_match:
            #if match is None:
                try:
                    match = self.matches_author(citation_string,fuzzy,distance_threshold)
                except Exception, e:
                    raise e
            if match is not None:
                #match = [(id,name,diff) for id, name, diff in match if diff == 0][:n_guess] # this has to be removed
                pass
            else:
                # fuzzy matching as author
                # then fuzzy matching as work
                # ad the end take the matching with lowest score
                pass
        
        # citation string has two tokens
        elif(len(citation_string.split(" "))==2):
            tok1 = citation_string.split(" ")[0]
            tok2 = citation_string.split(" ")[1]
            # case 1: tok1 is author and tok2 is work
            match_tok1 = self.matches_author(tok1,fuzzy,distance_threshold)
            match_tok2 = self.matches_work(tok2,fuzzy,distance_threshold)
            #print >> sys.stderr, match_tok1
            #print >> sys.stderr, match_tok2
            if(match_tok1 is not None and match_tok2 is not None):
                # take this
                for id1,label1,score1 in match_tok1:
                    for id2,label2,score2 in match_tok2:
                        if(id1 in id2):
                            match = [(id2,label2,score2)]
                            break
            else:
                # case 2: tok1 and tok2 are author
                match = self.matches_author(citation_string,fuzzy,distance_threshold)
                if match is None:
                    # case 3: tok1 and tok2 are work
                    match = self.matches_work(citation_string,fuzzy,distance_threshold)
                else:
                    # take this
                    pass
        
        # citation string has more than two tokens
        elif(len(citation_string.split(" "))>2):
            match = self.matches_author(citation_string,fuzzy,distance_threshold)
        
        # return only n_guess results
        if(match is None or len(match)==0):
            raise DisambiguationNotFound("For the string \'%s\' no candidates for disambiguation were found!"%citation_string)
        elif(len(match)<= n_guess):
            print >> sys.stderr, "There are %i results and `n_guess`==%i. Nothing to cut."%(len(match),n_guess)
        elif(len(match)> n_guess):
            # iterate and get what's the lowest ed_score
            # then keep only the matches with lowest (best) score
            # then keep the one with longest common string
            lowest_score = 1000
            for m in match:
                score = m[2]
                if(score < lowest_score):
                    lowest_score = score
            filtered_matches = [m for m in match if m[2]==lowest_score]
            best_match = ("",None)
            if(lowest_score > 0):
                for match in filtered_matches:
                    lcs = longestSubstringFinder(match[1],citation_string)
                    if(len(lcs)>len(best_match[0])):
                        best_match = (lcs,match)
                match = [best_match[1]]
                print match
            else:
                # TODO: use context here to disambiguate
                match = match[:n_guess]
        
        results = []
        for id, label, score in match:
            formatted_scope = self._format_scope(normalized_scope[0]['scp'])
            urn = CTS_URN("%s:%s"%(id, formatted_scope))
            # check: does the URN has a scope but is missing the work element (not possible)?
            if(urn.work is None):
                # if so, try to get the opus maximum from the KB
                opmax = self._kb.get_opus_maximum_of(urn.get_urn_without_passage())
                if(opmax is not None):
                    print >> sys.stderr, "%s is opus maximum of %s"%(opmax,urn)
                    urn = CTS_URN("%s:%s"%(opmax,formatted_scope))
            results.append(urn)
        print >> sys.stderr,results
        return results