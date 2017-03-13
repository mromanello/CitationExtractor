# -*- coding: utf-8 -*-
# author: Matteo Romanello, matteo.romanello@gmail.com

import logging
import rdflib
import re
from rdflib.graph import Graph
import sys

global logger
logger = logging.getLogger(__name__)

class KnowledgeBase(object):
    """
    docstring for KnowledgeBase

    TODO:
    * test allegrordf to connect to an AG 3store via rdflib (hard to run AG3 locally)
    * test mysql as backend

    Example:

    >>> kb = KnowledgeBase("/Users/rromanello/Documents/APh_Corpus_GUI/cwkb/export_triples/kb-all-in-one.ttl", "turtle")

    """

    def __init__(self, source_file=None, source_format=None, source_endpoint=None):
        super(KnowledgeBase, self).__init__()
        try:
            assert source_file is not None and source_format is not None
            self._source_file = source_file
            self._source_format = source_format
            self._graph = Graph()
            if (type(source_file) == type("string")):
                self._graph.parse(source_file, format=source_format)
            elif (type(source_file) == type([])):
                for file in source_file:
                    self._graph.parse(file, format=source_format)
            logger.info("Loaded %i triples" % len(self._graph))
            self._author_names = None
            self._author_abbreviations = None
            self._work_titles = None
            self._work_abbreviations = None
        except Exception, e:
            raise e

    def __getstate__(self):
        """
        Instances of `rdflib.Graph` cannot be serialised. Thus they need to be dropped
        when pickling
        """
        odict = self.__dict__.copy()
        del odict['_graph']
        return odict

    def __setstate__(self, dict):
        self.__dict__.update(dict)
        self._graph = Graph()
        if (type(self._source_file) == type("string")):
            self._graph.parse(self._source_file, format=self._source_format)
        elif (type(self._source_file) == type([])):
            for file in self._source_file:
                self._graph.parse(file, format=self._source_format)
            logger.info("Loaded %i triples" % len(self._graph))

    @property
    def author_names(self):
        if (self._author_names is not None):
            return self._author_names
        else:
            self._author_names = self._fetch_author_names()
            return self._author_names

    @property
    def author_abbreviations(self):
        if (self._author_abbreviations is not None):
            return self._author_abbreviations
        else:
            self._author_abbreviations = self._fetch_author_abbreviations()
            return self._author_abbreviations

    @property
    def work_titles(self):
        if (self._work_titles is not None):
            return self._work_titles
        else:
            self._work_titles = self._fetch_work_titles()
            return self._work_titles

    @property
    def work_abbreviations(self):
        if (self._work_abbreviations is not None):
            return self._work_abbreviations
        else:
            self._work_abbreviations = self._fetch_work_abbreviations()
            return self.work_abbreviations

    def _fetch_author_names(self, to_lowercase=True):
        authors_query = """
            PREFIX frbroo: <http://erlangen-crm.org/efrbroo/>
            PREFIX crm: <http://erlangen-crm.org/current/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

            SELECT DISTINCT ?author ?label ?urnstring

            WHERE {
                ?author crm:P1_is_identified_by ?name .
                ?name a frbroo:F12_Name .
                ?name rdfs:label ?label .

                OPTIONAL {
                    ?author crm:P1_is_identified_by ?urn .
                    ?urn a crm:E42_Identifier .
                    ?urn rdfs:label ?urnstring .
                }
            } ORDER BY (?author)
        """
        author_names = {}
        flat_author_names = {}
        query_result = self._graph.query(authors_query)
        for uri, author_name, urn in query_result:
            if (to_lowercase):
                name = author_name.lower()
            if author_name.language is not None:
                language = author_name.language
            else:
                language = "def"
            if urn is None:
                if author_names.has_key(uri):
                    author_names[uri][language] = name
                else:
                    author_names[uri] = {}
                    author_names[uri][language] = name
            else:
                if author_names.has_key(urn):
                    author_names[urn][language] = name
                else:
                    author_names[urn] = {}
                    author_names[urn][language] = name
        for key in author_names:
            for n, lang in enumerate(author_names[key]):
                flat_author_names["%s$$%i" % (key, n + 1)] = unicode(author_names[key][lang])
        return flat_author_names

    def _fetch_author_abbreviations(self, to_lowercase=True):
        author_abbreviations_query = """
            PREFIX frbroo: <http://erlangen-crm.org/efrbroo/>
            PREFIX crm: <http://erlangen-crm.org/current/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

            SELECT DISTINCT ?author ?label ?urnstring ?abbr

            WHERE {
                ?author crm:P1_is_identified_by ?name .
                ?name crm:P139_has_alternative_form ?abbrev .
                ?abbrev rdfs:label ?abbr .
                OPTIONAL {
                    ?author crm:P1_is_identified_by ?urn .
                    ?urn a crm:E42_Identifier .
                    ?urn rdfs:label ?urnstring .
                }
            } ORDER BY (?author)

        """
        abbreviations = {}
        flat_abbreviations = {}
        query_result = self._graph.query(author_abbreviations_query)
        for uri, author_name, urn, abbreviation in query_result:
            if (to_lowercase):
                abbr = abbreviation.lower()
            if urn is None:
                if abbreviations.has_key(uri):
                    abbreviations[uri] = abbreviations[uri].append(abbr)
                else:
                    abbreviations[uri] = []
                    abbreviations[uri].append(abbr)
            else:
                if abbreviations.has_key(urn):
                    abbreviations[urn].append(abbr)
                else:
                    abbreviations[urn] = []
                    abbreviations[urn].append(abbr)
        for key in abbreviations:
            if (abbreviations[key] is not None):
                for n, item in enumerate(abbreviations[key]):
                    flat_abbreviations["%s$$%i" % (key, n + 1)] = unicode(abbreviations[key][n])
            else:
                print key, abbreviations[key]
        return flat_abbreviations

    def _fetch_work_titles(self, to_lowercase=True):
        works_query = """
            PREFIX frbroo: <http://erlangen-crm.org/efrbroo/>
            PREFIX crm: <http://erlangen-crm.org/current/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

            SELECT DISTINCT ?work ?title ?urnstring

            WHERE {
                ?work a frbroo:F1_Work .
                ?work frbroo:P102_has_title ?utitle .
                ?utitle rdfs:label ?title
                OPTIONAL {
                    ?work crm:P1_is_identified_by ?urn .
                    ?urn a crm:E42_Identifier .
                    ?urn rdfs:label ?urnstring .
                }
            }
            ORDER BY (?work)

        """
        query_result = self._graph.query(works_query)
        work_titles = {}
        flat_work_titles = {}
        for uri, work_title, urn in query_result:
            if (to_lowercase):
                title = work_title.lower()
            if work_title.language is not None:
                language = work_title.language
            else:
                language = "def"
            # TODO remove articles according to language
            regexps = {
                "en": r'^the '
                , "de": r'^der |^die |^das '
                , "fr": r'^le |^la |^l\' |^les '
                , "it": r'^il | ^lo | ^la |^gli |^le '
                , "def": r''
                , "la": r''
            }
            if (language in regexps.keys()):
                title = re.sub(regexps[language], "", title)
            if urn is None:
                if (work_titles.has_key(uri)):
                    work_titles[uri][language] = title
                else:
                    work_titles[uri] = {}
                    work_titles[uri][language] = title
            else:
                if (work_titles.has_key(urn)):
                    work_titles[urn][language] = title
                else:
                    work_titles[urn] = {}
                    work_titles[urn][language] = title
        for key in work_titles:
            for n, lang in enumerate(work_titles[key]):
                flat_work_titles["%s$$%i" % (key, n + 1)] = unicode(work_titles[key][lang])
        return flat_work_titles

    def _fetch_work_abbreviations(self, to_lowercase=True):
        work_abbreviations_query = """
            PREFIX frbroo: <http://erlangen-crm.org/efrbroo/>
            PREFIX crm: <http://erlangen-crm.org/current/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

            SELECT DISTINCT ?work ?label ?urnstring ?abbr

            WHERE {
                ?work frbroo:P102_has_title ?title .
                ?title crm:P139_has_alternative_form ?abbrev .
                ?abbrev rdfs:label ?abbr .
                OPTIONAL {
                    ?work crm:P1_is_identified_by ?urn .
                    ?urn a crm:E42_Identifier .
                    ?urn rdfs:label ?urnstring .
                }
            } ORDER BY (?work)
        """
        work_abbreviations = {}
        flat_work_abbreviations = {}
        query_result = self._graph.query(work_abbreviations_query)
        for uri, work_title, urn, abbreviation in query_result:
            if (to_lowercase):
                abbr = abbreviation.lower()
            if urn is None:
                if work_abbreviations.has_key(uri):
                    work_abbreviations[uri] = work_abbreviations[uri].append(abbr)
                else:
                    work_abbreviations[uri] = []
                    work_abbreviations[uri].append(abbr)
            else:
                if work_abbreviations.has_key(urn):
                    work_abbreviations[urn].append(abbr)
                else:
                    work_abbreviations[urn] = []
                    work_abbreviations[urn].append(abbr)
        for key in work_abbreviations:
            if (work_abbreviations[key] is not None):
                for n, item in enumerate(work_abbreviations[key]):
                    flat_work_abbreviations["%s$$%i" % (key, n + 1)] = unicode(work_abbreviations[key][n])
            else:
                print key, work_abbreviations[key]
        return flat_work_abbreviations

    def get_URI_by_CTS_URN(self, input_urn):
        """
        Takes a CTS URN as input and returns the matching URI
        """
        search_query = """
            PREFIX frbroo: <http://erlangen-crm.org/efrbroo/>
            PREFIX crm: <http://erlangen-crm.org/current/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

            SELECT ?resource_URI

            WHERE {
                ?resource_URI crm:P1_is_identified_by ?urn .
                ?urn a crm:E42_Identifier .
                ?urn rdfs:label "%s"
            }
        """ % (input_urn)
        query_result = list(self._graph.query(search_query))
        # there must be only one URI match for a given URN
        assert len(query_result) == 1
        return query_result[0][0]

    def get_author_of(self, work_cts_urn):
        search_query = """
        PREFIX frbroo: <http://erlangen-crm.org/efrbroo/>
        PREFIX crm: <http://erlangen-crm.org/current/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX frbroo: <http://erlangen-crm.org/efrbroo/>

        SELECT ?author_urn_string
        WHERE {
            ?work crm:P1_is_identified_by ?urn .
            ?urn a crm:E42_Identifier .
            ?urn rdfs:label "%s" .
            ?creation frbroo:R16_initiated ?work .
            ?author frbroo:P14i_performed ?creation .
            ?author crm:P1_is_identified_by ?author_urn .
            ?author_urn a crm:E42_Identifier .
            ?author_urn rdfs:label  ?author_urn_string.
        }
        """ % work_cts_urn
        query_result = list(self._graph.query(search_query))
        return query_result[0][0]

    def get_name_of(self, author_cts_urn):
        search_query = """
        PREFIX frbroo: <http://erlangen-crm.org/efrbroo/>
        PREFIX crm: <http://erlangen-crm.org/current/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX frbroo: <http://erlangen-crm.org/efrbroo/>

        SELECT ?name
        WHERE {
            ?author crm:P1_is_identified_by ?author_urn .
            ?author_urn a crm:E42_Identifier .
            ?author_urn rdfs:label "%s" .
            ?author crm:P1_is_identified_by ?name_uri .
            ?name_uri a frbroo:F12_Name .
            ?name_uri rdfs:label ?name .
        }
        """ % author_cts_urn
        query_result = list(self._graph.query(search_query))
        return [name[0] for name in query_result]

    def get_title_of(self, work_cts_urn):
        search_query = """
        PREFIX frbroo: <http://erlangen-crm.org/efrbroo/>
            PREFIX crm: <http://erlangen-crm.org/current/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

            SELECT DISTINCT ?title

            WHERE {
                ?work a frbroo:F1_Work .
                ?work frbroo:P102_has_title ?utitle .
                ?utitle rdfs:label ?title .
                ?work crm:P1_is_identified_by ?urn .
                ?urn a crm:E42_Identifier .
                ?urn rdfs:label "%s" .
            }
        """ % work_cts_urn
        query_result = list(self._graph.query(search_query))
        return [title[0] for title in query_result]

    def get_opus_maximum_of(self, author_cts_urn):
        """
        given the CTS URN of an author, this method returns the CTS URN of
        its opus maximum. If not available returns None.
        """
        search_query = """
        PREFIX frbroo: <http://erlangen-crm.org/efrbroo/>
        PREFIX crm: <http://erlangen-crm.org/current/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX frbroo: <http://erlangen-crm.org/efrbroo/>
        PREFIX base: <http://127.0.0.1:8000/cwkb/types#>

        SELECT ?urn
        WHERE {
            ?work crm:P1_is_identified_by ?work_urn .
            ?work crm:P2_has_type base:opusmaximum .
            ?work_urn a crm:E42_Identifier .
            ?work_urn rdfs:label ?urn .
            ?creation frbroo:R16_initiated ?work .
            ?author frbroo:P14i_performed ?creation .
            ?author crm:P1_is_identified_by ?author_urn .
            ?author_urn a crm:E42_Identifier .
            ?author_urn rdfs:label  "%s".
        }
        """ % author_cts_urn
        query_result = list(self._graph.query(search_query))
        try:
            return query_result[0][0]
        except Exception, e:
            return None

    def validate(self):
        pass

    def describe(self, cts_urn, language="en"):
        """
        TODO: given a CTS URN, return a description
        Return a tuple where:
            result[0] is the label
            result[1] is the language of the label
        """
        pass
