"""Tests for the module `citation_extractor.utils.lookup`"""
# -*- coding: utf-8 -*-
# author: Matteo Romanello, matteo.romanello@gmail.com

import pkg_resources
from citation_extractor.utils.lookup import LookupDictionary

AUTHOR_CSV_PATH = pkg_resources.resource_filename(
    "citation_extractor", "data/authors.csv"
)

WORK_CSV_PATH = pkg_resources.resource_filename("citation_extractor", "data/works.csv")


def test_author_dictionary_lookup():
    lookup_dictionary = LookupDictionary(AUTHOR_CSV_PATH)


def test_work_dictionary_lookup():
    lookup_dictionary = LookupDictionary(WORK_CSV_PATH)
