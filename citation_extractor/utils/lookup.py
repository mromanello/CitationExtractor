# -*- coding: utf-8 -*-

"""Utility classes to perform efficient dictionary-based lookup."""
import os
import csv
import pdb
import pandas as pd
from typing import Dict
from citation_extractor.extra.pysuffix.suffixIndexers import DictValuesIndexer


class LookupDictionary:
    """TODO"""

    def __init__(self, csvpath: str):
        assert os.path.exists(csvpath)
        self.keys = self._process_source(csvpath)
        pdb.set_trace()
        self.indexer = self.index(self.keys)

    def _process_source(self, data_path: str):
        """Turns CSV data into a key-label dictionary."""

        with open(data_path, "r") as f:
            data = pd.read_csv(data_path, sep="\t", encoding="utf-8")

        # transform the CSV data into a dictionary. to accomodate
        # multiple lapels per resource (author/work), the dict key
        # is the concatenation of TYPE and URI columns
        try:
            return {f"{row.URI}#{row.TYPE}": row.TEXT for idx, row in data.iterrows()}
        except Exception as e:
            raise e

    def index(self, dictionary: Dict[str, str]):
        """Indexes a dictionary for quick lookups."""
        return DictValuesIndexer(dictionary)

    def lookup(self, search_key):
        """
        Searches within the internal dictionary.
        For each matching entry its key is returned.

        Args:
            search_key:
                ...

        Returns:
            a dictionary:

        """
        result = {}
        temp = self.indexer.searchAllWords(search_key)
        for res in temp:
            result[res] = self.keys[res]
        return result


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
