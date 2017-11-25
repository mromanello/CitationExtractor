"""Machine learning code related to the NED step."""

# -*- coding: utf-8 -*-


from __future__ import print_function

import pdb

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

from citation_extractor.Utils.strmatching import StringSimilarity
from citation_extractor.Utils.extra import avg

import pkg_resources

import pandas as pd
from stop_words import safe_get_stop_words
from sklearn.feature_extraction.text import TfidfVectorizer

LOGGER = logging.getLogger(__name__)


# TODO: how to deal with parallel computing?
# TODO: how to deal/apply with (optional?) refinement step?


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
    pass


if __name__ == '__main__':
    main()
