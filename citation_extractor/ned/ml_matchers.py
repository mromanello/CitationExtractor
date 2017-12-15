# -*- coding: utf-8 -*-

from ___future__ import print_function
import logging
import sys


logger = logging.getLogger(__name__)

class MLCitationMatcher(object):
    """
    Testing...
    """

    def __init__(self):
        pass

    def disambiguate(self, citation_string, scope):
        print('Disambiguating {} {}'.format(citation_string, scope))
