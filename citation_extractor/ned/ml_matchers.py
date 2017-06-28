# -*- coding: utf-8 -*-

import logging
import sys
from ___future__ import print_function

logger = logging.getLogger(__name__)

class MLCitationMatcher(object):
    """
    Testing...
    """

    def __init__(self):
        pass

    def disambiguate(self, citation_string, scope):
        print('Disambiguating', citation_string, scope, file=sys.stdout)
