"""Tests for the module `citation_extractor.Utils`."""
# -*- coding: utf-8 -*-
# author: Matteo Romanello, matteo.romanello@gmail.com

import os
import pkg_resources
import logging
from citation_extractor.Utils.strmatching import StringUtils

logger = logging.getLogger(__name__)

#####################
# Utils.strmatching #
#####################


def test_utils_stringutils():
    strings = [
        (
            "de",
            u"Wie seine Vorgänger verfolgt auch\
            Ammianus die didaktische Absicht,"
        ),
        (
            "en",
            u"Judgement of Paris, with actors playing the bribing goddesses,\
            at the end of Book 10 (11, 3-5 : cf. 10, 30-31)."
        ),
        (
            "it",
            u"Superior e databili tra l'età augustea e il 5° sec. : AE 1952,\
            16 ; CIL 13, 8648 = ILS 2244 ; AE 1938, 120 ;"
        )
    ]

    for language, text in strings:
        normalized_text = StringUtils.normalize(text)
        normalized_text = StringUtils.normalize(text, language)
        normalized_text = StringUtils.normalize(text, language, keep_dots=True)
        assert normalized_text is not None
