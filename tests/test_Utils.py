# -*- coding: utf-8 -*-
# author: Matteo Romanello, matteo.romanello@gmail.com

import pytest
import pdb
import pkg_resources
import logging
import pandas as pd
from pytest import fixture
from citation_extractor.Utils.IO import *
from citation_extractor.Utils.strmatching import StringUtils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

############
# Utils.IO #
############

def test_annotations2references(knowledge_base):
    datadir = ('citation_extractor','data/aph_corpus/goldset/ann/')
    dir = pkg_resources.resource_filename(*datadir)
    files = [file.replace('-doc-1.ann','') for file in pkg_resources.resource_listdir(*datadir) if '.ann' in file]
    all_annotations = [annotations2references(file, dir, knowledge_base) for file in files]
    references = reduce((lambda x, y: x + y), all_annotations)
    assert references is not None

#def test_sort_entities(): #TODO implement
#    raise NotImplementedError

#@pytest.mark.skip
def test_load_brat_data(crf_citation_extractor, knowledge_base, postaggers, aph_test_ann_files, aph_titles):
    # load the pandas.DataFrame
    dataframe = load_brat_data(crf_citation_extractor, knowledge_base, postaggers, aph_test_ann_files, aph_titles)
    assert dataframe is not None and type(dataframe)==type(pd.DataFrame()) and dataframe.shape[0]>0

#####################
# Utils.strmatching #
#####################

def test_utils_stringutils():

    strings = [
        ("de", u"Wie seine Vorgänger verfolgt auch Ammianus die didaktische Absicht,")
        , ("en", u"Judgement of Paris, with actors playing the bribing goddesses, at the end of Book 10 (11, 3-5 : cf. 10, 30-31).")
        , ("it", u"Superior e databili tra l'età augustea e il 5° sec. : AE 1952, 16 ; CIL 13, 8648 = ILS 2244 ; AE 1938, 120 ;")
    ]

    for language, text in strings:
        normalized_text = StringUtils.normalize(text)
        normalized_text = StringUtils.normalize(text, language)
        normalized_text = StringUtils.normalize(text, language, keep_dots=True)
