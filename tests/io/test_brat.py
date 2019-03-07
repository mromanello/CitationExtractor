"""Tests for the module `citation_extractor.io.brat`."""
# -*- coding: utf-8 -*-
# author: Matteo Romanello, matteo.romanello@gmail.com

import os
import pkg_resources
import logging
import pandas as pd
from citation_extractor.io.brat import load_brat_data, annotations2references


def test_annotations2references(knowledge_base):
    datadir = ('citation_extractor', 'data/aph_corpus/goldset/ann/')
    dir = pkg_resources.resource_filename(*datadir)
    files = [
        file.replace('-doc-1.ann', '')
        for file
        in pkg_resources.resource_listdir(*datadir) if '.ann' in file
    ]
    all_annotations = [
        annotations2references(file, dir, knowledge_base)
        for file in files[:10]
    ]
    references = reduce((lambda x, y: x + y), all_annotations)
    assert references is not None
    return


def test_load_brat_data(
    crfsuite_citation_extractor,
    knowledge_base, postaggers,
    aph_test_ann_files,
    aph_titles
):
    assert crfsuite_citation_extractor is not None
    # load the pandas.DataFrame
    dataframe = load_brat_data(
        crfsuite_citation_extractor,
        knowledge_base,
        postaggers,
        aph_test_ann_files,
        aph_titles
    )
    assert dataframe is not None
    assert isinstance(dataframe, pd.DataFrame)
    assert dataframe.shape[0] > 0
