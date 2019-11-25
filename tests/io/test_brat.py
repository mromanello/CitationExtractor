"""Tests for the module `citation_extractor.io.brat`."""
# -*- coding: utf-8 -*-
# author: Matteo Romanello, matteo.romanello@gmail.com

import pkg_resources
from functools import reduce
import logging
import pandas as pd
from citation_extractor.io.brat import read_ann_file
from citation_extractor.io.brat import load_brat_data, annotations2references

logger = logging.getLogger(__name__)


def test_annotations2references(knowledge_base):
    datadir = ('citation_extractor', 'data/aph_corpus/goldset/ann/')
    dir = pkg_resources.resource_filename(*datadir)
    files = [
        file.replace('-doc-1.ann', '')
        for file in pkg_resources.resource_listdir(*datadir)
        if '.ann' in file
    ]
    all_annotations = [
        annotations2references(file, dir, knowledge_base) for file in files[:10]
    ]
    references = reduce((lambda x, y: x + y), all_annotations)
    assert references is not None
    return


def test_load_brat_data(
    crfsuite_citation_extractor,
    knowledge_base,
    postaggers,
    aph_test_ann_files,
    aph_titles,
):
    assert crfsuite_citation_extractor is not None
    # load the pandas.DataFrame
    dataframe = load_brat_data(
        crfsuite_citation_extractor,
        knowledge_base,
        postaggers,
        aph_test_ann_files,
        aph_titles,
    )
    assert dataframe is not None
    assert isinstance(dataframe, pd.DataFrame)
    assert dataframe.shape[0] > 0


def test_read_ann_file():
    dir = pkg_resources.resource_filename(
        'citation_extractor', 'data/aph_corpus/goldset/ann/'
    )
    files = [
        file.replace('-doc-1.ann', '')
        for file in pkg_resources.resource_listdir(
            'citation_extractor', 'data/aph_corpus/goldset/ann/'
        )
        if '.ann' in file
    ]
    for file in files[:10]:

        logger.debug(file)
        entities, relations, annotations = read_ann_file(file, dir)
        logger.debug("Entities: {}".format(entities))

        # check the ids of entities which are arguments in relations
        # are actually contained in the list of entities
        for rel_id in relations:
            logger.debug(relations[rel_id])
            for entity_id in relations[rel_id]["arguments"]:
                assert entity_id in entities

        logger.debug(annotations)
        for annotation in annotations:
            assert (
                annotation["anchor"] in relations
                or annotation["anchor"] in entities
            )
