"""Tests for the module `citation_extractor.Utils`."""
# -*- coding: utf-8 -*-
# author: Matteo Romanello, matteo.romanello@gmail.com

import os
import pkg_resources
import logging
from citation_extractor.io.converters import DocumentConverter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_document_converter(aph_gold_ann_files, knowledge_base):

    standoff_dir = pkg_resources.resource_filename(
        'citation_extractor',
        'data/aph_corpus/goldset/ann/'
    )
    """
    uima_typesystem = pkg_resources.resource_filename(
        'citation_extractor',
        'data/uima/typesystem.xml'
    )
    """
    iob_data_dir = ('citation_extractor', 'data/aph_corpus/goldset/iob/')
    output_dir = pkg_resources.resource_filename(
        'citation_extractor',
        'data/converted/'
    )
    dir = pkg_resources.resource_filename(*iob_data_dir)
    files = [
        os.path.join(dir, file)
        for file
        in pkg_resources.resource_listdir(*iob_data_dir) if '.txt' in file
    ]

    dc = DocumentConverter(knowledge_base)

    for n in range(0, 5):
        iob_file = files[n]
        print(iob_file, standoff_dir)
        dc.load(iob_file, standoff_dir)
        # dc.to_xmi(output_dir, uima_typesystem)
        out = dc.to_json(output_dir)
        print(out)
