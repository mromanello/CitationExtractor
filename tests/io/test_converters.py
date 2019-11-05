"""Tests for the module `citation_extractor.io.converters`."""
# -*- coding: utf-8 -*-
# author: Matteo Romanello, matteo.romanello@gmail.com

import os
import pkg_resources
import logging
from citation_extractor.io.converters import DocumentConverter

logger = logging.getLogger(__name__)


def test_converter_json(knowledge_base):
    """
    Tests the conversion of a bunch of docs (IOB + ANN format) into JSON.
    """

    # directory containing the input IOB documents
    iob_data_dir = ('citation_extractor', 'data/aph_corpus/goldset/iob/')

    # directory containing the input brat documents
    # to each IOB document should correspond an `.ann` and `.iob` file with
    # the same name
    brat_dir = pkg_resources.resource_filename(
        'citation_extractor',
        'data/aph_corpus/goldset/ann/'
    )

    output_dir = pkg_resources.resource_filename(
        'citation_extractor',
        'data/converted/json/'
    )

    # reconstruct the absolute path of IOB files
    files = [
        os.path.join(pkg_resources.resource_filename(*iob_data_dir), file)
        for file
        in pkg_resources.resource_listdir(*iob_data_dir) if '.txt' in file
    ]

    dc = DocumentConverter(knowledge_base)

    for n in range(0, 5):
        iob_file = files[n]
        if not os.path.exists(iob_file):
            continue
        logger.info("{} {}".format(iob_file, brat_dir))
        dc.load(iob_file, standoff_dir=brat_dir)
        out = dc.to_json(output_dir)
        assert out is not None


def test_iob2json():
    """
    Tests the conversion of a bunch of IOB docs into JSON format.
    """

    # directory containing the input IOB documents
    iob_data_dir = ('citation_extractor', 'data/aph_corpus/goldset/iob/')

    # directory where to store the converted files
    output_dir = pkg_resources.resource_filename(
        'citation_extractor',
        'data/converted/json/'
    )

    # reconstruct the absolute path of IOB files
    files = [
        os.path.join(pkg_resources.resource_filename(*iob_data_dir), file)
        for file
        in pkg_resources.resource_listdir(*iob_data_dir) if '.txt' in file
    ]

    dc = DocumentConverter()

    for n in range(0, 5):
        iob_file = files[n]
        logger.info(iob_file)
        logger.info("{}".format(iob_file))
        dc.load(iob_file)
        dc.to_json(output_dir)
    pass
