# -*- coding: utf-8 -*-
# author: Matteo Romanello, matteo.romanello@gmail.com

import logging
import os

import pkg_resources
import pytest

from citation_extractor.pipeline import (detect_language, do_ned, do_ner,
                                         do_relex, preproc_document)

logger = logging.getLogger(__name__)


def test_tokenize_string(aph_titles, postaggers):
    # as a string example, take the first title of APh reviews
    aph_title = aph_titles.iloc[0]["title"]
    lang = detect_language(aph_title)
    postagged_string = postaggers[lang].tag(aph_title)
    logger.debug(postagged_string)
    assert postagged_string is not None


def test_preprocessing(processing_directories, postaggers):
    """Run pre-processing on selected APh documents in the devset."""
    inp_dir = processing_directories["input"]
    docids = [
        "75-00557.txt",
        "75-00351.txt",
        "75-00087.txt",
        "75-00060.txt",
        "75-00046.txt"
    ]
    abbreviations = pkg_resources.resource_filename(
        'citation_extractor',
        'data/aph_corpus/extra/abbreviations.txt'
    )
    interm_dir = processing_directories["txt"]
    out_dir = processing_directories["iob"]
    for docid in docids:
        logger.info(
            preproc_document(
                docid,
                inp_dir,
                interm_dir,
                out_dir,
                abbreviations,
                postaggers,
                False
            )
        )
    assert len(os.listdir(out_dir)) > 0


def test_do_ner(processing_directories, crfsuite_citation_extractor):
    """Test the Named Entity Recognition step of the pipeline."""

    # folder with tokenized and postagged docs
    inp_dir = processing_directories["iob"]

    # folder for intermediate output
    interm_dir = processing_directories["iob_ne"]

    # the default output format is JSON
    out_dir = processing_directories["json"]
    docids = os.listdir(inp_dir)

    for docid in docids:
        logger.info(
            do_ner(
                docid,
                inp_dir=inp_dir,
                interm_dir=interm_dir,
                out_dir=out_dir,
                extractor=crfsuite_citation_extractor
            )
        )
    assert len(os.listdir(out_dir)) > 0


def test_do_relex_rulebased(processing_directories):
    """Test the Relation Extraction step of the pipeline."""
    inp_dir = processing_directories["ann"]
    docids = [
        file.replace('-doc-1.ann', '')
        for file in os.listdir(inp_dir)
        if '.ann' in file
    ]
    logger.info(docids)
    for docid in docids:
        docid, success, data = do_relex(docid, inp_dir)
        logger.info(data)
        assert success


@pytest.mark.skip
def test_do_ned_fuzzymatching(processing_directories, citation_matcher):
    """Test the Named Entity Disambiguation step of the pipeline (baseline)."""
    inp_dir = processing_directories["ann"]
    docids = [
        file.replace('-doc-1.ann', '')
        for file in os.listdir(inp_dir)
        if '.ann' in file
    ]
    for docid in docids[:50]:
        docid, success, n_disambiguations = do_ned(
            docid,
            inp_dir,
            citation_matcher,
            True,
            0,
            False
        )
        assert success
