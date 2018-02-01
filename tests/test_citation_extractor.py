"""Tests for the module `citation_extractor.core`."""

# -*- coding: utf-8 -*-
# author: Matteo Romanello, matteo.romanello@gmail.com

import pickle
import logging
from citation_extractor.settings import crf
from citation_extractor import pipeline
from citation_extractor.Utils.IO import read_iob_files, filter_IOB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_string2entities(aph_titles, crfsuite_citation_extractor, postaggers):
    """Demonstrates how to extract entities (aauthor, awork) from a string."""
    aph_title = aph_titles.iloc[0]["title"]
    print aph_title

    # detect the language of the input string for starters
    lang = pipeline.detect_language(aph_title)

    # tokenise and do Part-of-Speech tagging
    postagged_string = postaggers[lang].tag(aph_title)

    # convert to a list of lists; keep just token and PoS tag, discard lemma
    iob_data = [
        [
            token[:2]
            for token in sentence
        ] for sentence in [postagged_string]
    ]

    # put the PoS tags into a separate nested list
    postags = [
        [
            ("z_POS", token[1])
            for token in sentence
        ] for sentence in iob_data if len(sentence) > 0
    ]

    # put the tokens into a separate nested list
    tokens = [
        [
            token[0]
            for token in sentence
        ] for sentence in iob_data if len(sentence) > 0
    ]

    # invoke the citation extractor
    tagged_sents = crfsuite_citation_extractor.extract(tokens, postags)

    # convert the (verbose) output into an IOB structure
    output = [
        [
            (
                res[n]["token"].decode('utf-8'),
                postags[i][n][1], res[n]["label"]
            )
            for n, d_res in enumerate(res)
        ]
        for i, res in enumerate(tagged_sents)
    ]
    authors = filter_IOB(output, "AAUTHOR")
    works = filter_IOB(output, "AWORK")
    logger.info("Extracted AAUTHOR entities: %s" % ", ".join(authors))
    logger.info("Extracted AWORK entities: %s" % ", ".join(works))
    assert output is not None


def test_pickle_citation_extractor(crfsuite_citation_extractor):
    """
    Make sure that instances of `citation_extractor` can be pickled (important
    for parallel processing!)
    """
    # try to pickle the extractor
    data = pickle.dumps(crfsuite_citation_extractor)

    # now unpickle it
    unpickled_extractor = pickle.loads(data)

    # get some data for testing
    test = read_iob_files(crf.DATA_DIRS[0], extension='.txt')
    logger.debug(test)
    postags = [
        [
            ("z_POS", token[1])
            for token in instance
        ]
        for instance in test if len(instance) > 0
    ]
    instances = [
        [
            token[0]
            for token in instance
        ]
        for instance in test if len(instance) > 0
    ]

    # run the extractor on the test data
    # TODO: improve the display
    logger.debug(crfsuite_citation_extractor.extract(instances, postags))
    logger.debug(unpickled_extractor.extract(instances, postags))
