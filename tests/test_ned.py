# -*- coding: utf-8 -*-
# author: Matteo Romanello, matteo.romanello@gmail.com

import pytest
import logging
import pickle

logger = logging.getLogger(__name__)

@pytest.mark.skip
def test_pickle_kb(knowledge_base):
	"""
	Tests whether instances of `KnowledgeBase` can be pickled.
	"""
	pickled_kb = pickle.dumps(knowledge_base)
	unpickled_kb = pickle.loads(pickled_kb)
	logger.info("The KnowledgeBase contains %i author names"%len(unpickled_kb.author_names))

@pytest.mark.skip
def test_pickle_citation_matcher(citation_matcher):
	"""
	Tests whether instances of `CitationMatcher` (and contained objects) can be pickled.
	"""
	pickled_citation_matcher = pickle.dumps(cm)
	unpickled_citation_matcher = pickle.loads(pickled_citation_matcher)
