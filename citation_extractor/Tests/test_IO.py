# -*- coding: utf-8 -*-
# author: Matteo Romanello, matteo.romanello@gmail.com

import pkg_resources
import logging
from pytest import fixture
from citation_extractor.Utils.IO import annotations2references

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_annotations2references(knowledge_base):
	datadir = ('citation_extractor','data/aph_corpus/goldset/ann/')
	dir = pkg_resources.resource_filename(*datadir)
	files = [file.replace('-doc-1.ann','') for file in pkg_resources.resource_listdir(*datadir) if '.ann' in file]
	references = reduce((lambda x, y: x + y), [annotations2references(file, dir, knowledge_base) for file in files])
	print references
