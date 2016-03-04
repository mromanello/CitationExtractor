# -*- coding: utf-8 -*-
# author: Matteo Romanello, matteo.romanello@gmail.com

import logging
import pkg_resources
from citation_extractor.relex import prepare_for_training

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_prepare_training():
	dir = pkg_resources.resource_filename('citation_extractor','data/aph_corpus/goldset/ann/')
	files = [file.replace('-doc-1.ann','') for file in pkg_resources.resource_listdir('citation_extractor','data/aph_corpus/goldset/ann/') if '.ann' in file]
	logger.debug(prepare_for_training(files[1],dir))
	