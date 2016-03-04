# -*- coding: utf-8 -*-
# author: Matteo Romanello, matteo.romanello@gmail.com

from pytest import fixture
import pickle
import logging
import citation_extractor
from citation_extractor.settings import crf
from citation_extractor.core import citation_extractor
from citation_extractor import pipeline
from citation_extractor.Utils.IO import read_iob_files

logging.basicConfig(level=logging.INFO)

@fixture
def crf_citation_extractor(tmpdir):
	"""
	Initialise a citation extractor with CRF model trained on the
	goldset of the APh corpus.
	"""
	crf.TEMP_DIR = str(tmpdir.mkdir('tmp'))+"/"
	crf.OUTPUT_DIR = str(tmpdir.mkdir('out'))+"/"
	crf.LOG_FILE = "%s/extractor.log"%crf.TEMP_DIR
	return pipeline.get_extractor(crf)
def test_pickle_crf_citation_extractor(crf_citation_extractor):
	# try to pickle the extractor
	data = pickle.dumps(crf_citation_extractor) 
	# now unpickle it
	unpickled_extractor = pickle.loads(data)
	# get some data for testing
	test = read_iob_files(crf.DATA_DIRS[0],extension='.txt')
	postags = [[("z_POS",token[1]) for token in instance] for instance in test if len(instance)>0]
	instances = [[token[0] for token in instance] for instance in test if len(instance)>0]
	crf_citation_extractor.extract(instances,postags)
	unpickled_extractor.extract(instances,postags)
def test_multiprocessing_crf_citation_extractor(crf_citation_extractor):
	pass


