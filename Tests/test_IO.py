import citation_extractor, pprint
import os,sys, logging
from citation_extractor.Utils import IO

#create logger
logger = logging.getLogger("test")
logger.setLevel(logging.DEBUG)
#create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
#create formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
#add formatter to ch
ch.setFormatter(formatter)
#add ch to logger
logger.addHandler(ch)

# this should be replaced by using command line arguments!
def test_read_JSTOR_dataset(path):
	res = IO.read_jstor_data("%sdata/"%(path))
	logger.info(len(res))
	empty = 0
	tot = 0
	for file in res:
		refs = IO.parse_jstordfr_XML(open(file).read())
		if(len(refs)>0):
			logger.info("%s contains %i references"%(file, len(IO.parse_jstordfr_XML(open(file).read()))))
			pass
		else:
			empty+=1
		tot +=1
	

if __name__ == "__main__":
	if(len (sys.argv)>1):
		test_read_JSTOR_dataset(sys.argv[1])
	else:
		"Usage: <jstor_dataset_path>"