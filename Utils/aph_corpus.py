import urllib,json

def save_collection_allinone(download_dir,data):
	fname = "%s%s"%(download_dir,"all_in_one.iob")
	try:
		all_in_one_file = open(fname,"w")
		all_in_one = "\n\n".join(data)
		all_in_one_file.write(all_in_one)
		all_in_one_file.close()
		return fname
	except IOError as (errno, strerror):
		print "there was a problem writing file %s"%fname
		print "I/O error({0}): {1}".format(errno, strerror)

def save_collection_onebyone(download_dir,files,data):
	"""
	>>> coll = download_collection()
	>>> print len(coll)
	2
	>>> save_collection_allinone("/Users/56k/phd/code/APh/experiments/20121603/train/",coll[0],coll[1])
	/Users/56k/phd/code/APh/experiments/20121603/train/all_in_one.iob
	>>> save_collection_onebyone("/Users/56k/phd/code/APh/experiments/20121603/train/",coll[0],coll[1])
	"""
	out_files = []
	try:
		for i,f in enumerate(files):
			fname = "%s%s"%(download_dir,f.replace("/iob",".iob").replace("/",""))
			#print "Writing %s to disk"%fname
			file = open(fname,"w")
			file.write(data[i])
			file.close()
			#print "File %s written to disk."%fname
			#print "done"
			out_files.append(fname)
		return out_files
	except IOError as (errno, strerror):
		print "there was a problem writing file %s"%fname
		print "I/O error({0}): {1}".format(errno, strerror)

def download_collection(coll_url="http://cwkb.webfactional.com/aph_corpus/collections/C1/gold"):
	"""
	
	"""
	import urllib,json
	data = json.loads(urllib.urlopen(coll_url).read()) # the json returned here is just a list of file names
	return (data,[urllib.urlopen("%s%s"%("http://cwkb.webfactional.com",id)).read() for id in data])

def tokenise_collection(coll_name):
	return
	
if __name__ == "__main__":
	import doctest
	doctest.testmod(verbose=True)