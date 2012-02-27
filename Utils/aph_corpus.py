import urllib,json
download_dir = "aph_data/"
download_dir = "aph_data_C1/"
#collection_url = "http://cwkb.webfactional.com/aph_corpus/collections/100_positive/gold"
collection_url = "http://cwkb.webfactional.com/aph_corpus/collections/C1/gold"
data = json.loads(urllib.urlopen(collection_url).read())
print "Downloaded list of file URLs. %i in total"%len(data)
files = [urllib.urlopen("%s%s"%("http://cwkb.webfactional.com",id)).read() for id in data]
print "Fecthed %i files."%len(files)
"""
for i,f in enumerate(data):
	fname = f.replace("/iob",".iob").replace("/","")
	try:
		print "Writing %s to disk"%fname
		file = open("%s%s"%(download_dir,fname),"w")
		file.write(files[i])
		file.close()
		print "File %s written to disk."%fname
	except IOError as (errno, strerror):
		print "there was a problem writing file %s"%fname
		print "I/O error({0}): {1}".format(errno, strerror)
print "done"
"""

all_in_one_file = open("%s%s"%(download_dir,"all_in_one.iob"),"w")
all_in_one = "\n".join(files)
all_in_one_file.write(all_in_one)
all_in_one_file.close()