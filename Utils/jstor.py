import os,sys,pprint,codecs,logging
import re
import citation_extractor
from citation_extractor import Utils
from citation_extractor.Utils import IO
import urllib


_default_username_ = ""
_default_password_ = ""

def read_jstor_rdf_catalog(file_path):
	"""
	TODO
	"""
	from lxml import etree
	print file_path
	file=open(file_path)
	fcont=file.read()
	file.close()
	res = etree.fromstring(fcont)
	children = list(res)
	res = []
	for el in children:
		el_children = list(el)
		dcns = "http://purl.org/dc/elements/1.1/"
		el_res = {}
		for n in el.getchildren():
			if(n.tag=="{%s}identifier"%dcns):
				print "%s: %s"%(n.tag,n.text)
				el_res["urn"]=n.text
			elif(n.tag=="{%s}title"%dcns):
				print "%s: %s"%(n.tag,n.text)
				el_res["title"]=n.text
			elif(n.tag=="{%s}identifer"%dcns):
				print "%s: %s"%(n.tag,n.text)
				el_res["id"]=n.text
			elif(n.tag=="{%s}relation"%dcns):
				print "%s: %s"%(n.tag,n.text)
				el_res["relation"]=n.text
			elif(n.tag=="{%s}creator"%dcns):
				print "%s: %s"%(n.tag,n.text)
				el_res["author"]=n.text
			elif(n.tag=="{%s}publisher"%dcns):
				print "%s: %s"%(n.tag,n.text)
				el_res["publisher"]=n.text
			elif(n.tag=="{%s}date"%dcns):
				print "%s: %s"%(n.tag,n.text)
				el_res["date"]=n.text
			elif(n.tag=="{%s}type"%dcns):
				print "%s: %s"%(n.tag,n.text)
				el_res["type"]=n.text
		return

def get_jstor_info(doc_id=None,reqs = ["wordcounts","references","keyterms"],username=_default_username_,password=_default_password_):
	#reqs = ["wordcounts","bigrams","trigrams","quadgrams","references","keyterms"]
	logger = logging.getLogger("jstor")
	res = {}
	if(doc_id is not None):
		for r in reqs:
			logger.info("Getting %s for %s"%(r,doc_id))
			res[r] = get_jstor(r,doc_id,username,password)
	return res

	
def get_jstor(request,doc_id=None,username=_default_username_,password=_default_password_):
	base_url = "dfr.jstor.org/resource/"
	result = None
	if(doc_id is not None):
		try:
			url = "http://%s:%s@%s%s?view=%s"%(username,password,base_url,doc_id,request)
			result = urllib.urlopen(url)
			url = "http://%s%s?view=%s"%(base_url,doc_id,request)
		except Exception, e:
			raise e
	if(result is not None):
		return url,result.read()
	else:
		return None,result

def read_jstor_csv_catalog(file_path):
	import csv,re
	indexes = {'JOURNALTITLE':{},'PUBDATE':{},'TYPE':{}}
	ids={}
	res = list(csv.DictReader(codecs.open(file_path, "r", "utf-8" )))
	for n in range(len(res)):
		i=res[n]
		if(i['VOLUME']==""):
			i['VOLUME']=0
		ids[i['ID']] = i
		for key in indexes.keys():
			if(key=="PUBDATE"):
				r=re.compile(r'[A-Za-z0-9\-,.\s]+ \n?([0-9]{4})')
				r2=re.compile(r'([0-9]{4}/[0-9]{4})')
				if(r.match(i[key])):
					i[key] = r.search(i[key]).group(1)
				elif(r2.match(i[key])):
					i[key] = r2.search(i[key]).group(1)
			if(indexes[key].has_key(i[key])):
				indexes[key][i[key]].append(i['ID'])
			else:
				indexes[key][i[key]] = []
				indexes[key][i[key]].append(i['ID'])
	#pprint.pprint(ids)
	for i in indexes:
		for n in indexes[i].keys():
			#print "%s: count=%i"%(n,len(indexes[i][n]))
			pass
	return ids,indexes

if __name__ == "__main__":
	if(len (sys.argv)>1):
		res=[]
		res = read_jstor_csv_catalog("%scitations.csv"%sys.argv[1])
		ids = res[0]
		paths = IO.read_jstor_data(sys.argv[1])
		fnames=[]
		for p in paths:
			path,fn = os.path.split(p)
			fn = fn.replace('_','/').replace('.xml','')
			fnames.append(fn)
		# explain
		commons = set(ids).intersection(set(fnames))
		print len(commons)
	else:
		print "Usage: <jstor_dataset_path>"
