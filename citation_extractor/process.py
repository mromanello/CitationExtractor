import sys

def get_taggers(treetagger_dir = '/Applications/treetagger/cmd/',abbrev_file=None):
	"""docstring for create_taggers"""
	from treetagger import TreeTagger
	import os
	os.environ["TREETAGGER"]=treetagger_dir
	lang_codes = {
		'en':('english','latin-1'),
		'it':('italian','utf8'),
		'es':('spanish','utf8'),
		'de':('german','utf8'),
		'fr':('french','utf8'),
	}
	taggers = {}
	for lang in lang_codes.keys():
		try:
			taggers[lang]=TreeTagger(encoding=lang_codes[lang][1],language=lang_codes[lang][0],abbreviation_list=abbrev_file)
		except Exception, e:
			raise e
	return taggers	

def detect_language(text):
	"""
	Detect language of a notice by using the module guess_language.
	The IANA label is returned.
	
	Args:
		text:
			the text whose language is to be detected
	Returns:
		lang:
			the language detected
	"""
	import guess_language
	try:
		lang = guess_language.guessLanguage(text)
		return lang
	except Exception,e:
		print "lang detection raised error \"%s\""%str(e)

def create_instance_tokenizer(train_dirs=[("/Users/56k/phd/code/APh/corpus/txt/",'.txt'),]):
        from nltk.tokenize.punkt import PunktSentenceTokenizer
        import glob
        import os
        import re
        import codecs
        sep = "\n"
        train_text = []
        for dir in train_dirs:
                train_text += [codecs.open(file,'r','utf-8').read() for file in glob.glob( os.path.join(dir[0], '*%s'%dir[1]))]
        return PunktSentenceTokenizer(sep.join(train_text))

def split_sentences(filename,outfilename=None):
	"""	
    sentence tokenization
    text tokenization
    POS-tagging
	"""
	import codecs
	import os.path
	import re
	import sys
	file = codecs.open(filename,'r','UTF-8')
	text = file.read()
	file.close()
	# determine the language
	try:
		sent_tok = create_instance_tokenizer(train_dirs=[("/Users/rromanello/Documents/APh_Corpus/goldset/txt/",'.txt'),])
		sentences = sent_tok.tokenize(text)
		blurb = "\n".join(sentences)
		# the following lines try to correct the most predictable mistakes of the sentence tokenizer 
		recover = r'((.?[A-Z][a-z]+\.?) ([()0-9]+\.?\n?)+)'
		matches = re.findall(recover,blurb)
		for match in matches:
		    # TODO check that match[1] is an abbrev. or an author name
		    blurb = blurb.replace(match[0],match[0].replace("\n"," "))
		new_sentences = blurb.split("\n")
		print >> sys.stderr, "%i sentence breaks were recovered"%(len(sentences)-len(new_sentences))
	except Exception, e:
		raise e
	return new_sentences

def output_to_json(fileid, dir, metadata):
	"""
	TODO
	"""
	import json
	fname="%s%s"%(dir,fileid.replace(".txt",".json"))
	f=open(fname,"w")
	json.dump([metadata],f)
	f.close()
	return

def output_to_oac(fileid, dir, metadata, annotations):
	"""
	TODO
	"""
	# import libraries
	from rdflib import Namespace, BNode, Literal, URIRef,RDF,RDFS
	from rdflib.graph import Graph, ConjunctiveGraph
	from rdflib.plugins.memory import IOMemory
	# declare namespaces
	oac = Namespace("http://www.w3.org/ns/oa#")
	perseus = Namespace("http://data.perseus.org/citations/")
	myanno = Namespace("http://hellespont.org/annotations/jstor")
	store = IOMemory()
	# initialise the graph
	g = ConjunctiveGraph(store=store)
	# bind namespaces
	g.bind("oac",oac)
	g.bind("perseus",perseus)
	g.bind("myanno",myanno)
	for n,ann in enumerate(metadata["citations"]):
	    anno1 = URIRef(myanno["#%i"%n])
	    g.add((anno1, RDF.type,oac["Annotation"]))
	    g.add((anno1, oac["hasTarget"],URIRef("%s%s"%("http://jstor.org/stable/",metadata["doi"]))))
	    g.add((anno1, RDFS.label, Literal(ann["label"])))
	    g.add((anno1,oac["hasBody"],perseus[ann["ctsurn"]]))
	    g.add((anno1,oac["motivatedBy"],oac["linking"]))
	fname="%s%s"%(dir, fileid.replace(".txt",".ttl"))
	f=open(fname,"w")
	f.write(g.serialize(format="turtle"))
	f.close()
	return

def annotations_to_ctsurns(doc_metadata, annotations):
	"""
	TODO
	"""
	from pyCTS import CTS_URN
	doc_metadata["citations"] = []
	for ann in annotations:
	    label = ann[1]
	    cts_urn = CTS_URN(ann[2])
	    temp = {}
	    if(cts_urn.is_range()):
	    	resolv_urn = "%s%s:%s"%("http://data.perseus.org/citations/",cts_urn.get_urn_without_passage(),cts_urn._range_begin)
	    else:
	    	resolv_urn = "%s%s"%("http://data.perseus.org/citations/",ann[2])
	    temp["perseus_uri"]=resolv_urn
	    temp["label"]=label
	    temp["ctsurn"]=str(cts_urn)
	    doc_metadata["citations"].append(temp)
	return doc_metadata

def read_ann_file(fileid,ann_dir):
	"""
	TODO
	"""
	import codecs
	ann_file = "%s%s-doc-1.ann"%(ann_dir,fileid)
	f = codecs.open(ann_file,'r','utf-8')
	data = f.read()
	f.close()
	rows = data.split('\n')
	entities = {}
	ent_count = 0
	relations = {}
	annotations = []
	for row in rows:
	    cols = row.split("\t")
	    if(u"#" in cols[0]):
	    	tmp = cols[1].split()[1:]," ",cols[2]
	    	annotations.append(tmp)
	    elif(len(cols)==3 and u"T" in cols[0]):
	        # is an entity
	        ent_count += 1
	        ent_type, start, end = cols[1].split()
	        entities[cols[0]] = (ent_type,cols[2],start,end)
	    elif(len(cols)>=2 and u"R" in cols[0]):
	        rel_type, arg1, arg2 = cols[1].split()
	        relations[cols[0]] = (rel_type,arg1, arg2)
	    else:
	        pass
	res_annotations = []
	for annot in annotations:
		rel_id,label,urn = annot
		try:
			rel = relations[rel_id[0]]
			arg1 = rel[1].split(":")[1]
			arg2 = rel[2].split(":")[1]
			label = "%s %s"%(entities[arg1][1],entities[arg2][1])
			res_annotations.append([rel_id[0],label,urn])
		except Exception, e:
			entity = entities[rel_id[0]]
			res_annotations.append([rel_id[0],entity[1],urn])
	return entities, relations, res_annotations

def extract_citations(extractor,outputdir,filename,iob_sentences,outfilename=None):
	"""docstring for extract_citations"""
	# this is the important bit which performs the citation extraction
	import sys
	import os
	from citation_extractor.eval import IO

	result,out_fname = None, ""
	if(outfilename is None):
		path,name = os.path.split(filename)
		out_fname = '%s%s'%(outputdir,name)
	else:
		out_fname = outfilename
	try:
		postags = [[("z_POS",token[1]) for token in instance] for instance in iob_sentences if len(instance)>0]
		instances = [[token[0] for token in instance] for instance in iob_sentences if len(instance)>0]
		result = extractor.extract(instances, postags)
		output = []
		for i,res in enumerate(result):
		    temp = []
		    for n,d_res in enumerate(res):
		        temp.append((res[n]["token"], postags[i][n][1], res[n]["label"]))
		    output.append(temp)
		try:
		    IO.write_iob_file(output,out_fname)
		    print >> sys.stderr, "Output successfully written to file \"%s\""%out_fname
		    return result,out_fname
		except Exception, e:
		    raise e
	except Exception, e:
		raise e

def extract_relationships(entities):
	"""
	TODO: implement properly the pseudocode!
	"""
	relations = {}
	arg1 = None
	arg2 = None
	# why it's important to sort this way the entities?
	items = entities.items()
	items.sort(key=lambda x:int(x[1][2]))
	for item in items:
		entity_type,entity_label,entity_start,entity_end = item[1]
		if(entity_type!="REFSCOPE"):
			arg1 = item[0]
			arg2 = None
		else:
			arg2 = item[0]
			if(arg1 is not None):
				rel_id = "R%s"%(len(relations.keys())+1)
				relations[rel_id] = (arg1,arg2)
				print "Detected relation %s"%str(relations[rel_id])

	return relations

def extract_relationships_old(doc_tree):
	"""
	TODO
	"""
	def traverse(t,n):
	    global token_count,context
	    try:
	        t.label()
	    except AttributeError:
	        token_count+=1
	        print "%s / %s"%(t[0],t[1])
	    else:
	        if(t.label()!='S'):
	            print context
	        	# clear the context
	            if(t.label()=='AAUTHOR' or t.label()=='REFAUWORK'):
	                context = {}
	            start = token_count
	            end = token_count
	            for leave in t.leaves():
	                end +=1
	            #text=[doc_tree[n][x] for x in range(start,end)]
	            #print t.label(),':'," ".join(text),"[start %i, end %i]"%(start,end)
	            entity_num = len(entities.keys())
	            entities[entity_num]=(t.label(),start,end)
	            context[t.label()]=entity_num
	            if(t.label()=='REFSCOPE'):
	                #print context
	                relation_num = len(relations.keys())+1
	                if(context.has_key("REFAUWORK")):
	                    relations["R%s"%relation_num]=(context["REFAUWORK"],context["REFSCOPE"])
	                elif(context.has_key("AWORK")):
	                	relations["R%s"%relation_num]=(context["AWORK"],context["REFSCOPE"])
	                elif(context.has_key("AAUTHOR")):
	                	relations["R%s"%relation_num]=(context["AAUTHOR"],context["REFSCOPE"])
	            for child in t:
	                traverse(child,n)
	        else:
	            for child in t:
	                traverse(child,n)
	    return context
	entities = {}
	relations = {}
	annotations = {}
	for n_sentence,sentence in enumerate(doc_tree):
		global token_count,context
		context = {}
		token_count=0
		traverse(sentence,n_sentence)
	return entities,relations

def save_scope_relationships(fileid, ann_dir, relations, entities):
	"""
	appends relationships (type=scope) to an .ann file. 
	"""
	import codecs
	ann_file = "%s%s-doc-1.ann"%(ann_dir,fileid)
	keys = relations.keys()
	keys.sort(key=lambda k:(k[0], int(k[1:])))
	result = "\n".join(["%s\tScope Arg1:%s Arg2:%s"%(rel,relations[rel][0],relations[rel][1]) for rel in keys])
	try:
		f = codecs.open(ann_file,'r','utf-8')
		hasblankline = f.read().endswith("\n")
		f.close()
		f = codecs.open(ann_file,'a','utf-8')
		if(not hasblankline):
			f.write("\n")
		f.write(result)
		f.close()
		print >> sys.stderr,"Written %i relations to%s"%(len(relations),ann_file)
	except Exception, e:
		raise e
	return result

def clean_relations_annotation(fileid, ann_dir, entities):
	"""
	overwrites relationships (type=scope) to an .ann file. 
	"""
	import codecs
	import sys
	ann_file = "%s%s-doc-1.ann"%(ann_dir,fileid)
	keys = entities.keys()
	keys.sort(key=lambda k:(k[0], int(k[1:])))
	result = "\n".join(["%s\t%s %s %s\t%s"%(ent,entities[ent][0],entities[ent][2],entities[ent][3],entities[ent][1]) for ent in keys])
	#result2 = "\n".join(["%s\tScope Arg1:%s Arg2:%s"%(rel,relations[rel][0],relations[rel][1]) for rel in keys])
	try:
		f = codecs.open(ann_file,'w','utf-8')
		f.write(result)
		f.close()
		print >> sys.stderr,"Cleaned relations annotations from %s"%ann_file
	except Exception, e:
		raise e
	return result

def remove_all_annotations(fileid, ann_dir):
	import codecs
	ann_file = "%s%s-doc-1.ann"%(ann_dir,fileid)
	entities, relations, annotations = read_ann_file(fileid, ann_dir)

	entity_keys = entities.keys()
	entity_keys.sort(key=lambda k:(k[0], int(k[1:])))
	entities_string = "\n".join(["%s\t%s %s %s\t%s"%(ent,entities[ent][0],entities[ent][2],entities[ent][3],entities[ent][1]) for ent in entity_keys])

	relation_keys = relations.keys()
	relation_keys.sort(key=lambda k:(k[0], int(k[1:])))
	relation_string = "\n".join(["%s\tScope Arg1:%s Arg2:%s"%(rel,relations[rel][1].replace('Arg1:',''),relations[rel][2].replace('Arg2:','')) for rel in relation_keys])
	
	try:
		f = codecs.open(ann_file,'w','utf-8')
		f.write(entities_string)
		f.write("\n")
		f.write(relation_string)
		f.close()
		print >> sys.stderr,"Cleaned all relations annotations from %s"%ann_file
	except Exception, e:
		raise e
	return

def tokenize(sentences,taggers, outfilename=None):
	"""
	Detect language of a notice by using the module guess_language.
	The IANA label is returned.
	
	Args:
		text:
			the text whose language is to be detected
	Returns:
		lang:
			the language detected
	"""
	import codecs
	from citation_extractor.Utils import IO
	import os.path
	import sys
	
	text = "\n".join(sentences)
	# determine the language
	lang = detect_language(text)
	if(lang=="la"):
		lang = "en**"
	# split into sentences, tokenize and POS-tag
	if(lang=="UNKNOWN"):
		lang = "en*"
	print >> sys.stderr,"Language detected is %s"%lang
	iob = []
	for n,sent in enumerate(sentences):
		tok_lang = lang
		if(tok_lang in ["en*","en**"]):
			tok_lang = "en"
		tmp = [result[:2] for result in taggers[tok_lang].tag(sent)]
		#print >> sys.stderr,"Tokenized sentence %i / %i"%(n,len(sentences))
		iob.append(tmp)
	return lang,iob

def preprocess(filename,taggers, outputdir, outfilename=None,split_sentence=False):
	"""	
    sentence tokenization
    text tokenization
    POS-tagging
	"""
	import codecs
	from citation_extractor.Utils import IO
	import os.path
	import sys
	
	file = codecs.open(filename,'r','UTF-8')
	text = file.read()
	file.close()
	# split into sentences
	if(split_sentence):
		sentences = split_sentences(filename)
	else:
		sentences = [text.replace("\n"," ")]
	print >> sys.stderr, "Text was split into %i sentences"%len(sentences)
	# tokenize
	lang, iob = tokenize(sentences,taggers)
	print >> sys.stderr, "%i sentences were tokenized into %i tokens"%(len(sentences),IO.count_tokens(iob))
	# save the intermediate output
	if(outfilename is None):
		path,name = os.path.split(filename)
		out_fname = '%siob/%s'%(outputdir,name)
	else:
		out_fname = outfilename
	try:
	    IO.write_iob_file(iob,out_fname)
	    print >> sys.stderr,"IOB output successfully written to file \"%s\""%out_fname
	except Exception, e:
	    print "Failed while writing IOB output to file \"%s\", %s"%(out_fname,e)
	return lang, iob, out_fname

def save_scope_annotations(fileid, ann_dir, annotations):
	"""
	this method expects a tuple `t` where
	t[0] is the ID of the entity/relation the annotation is about
	t[1] is the label (it doesn't get written to the file)
	t[2] is the URN, i.e. the content of the annotation
	if t[2] is None the annotation is skipped

	"""
	ann_file = "%s%s-doc-1.ann"%(ann_dir,fileid)
	file_content = open(ann_file,'r').read()
	file = open(ann_file,'a')
	if(not (file_content.endswith('\n') or file_content.endswith('\r'))):
		file.write("\n")
	for n,annot in enumerate(annotations):
		if(annot[2] is not None):
			file.write("#%i\tAnnotatorNotes %s\t%s\n"%(n,annot[0],annot[2]))
		else:
			print >> sys.stderr, "The annotation \"%s\" in %s is None, therefore was not written to file"%(annot[1],fileid)
	file.close()
	return

def tostandoff(iobfile,standoffdir,brat_script):
	"""
	Converts the .iob file with NE annotation into standoff markup.
	"""
	import sys
	import os
	try:
		cmd = "python %s -o %s %s"%(brat_script,standoffdir,iobfile)
		os.popen(cmd).readlines()
		print >> sys.stderr,".ann output written successfully."
	except Exception, e:
		raise e

def disambiguate_relations(citation_matcher, relations,entities,docid):
	"""

	TODO

	Returns:
		 (u'R5', u'[ Verg. ] catal. 47s', u'urn:cts:TODO:47s')
	"""
	import re
	print >> sys.stderr, "Disambiguating the %i relation contained in %s..."%(len(relations), docid)
	result = []
	for relation in relations:
	    relation_type = relations[relation][0]
	    arg1 = relations[relation][1].split(":")[1]
	    arg2 = relations[relation][2].split(":")[1]
	    refauwo=entities[arg1][1]
	    refauwo=re.sub("[\(, \)]","",refauwo) # TODO move this to CitationParser
	    scope = entities[arg2][1]
	    scope = re.sub("\.$","",scope)
	    scope = re.sub("\,$","",scope)
	    scope = re.sub("[\(, \)]","",scope)
	    try:
	        urn = citation_matcher.disambiguate(refauwo,scope)[0]
	        result.append((relation,"%s %s"%(refauwo,scope),urn))
	    except Exception, e:
	    	normalized_scope = scope
	    	try:
	    		normalized_scope = citation_matcher._citation_parser.parse(scope)
	    		normalized_scope = citation_matcher._format_scope(normalized_scope[0]['scp'])
	    	except Exception, e:
	    		print >> sys.stderr, e
	        result.append((relation,"%s %s"%(refauwo,scope),None))
	return result

def disambiguate_entities(citation_matcher,entities,docid,min_distance_threshold,max_distance_threshold):
	print >> sys.stderr, "Disambiguating the %i entities contained in %s..."%(len(entities), docid)
	result = []
	distance_threshold = min_distance_threshold
	for entity in entities:
		entity_type = entities[entity][0]
		if entity_type == "AAUTHOR":
			string = entities[entity][1]
			matches = citation_matcher.matches_author(string,True,distance_threshold)
			while(matches is None and distance_threshold <= max_distance_threshold):
				distance_threshold+=1
				matches = citation_matcher.matches_author(string,True,distance_threshold)
			if(matches is not None):
				result.append((entity, string ,matches[0][0]))
		elif(entity_type == "AWORK"):
			string = entities[entity][1]
			matches = citation_matcher.matches_work(string,True,distance_threshold)
			while(matches is None and distance_threshold <= max_distance_threshold):
				distance_threshold+=1
				matches = citation_matcher.matches_work(string,True,distance_threshold)
			if(matches is not None):
				result.append((entity, string ,matches[0][0]))
	return result