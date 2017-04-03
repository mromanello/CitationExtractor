# -*- coding: utf-8 -*-
# author: Matteo Romanello, matteo.romanello@gmail.com


import codecs
import sys,pprint,re,string,logging
import citation_extractor
from random import *
import xml.dom.minidom as mdom
from pyCTS import CTS_URN

global logger
logger = logging.getLogger(__name__)
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
def read_ann_file_new(fileid, ann_dir, suffix="-doc-1.ann"):
    """
    TODO
    """
    import codecs
    ann_file = "%s%s%s"%(ann_dir,fileid,suffix)
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
        ann_id = cols[0]
        if(u"#" in cols[0]):
            tmp = {
                "ann_id":"%s%s"%(cols[1].split()[0],cols[0])
                ,"anchor":cols[1].split()[1:][0]
                ,"text":cols[2]
            }
            annotations.append(tmp)
        elif(len(cols)==3 and u"T" in cols[0]):
            # is an entity
            ent_count += 1
            ent_type = cols[1].split()[0]
            ranges = cols[1].replace("%s"%ent_type,"")
            entities[cols[0]] = {"ann_id":ann_id
                                ,"entity_type": ent_type
                                ,"offset_start":ranges.split()[0]
                                ,"offset_end":ranges.split()[1]
                                ,"surface":cols[2]}
        elif(len(cols)>=2 and u"R" in cols[0]):
            rel_type, arg1, arg2 = cols[1].split()
            relations[cols[0]] = {"ann_id":ann_id
                                ,"arguments":(arg1.split(":")[1], arg2.split(":")[1])
                                ,"relation_type":rel_type}
    return entities, relations, annotations
def annotations2references(doc_id, directory, kb):
    """
    Read annotations from a brat stand-off file (.ann).
    For each entity and relation keep also the context, i.e. the containing sentences.
    
    TODO: 
    - add author and work labels
    - if annotation is a scope relation, add work- and author-urn
    if annotation is an AWORK, add work- and author-urn
    """
    def find_newlines(text,newline=u'\n'):
        positions = []
        last_position = 0
        if(text.find(newline) == -1):
            return positions
        else:
            while(text.find(newline,last_position+1)>-1):
                last_position = text.find(newline,last_position+1)
                positions.append((last_position,last_position+len(newline)))
            return positions
    def find_linenumber_newlineoffset_for_string(offset_start,offset_end,newline_offsets):
        """
        TODO
        """
        for n,nl_offset in enumerate(newline_offsets):
            #print offset_start,offset_end,nl_offset
            if(offset_start <= nl_offset[0] and offset_end <= nl_offset[0]):
                return (n,newline_offsets[n-1][1],newline_offsets[n][0])
    entities, relations, disambiguations = read_ann_file_new(doc_id, directory)
    fulltext = codecs.open("%s%s%s"%(directory,doc_id,"-doc-1.txt"),"r","utf-8").read()
    newlines = find_newlines(fulltext)
    annotations = []
    for disambiguation in disambiguations:
        annotation = {}
        anchor = disambiguation["anchor"]
        urn = disambiguation["text"]
        ann_id = disambiguation["ann_id"]
        # the annotation refers to a scope relation
        if(anchor.startswith("R")):
            entity_ids =  relations[anchor]["arguments"]
            annotation["annotation_type"] = relations[anchor]["relation_type"].lower()
            arg_entities = [entities[id] for id in entity_ids]
            ann_type = relations[anchor]["relation_type"].lower()
            spanning_lines = [find_linenumber_newlineoffset_for_string(int(entity["offset_start"])
                                                           ,int(entity["offset_end"])
                                                          ,newlines) for entity in arg_entities]
            line_numbers = list(set([line[0] for line in spanning_lines]))
            line_numbers = sorted(line_numbers)
            start = spanning_lines[0][1]
            end = spanning_lines[-1][2]
            if(len(line_numbers)==1):
                sentence = "\n".join(fulltext.split("\n")[line_numbers[0]])
            else:
                sentence = "\n".join(fulltext.split("\n")[line_numbers[0]:line_numbers[1]])
            context = "%s<em>%s</em>%s<em>%s</em>%s"%(fulltext[start:int(arg_entities[0]["offset_start"])]
                        ,fulltext[int(arg_entities[0]["offset_start"]):int(arg_entities[0]["offset_end"])]
                        ,fulltext[int(arg_entities[0]["offset_end"]):int(arg_entities[1]["offset_start"])]
                        ,fulltext[int(arg_entities[1]["offset_start"]):int(arg_entities[1]["offset_end"])]
                        ,fulltext[int(arg_entities[1]["offset_end"]):end]
                        )
            annotation["surface"] = " ".join([entity["surface"] for entity in arg_entities])
            annotation["context"] = context
            annotation["line_number"] = line_numbers[0]
        # the annotation refers to an entity
        elif(anchor.startswith("T")):
            entity = entities[anchor]
            annotation["annotation_type"] = entity["entity_type"].lower()
            line_number,start,end = find_linenumber_newlineoffset_for_string(int(entity["offset_start"])
                                                           ,int(entity["offset_end"])
                                                          ,newlines)
            sentence = fulltext.split("\n")[line_number]
            before_mention = sentence[start-start:int(entity["offset_start"])-start]
            mention = sentence[int(entity["offset_start"])-start:int(entity["offset_end"])-start]
            after_mention = sentence[int(entity["offset_end"])-start:]
            context = "%s<em>%s</em>%s"%(before_mention,mention,after_mention)
            annotation["surface"] = entity["surface"]
            annotation["context"] = context
            annotation["line_number"] = line_number
        annotation["filename"] = doc_id
        annotation["annotation_id"] = ann_id
        annotation["urn"] = urn
        annotation["anchor"] = anchor
        try:
            if(annotation["annotation_type"]=="aauthor"):
                author = kb.get_resource_by_urn(urn)
                annotation["author_label"] = "%s"%author
                annotation["work_label"] = None
                annotation["author_urn"] = str(author.get_urn())
                annotation["work_urn"] = None
            elif(annotation["annotation_type"]=="awork"):
                work = kb.get_resource_by_urn(urn)
                annotation["author_label"] = unicode(work.author)
                annotation["work_label"] = unicode(work)
                annotation["author_urn"] = str(work.author.get_urn())
                annotation["work_urn"] = str(work.get_urn())
            elif(annotation["annotation_type"]=="scope"):
                try:
                    temp = CTS_URN(annotation["urn"]).get_urn_without_passage()
                    resource = kb.get_resource_by_urn(temp)
                    if(isinstance(resource,knowledge_base.surfext.HucitWork)):
                        annotation["author_label"] = unicode(resource.author)
                        annotation["work_label"] = unicode(resource)
                        annotation["author_urn"] = str(resource.author.get_urn())
                        annotation["work_urn"] = str(resource.get_urn())
                    elif(isinstance(resource,knowledge_base.surfext.HucitAuthor)):
                        annotation["author_label"] = unicode(resource)
                        annotation["work_label"] = None
                        annotation["author_urn"] = str(resource.get_urn())
                        annotation["work_urn"] = None
                except Exception as e:
                    annotation["author_label"] = None
                    annotation["work_label"] = None
                    annotation["author_urn"] = None
                    annotation["work_urn"] = None
                    logger.error("Annotation %s raised the following error: %s"%(annotation,e))
            annotations.append(annotation)
        except Exception as e:
            logger.error("The annotations %s raised an error: %s"%(annotation,e))
    logger.info("Read %i annotations from file %s%s"%(len(annotations), directory, doc_id))
    return annotations
def init_logger(log_file=None, loglevel=logging.DEBUG):
    """
    Initialise the logger
    """
    if(log_file !="" or log_file is not None):
        logging.basicConfig(
            filename=log_file
            ,level=loglevel,format='%(asctime)s - %(name)s - [%(levelname)s] %(message)s',filemode='w',datefmt='%a, %d %b %Y %H:%M:%S'
        )
        logger = logging.getLogger(__name__)
        logger.info("Logger initialised")
    else:
        logger = logging.getLogger(__name__)
        logger.setLevel(loglevel)
        ch = logging.StreamHandler()
        ch.setLevel(loglevel)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        logger.info("Logger initialised")
    return logger
def count_tokens(instances):
    """
    """
    return sum([1 for instance in instances for token in instance])
def write_iob_file(instances,dest_file):
    to_write = "\n\n".join(["\n".join(["\t".join(token) for token in instance]) for instance in instances])
    try:
        import codecs
        f = codecs.open(dest_file,'w','utf-8')
        f.write(to_write)
        f.close()
        return True
    except Exception, e:
        raise e
def file_to_instances(inp_file):
    """
    Reads a IOB file a converts it into a list of instances.
    Each instance is a list of tuples, where tuple[0] is the token and tuple[1] contains its assigned label
    Example:
    >>> file_to_instances("data/75-02637.iob")
    """
    f = codecs.open(inp_file,"r","utf-8")
    inp_text = f.read()
    f.close()
    out=[]
    comment=re.compile(r'#.*?')
    for i in inp_text.split("\n\n"):
        inst=[]
        for j in i.split("\n"):
            if(not comment.match(j)):
                temp = tuple(j.split("\t"))
                if(len(temp)>1):
                    inst.append(temp)
        if(inst and len(inst)>1):
            out.append(inst)
    return out
def read_instances(inp_text):
    out=[]
    comment=re.compile(r'#.*?')
    for i in inp_text.split("\n\n"):
        inst=[]
        for j in i.split("\n"):
            if(not comment.match(j)):
                inst.append(j.split("\t"))
        if(inst):
            out.append(inst)
    return out
def out_html(out):
    """docstring for out_xml"""
    import libxml2,libxslt
    from citation_extractor import core
    xsl_path="%s/%s/%s"%(core.determine_path(),"data","reply2html.xsl")
    styledoc = libxml2.parseFile(xsl_path)
    logger=logging.getLogger("CREX.IO")
    logger.info("Using stylesheet \"%s\""%xsl_path)
    style = libxslt.parseStylesheetDoc(styledoc)
    doc = libxml2.parseDoc(out.encode("utf-8"))
    result = style.applyStylesheet(doc, None)
    return style.saveResultToString(result)
def verbose_to_XML(instances):
    """docstring for verbose_to_XML"""
    out = mdom.Document()
    root = out.createElement('reply')
    root.setAttribute("service","crefex")
    root.setAttribute("version",citation_extractor.__version__)
    for inst in instances:  
        ins = out.createElement('instance')
        for t in inst:
            #print type(t['token'])
            try:
                x = unicode(t['token'],"UTF-8")
            except Exception, e:
                print t['token']
                raise e
            
            text = out.createTextNode(x)
            tok = out.createElement('token')
            tok.setAttribute("label",t['label'])
            tok.setAttribute("id",str(t['id']))
            tok.setAttribute("value",x)
            features = out.createElement('features')
            features.appendChild(out.createTextNode(unicode(", ".join(t['features']),"UTF-8")))
            #features.appendChild(out.createTextNode(", ".join(t['features'])))
            probs = out.createElement('tags')
            for tag in t['probs'].keys():
                prob = out.createElement('tag')
                prob.setAttribute("label",tag)
                prob.setAttribute("alpha",str(t['probs'][tag]['alpha']))
                prob.setAttribute("prob",str(t['probs'][tag]['prob']))
                prob.setAttribute("beta",str(t['probs'][tag]['beta']))
                out.createTextNode(tag)
                probs.appendChild(prob)
            tok.appendChild(probs)
            tok.appendChild(features)
            tok.appendChild(text)
            tok.appendChild(out.createTextNode(" "))
            ins.appendChild(tok)
        root.appendChild(ins)
    out.appendChild(root)
    return out.toprettyxml(encoding="UTF-8")
def read_IOB_file(file):
    # instances is a list of lists
    instances=[]
    inp_text = open(file,'r').read()
    comment=re.compile(r'#.*?')
    for n,i in enumerate(inp_text.split("\n\n")):
        # each instance is a list of tuples: [0] id, [1] token, [2] tag
        instance=[]
        for j in i.split("\n"):
            if(not comment.match(j)):
                t= j.split("\t")
                temp=(n+1,t[0],t[1]) 
                instance.append(temp)
        if(instance):
            instances.append(instance)
    return instances
def token_to_string(tok_dict):
    tmp = []
    for k in sorted(tok_dict.keys()):
        tmp.append(tok_dict[k])
    return tmp
def instance_to_string(inst):
    out = []
    for fs in inst:
        tmp = token_to_string(fs)
        out.append("\t".join(tmp))
    return out
def instance_to_IOB(instance):
    pass
def instance_contains_label(instance,labels=["O"]):
    """
    TODO: 
    """
    temp=[token[len(token)-1] for token in instance] 
    res = set(temp).intersection(set(labels))
    if(len(res)==0):
        return False
    else:
        return True
def token_tostring(token):
    string=""
    for count,t in enumerate(token):
            if(count<len(token)-1):
                string+="%s\t"%t
            else:
                string+="%s"%t
    return string
def filter_IOB(instances,tag_name):
    """docstring for filter_IOB"""
    out=[]
    res=[]
    temp = []
    count = 0
    for instance in instances:
        temp = []
        open = False
        for i in instance:
                if(i[2]=='B-%s'%tag_name):
                    temp.append(i[0])
                    open = True
                elif(i[2]=='I-%s'%tag_name):
                    if(open):
                        temp.append(i[0])
                elif(i[2]=='O'):
                    if(open):
                        out.append(temp)
                        temp = []
                        open = False
                    else:
                        pass
        if(len(temp) > 0 and open):
            out.append(temp)
    for r in out:
        res.append(' '.join(r)) 
    return res
def instance_tostring(instance):
    string=""
    for count,t in enumerate(instance):
            if(count!=len(t)):
                string+="%s "%t[0]
            else:
                string+="%s"%t[0]
    return string
def result_to_string(result):
    """
    Tranform the result to a string.
    """
    out=''
    for i,t in enumerate(result):
        out+=t['token']+"/"+t['label']
        if(i<len(result)-1):
            out+=" "
    return out
def eval_results_to_HTML(results,labels=[]):
    """
    Tranform the result to a string.
    """
    out="<html><head><meta http-equiv=\"Content-Type\" content=\"text/html; charset=utf-8\"/> <style type=\"text/css\">div.result{padding:5px}span.token_B-CRF,span.tp{font-weight:bold} span.fn{color:red} span.fp{color:orange}</style></head><body>"
    for n,r in enumerate(results):
        out+="<div class=\"result\">[%s] "%str(n+1)
        for i,t in enumerate(r):
            value=""
            if(t['gt_label']==t['label']):
                if(t['gt_label']=="O"):
                   value='tn'
                else:
                   value='tp'
            else:
                if(t['gt_label']=="O"):
                   value='fp'
                else:
                   value='fn'
            error="%s -&gt; %s"%(t['gt_label'],t['label'])
            out+="<span title=\"%s\" class=\"%s\">%s</span>"%(error,value,t['token'])
            if(i<len(r)-1):
                out+=" "
        out+="</div>"
    out+="</body></html>"
    return out
def results_to_HTML(results,labels=[]):
    """
    Tranform the result to a string.
    """
    #out="<html><head><meta http-equiv=\"Content-Type\" content=\"text/html; charset=utf-8\"/> <style type=\"text/css\">div.result{padding:5px}span.token_B-CRF,span.tp{font-weight:bold} span.fn{color:red} span.fp{color:orange}</style></head><body>"
    out="<div class=\"results\">"
    for n,r in enumerate(results):
        out+="<span title=\"%s\">%s</span>"%(str(r['label']),str(r['token']))
    out+="</div></body>"
    #out+="</html>"
    return out
def parse_jstordfr_XML(inp):
    """
    Describe what the function does.
    """
    out=[]
    xml_exp=re.compile(r'(?:<citation>)(.*?)(?:</citation>)')
    xml_exp=re.compile(r'(?:<reference)(?:.*?>)(.*?)(?:</reference>)')
    cdata_exp=re.compile(r'(?:<!\[CDATA\[)(.*?)(\]\]>)')
    if(xml_exp.search(inp) is not None):
        out=xml_exp.findall(inp)
        #for item in res2:
        #   res1=cdata_exp.match(item)
        #   out.append(res1.groups()[0])
    return out
def parse_dfr_keyword_xml(inp):
    """docstring for fname"""
    out=[]
    xml_exp=re.compile(r'(?:<keyterm frequency=\")(.*?)(?:\" >)(.*?)(?:</keyterm>)')
    if(xml_exp.search(inp) is not None):
        res2=xml_exp.findall(inp)
        out = res2
    return out
def parse_dfr_wordcount_xml(inp):
    """docstring for fname"""
    out=[]
    xml_exp=re.compile(r'(?:<wordcount frequency=\")(.*?)(?:\" >)(.*?)(?:</wordcount>)')
    if(xml_exp.search(inp) is not None):
        res2=xml_exp.findall(inp)
        out = res2
    return out
def parse_one_p_line(inp):
    """
    Describe what the function does.
    """
    return [line for line in inp.split("\n")]
def prepare_for_tagging(file_name,inp="jstor/xml"):
    """
    """
    inp=open(file_name).read()
    prolog="""# File generated from %s
    # The tag to be used to mark up Canonical References are: B-CRF, I-CRF and O (according to the
    # classical IOB format for NER)
    """%(file_name)
    out=""
    out+=prolog
    raw = None
    if(inp=="jstor/xml"):
        raw = parse_jstordfr_XML(inp)
    else:
        raw = parse_one_p_line(inp)
    for i in raw:
        out+=("\n# Original line: %s\n"%i)
        for t in i.split(' '):
            out+="%s\tO\n"%t
        out+="\n"
    return out
def read_jstor_data(dir):   
    """
    Returns a list of strings being the absolute paths to XML files in the dataset
    """
    import os,logging
    logger = logging.getLogger('IO')
    logger.info("Reading %s"%dir)
    xml_files= []
    next =[]
    for x in os.listdir(dir):
        if(os.path.isdir("%s%s"%(dir,x))):
            xml_files += ["%s%s/%s"%(dir,x,y) for y in os.listdir("%s%s/"%(dir,x)) if y.endswith('.xml')]
            next  += read_jstor_data("%s%s/"%(dir,x))
    return xml_files + next 
def read_iob_files(inp_dir,extension=".iob"):
    import glob
    import os
    logger = logging.getLogger("CREX.IO")
    instances = []
    for infile in glob.glob( os.path.join(inp_dir, '*%s'%extension) ):
        temp = file_to_instances(infile)
        instances +=temp
        logger.debug("read %i instances from file %s"%(len(temp),infile))
    return instances
def scan_iob_files(inp_dir):
    import glob
    import os
    import re
    logger = logging.getLogger()
    exp=r'(?:aph_corpus)([0-9\-a-z]+)(?:\.iob)'
    result = {}
    for infile in glob.glob( os.path.join(inp_dir, '*.iob') ):
        fname = os.path.split(infile)[1]
        aph_number = re.match(exp,fname).groups()[0]
        result[fname] = aph_number
    return result
def main():
    insts = read_IOB_file(sys.argv[1])
    tag_name = 'CRF'
    print "The file contains %i instances"%len(insts)
    res=filter_IOB(insts,tag_name)
    print "%i of them have tag %s"%(len(res),tag_name)
    for i in res:
        print i
        print re.sub(r'[^\w]','',i)
if __name__ == "__main__":
    main()
