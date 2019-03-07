# -*- coding: utf-8 -*-
# author: Matteo Romanello, matteo.romanello@gmail.com

from __future__ import print_function
import pdb
import logging
import os
import codecs
#import knowledge_base
import glob
import sys,pprint,re,string
import pandas as pd
from random import *
from pyCTS import CTS_URN
from .strmatching import StringUtils

#import citation_extractor
#import xml.dom.minidom as mdom

global logger
logger = logging.getLogger(__name__)

NIL_ENTITY = "urn:cts:GreekLatinLit:NIL"

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
    return entities, relations, res_annotations # superseded by `read_ann_file_new` TODO: delete

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

    import knowledge_base

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
