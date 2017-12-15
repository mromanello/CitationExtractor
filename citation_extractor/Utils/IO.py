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

def read_ann_file_new(fileid, ann_dir, suffix="-doc-1.ann"):
    """
    TODO
    """
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
            # it's a text annotation
            tmp = {
                "ann_id":"%s%s"%(cols[1].split()[0],cols[0])
                ,"anchor":cols[1].split()[1:][0]
                ,"text":cols[2]
            }
            annotations.append(tmp)

        elif(len(cols)==3 and u"T" in cols[0]):
            # it's an entity
            ent_count += 1
            ent_type = cols[1].split()[0]
            ranges = cols[1].replace("%s"%ent_type,"")
            entities[cols[0]] = {"ann_id":ann_id
                                ,"entity_type": ent_type
                                ,"offset_start":ranges.split()[0]
                                ,"offset_end":ranges.split()[1]
                                ,"surface":cols[2]}

        elif(len(cols)>=2 and u"R" in cols[0]):
            # it's a relation
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

def count_tokens(instances):
    """
    """
    return sum([1 for instance in instances for token in instance])

def write_iob_file(instances, dest_file):
    to_write = "\n\n".join(["\n".join(["\t".join(token) for token in instance]) for instance in instances])
    try:
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

def read_iob_files(inp_dir, extension=".iob"):
    instances = []
    for infile in glob.glob( os.path.join(inp_dir, '*%s'%extension) ):
        temp = file_to_instances(infile)
        instances +=temp
        logger.debug("read %i instances from file %s"%(len(temp),infile))
    return instances

def token_to_string(tok_dict):
    """
    """
    return [tok_dict[k] for k in sorted(tok_dict.keys())]

def instance_to_string(instance):
    """
    TODO
    """
    return ["\t".join(token_to_string(feature_set)) for feature_set in instance]

def sort_mentions_by_appearance(entities, relations):
    """
    Return an ordered sequence of entity/relation IDs, in the same order as they appear in the document.

    :param entities:
        The dictionary of the entities in the document.
    :param relations:
        The dictionary of the relations in the document.
    :return:
        A list containing the ids in order.
    """
    entities_id_set = set(entities.keys())
    by_offset = []

    for k, v in relations.iteritems():
        entity_id_0 = v['arguments'][0]
        entity_id_1 = v['arguments'][1]
        entities_id_set.discard(entity_id_0)
        entities_id_set.discard(entity_id_1)
        entity_0 = entities[entity_id_0]
        entity_1 = entities[entity_id_1]
        first_start = min(int(entity_0['offset_start']), int(entity_1['offset_start']))
        second_start = max(int(entity_0['offset_start']), int(entity_1['offset_start']))
        offset = float(str(first_start) + '.' + str(second_start))
        t = (k, offset)
        by_offset.append(t)

    for eid in entities_id_set:
        entity_0 = entities[eid]
        offset = float(entity_0['offset_start'])
        t = (eid, offset)
        by_offset.append(t)

    by_offset = [id for id, offset in sorted(by_offset, key=lambda (id, offset): offset)]
    return by_offset

def load_brat_data(extractor, knowledge_base, postaggers, aph_ann_files, aph_titles):
    """
    Utility function to load a set of brat documents and prepare them
    in a format suitable for processing (typically when carrying out the evaluation or the training).

    :param citation_extractor: instance of `core.citation_extractor`
    :parm knowledge_base: instance of `knowledge_base.KnowledgeBase`
    :param aph_ann_files: a tuple: [0] the base directory; [1] a list of file names
    :param aph_titles: `pandas.DataFrame` with column 'title'
    :return: a `pandas.DataFrame` (columns: 'surface', 'surface_norm', 'scope', 'type',
        'other_mentions', 'prev_mentions', 'urn', 'urn_clean','doc_id', 'doc_title',
        'doc_title_mentions', 'doc_title_norm', 'doc_text', 'sentence_start', 'sentence_end')

    """
    from citation_extractor.pipeline import extract_entity_mentions

    cols = ['surface', 'surface_norm', 'scope', 'type', 'other_mentions', 'prev_mentions', 'urn', 'urn_clean',
            'doc_id', 'doc_title', 'doc_title_mentions', 'doc_title_norm', 'doc_text', 'sentence_start', 'sentence_end']
    df_data = pd.DataFrame(dtype='object', columns=cols)

    # Read all annotated files
    ann_dir, files = aph_ann_files
    for filename in files:
        if filename.endswith('.ann'):
            logger.debug('Reading file: {}'.format(filename))

            # Read doc annotations
            file_suffix = filename.replace('-doc-1.ann', '')
            entities, relations, disambiguations = read_ann_file_new(file_suffix, ann_dir + '/')
            # Read doc text
            doc_text = None
            filename_text = filename.replace('.ann', '.txt')
            with open(os.path.join(ann_dir, filename_text)) as f:
                doc_text = f.read()
                doc_text = unicode(doc_text, 'utf-8')
            logger.debug(u'Document text: {}'.format(doc_text))
            doc_newlines = _find_newlines(doc_text)

            # Get title
            doc_title = None
            file_id = file_suffix.replace('.txt', '')
            if file_id in aph_titles.index:
                doc_title = aph_titles.loc[file_id, 'title']
                doc_title = unicode(doc_title, 'utf-8')
            logger.debug(u'Document title: {}'.format(doc_title))

            try:
                # Extract mentions from the title, list of (type, surface) tuples
                doc_title_extracted_mentions = extract_entity_mentions(doc_title, extractor, postaggers, norm=True)
            except Exception, e:
                doc_title_extracted_mentions = []
                print(e)
                print(doc_title)
                print(file_id)

            # Normalize title
            doc_title_norm = StringUtils.normalize(doc_title)

            # Order the appearance of the mentions in the doc
            ordered_mentions = sort_mentions_by_appearance(entities, relations)
            logger.debug('Mentions appearance: {}'.format(ordered_mentions))

            # Rearrange disambiguations
            disambiguations_new = dict(map(lambda e: (e['anchor'], e['text']), disambiguations))

            prev_entities = []
            for mention_id in ordered_mentions:
                # Note: added utf-8 encoding after new error 
                mention_data_id = file_id + '-' + mention_id.encode('utf-8')
                mention_urn = NIL_ENTITY
                clean_urn = mention_urn
                mention_surface = None
                mention_scope = None
                mention_type = None
                sentence_start = None
                sentence_end = None

                # It's a relation
                if mention_id.startswith('R'):
                    relation = relations[mention_id]

                    # Unpack the relation
                    entity_0 = entities[relation['arguments'][0]]
                    entity_1 = entities[relation['arguments'][1]]

                    # Sanity check for types of relation members
                    no_refscope = ['AAUTHOR', 'AWORK', 'REFAUWORK']
                    if entity_0['entity_type'] in no_refscope and entity_1['entity_type'] == 'REFSCOPE':
                        pass

                    elif entity_1['entity_type'] in no_refscope and entity_0['entity_type'] == 'REFSCOPE':
                        logger.warning('Swapped entities in relation {} in doc {}'.format(mention_id, filename))
                        entity_0 = entities[relation['arguments'][1]]
                        entity_1 = entities[relation['arguments'][0]]

                    else:
                        logger.error('Unknown types in relation {} in doc {}'.format(mention_id, filename))
                        continue

                    # Update fields
                    if mention_id in disambiguations_new:
                        mention_urn = disambiguations_new[mention_id]
                    mention_surface = entity_0['surface']
                    mention_scope = entity_1['surface']
                    mention_type = entity_0['entity_type']

                    if entity_0["offset_start"] > entity_1["offset_start"]:
                        sentence_start = _find_linenumber_by_offset(int(entity_1["offset_start"])
                                                                , int(entity_1["offset_end"])
                                                                , doc_newlines)[0]
                        sentence_end = _find_linenumber_by_offset(int(entity_0["offset_start"])
                                                                , int(entity_0["offset_end"])
                                                                , doc_newlines)[0]
                    else:
                        sentence_start = _find_linenumber_by_offset(int(entity_0["offset_start"])
                                                                , int(entity_0["offset_end"])
                                                                , doc_newlines)[0]
                        sentence_end = _find_linenumber_by_offset(int(entity_1["offset_start"])
                                                                , int(entity_1["offset_end"])
                                                                , doc_newlines)[0]


                # It's a non-relation
                elif mention_id.startswith('T'):
                    entity = entities[mention_id]

                    # Avoid to disambiguate the mention if it's a REFSCOPE (alone)
                    if entity['entity_type'] == 'REFSCOPE':
                        logger.warning('Lonely REFSCOPE with id: {} in doc: {}'.format(mention_id, filename))
                        continue

                    # Update fields
                    if mention_id in disambiguations_new:
                        mention_urn = disambiguations_new[mention_id]
                    mention_surface = entity['surface']
                    mention_type = entity['entity_type']
                    mention_offset_start = int(entity['offset_start'])
                    mention_offset_end = int(entity['offset_end'])
                    sentence_start = _find_linenumber_by_offset(mention_offset_start
                                                            , mention_offset_end
                                                            , doc_newlines)[0]
                    sentence_end = sentence_start

                else:
                    logger.error('Unknown mention id: {} in doc {}'.format(mention_id, filename))
                    continue

                # Get clean URN (without passage), skip if non-valid
                if mention_urn != NIL_ENTITY:
                    try:
                        cts_urn = CTS_URN(mention_urn)
                        clean_urn = cts_urn.get_urn_without_passage()
                        knowledge_base.get_resource_by_urn(clean_urn)
                    except Exception, e:
                        logger.error(e)
                        logger.warning('Failed parsing the URN: |{}| at: {}'.format(mention_urn, file_id))
                        continue

                # Keep track of previous mentions
                mention_prev_entities = list(prev_entities) # copy
                prev_entities.append(mention_data_id)

                df_data.loc[mention_data_id, 'surface'] = mention_surface
                df_data.loc[mention_data_id, 'sentence_start'] = sentence_start
                df_data.loc[mention_data_id, 'sentence_end'] = sentence_end
                df_data.loc[mention_data_id, 'surface_norm'] = StringUtils.normalize(mention_surface)
                df_data.loc[mention_data_id, 'scope'] = mention_scope
                df_data.loc[mention_data_id, 'type'] = mention_type
                df_data.loc[mention_data_id, 'prev_mentions'] = mention_prev_entities
                df_data.loc[mention_data_id, 'doc_id'] = file_id
                df_data.loc[mention_data_id, 'doc_title'] = doc_title
                df_data.loc[mention_data_id, 'doc_title_mentions'] = doc_title_extracted_mentions
                df_data.loc[mention_data_id, 'doc_title_norm'] = doc_title_norm
                df_data.loc[mention_data_id, 'doc_text'] = doc_text
                df_data.loc[mention_data_id, 'urn'] = mention_urn
                df_data.loc[mention_data_id, 'urn_clean'] = clean_urn

            # Add successfully parsed mentions of the doc to other_mentions field of each mention of the doc
            for m_id in prev_entities:
                other_mentions = list(prev_entities)
                other_mentions.remove(m_id)
                df_data.loc[m_id, 'other_mentions'] = other_mentions

    nb_total = df_data.shape[0]
    nb_authors = df_data[df_data['type'] == 'AAUTHOR'].shape[0]
    nb_works = df_data[df_data['type'] == 'AWORK'].shape[0]
    nb_refauworks = df_data[df_data['type'] == 'REFAUWORK'].shape[0]
    nb_nil = df_data[df_data['urn'] == NIL_ENTITY].shape[0]

    logger.info('DATA - Total: {}, AAUTHORS: {} ({:.1f}%), AWORK: {} ({:.1f}%), REFAUWORK: {} ({:.1f}%), NIL: {} ({:.1f}%)'.format(
        nb_total,
        nb_authors,
        float(nb_authors) / max(nb_total, 1) * 100,
        nb_works,
        float(nb_works) / max(nb_total, 1) * 100,
        nb_refauworks,
        float(nb_refauworks) / max(nb_total, 1) * 100,
        nb_nil,
        float(nb_nil) / max(nb_total, 1) * 100
    ))

    return df_data

def _find_newlines(text, newline=u'\n'):
    """
    TODO
    """
    positions = []
    last_position = 0
    if(text.find(newline) == -1):
        return positions
    else:
        while(text.find(newline,last_position+1)>-1):
            last_position = text.find(newline,last_position+1)
            positions.append((last_position,last_position+len(newline)))
        return positions

def _find_linenumber_by_offset(offset_start, offset_end, newline_offsets):
    """
    TODO
    """
    for n,nl_offset in enumerate(newline_offsets):
        #print offset_start,offset_end,nl_offset
        if(offset_start <= nl_offset[0] and offset_end <= nl_offset[0]):
            return (n+1, newline_offsets[n-1][1], newline_offsets[n][0])
