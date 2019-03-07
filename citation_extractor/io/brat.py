"""
Functions to deal with input/output of data in brat standoff format.
"""

from __future__ import print_function
import logging

logger = logging.getLogger(__name__)


def read_ann_file(fileid, ann_dir, suffix="-doc-1.ann"):
    """Reads annotations from a brat standoff file.

    :param fileid:
    :type fileid:
    :param ann_dir:
    :type ann_dir:
    :returns:
    :rtype: tuple
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
            relations[cols[0]] = {
                "ann_id": ann_id,
                "arguments": (arg1.split(":")[1], arg2.split(":")[1]),
                "relation_type":rel_type
            }

    return entities, relations, annotations


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


def load_brat_data(extractor, knowledge_base, postaggers, aph_ann_files, aph_titles, context_window=None):
    """
    Utility function to load a set of brat documents and prepare them
    in a format suitable for processing (typically when carrying out the evaluation or the training).

    :param citation_extractor: instance of `core.citation_extractor`
    :param knowledge_base: instance of `knowledge_base.KnowledgeBase`
    :param postaggers: TODO
    :param aph_ann_files: a tuple: [0] the base directory; [1] a list of file names
    :param aph_titles: `pandas.DataFrame` with column 'title'
    :param context_window: TODO
    :return: a `pandas.DataFrame` (columns: 'surface', 'surface_norm', 'surface_norm_dots',
            'scope', 'type', 'other_mentions', 'prev_mentions', 'urn', 'urn_clean','doc_id', 'doc_title',
            'doc_title_mentions', 'doc_title_norm', 'doc_text', 'sentence_start', 'sentence_end')

    """
    from citation_extractor.pipeline import extract_entity_mentions

    # define the columns of the resulting dataframe
    cols = ['surface', 'surface_norm', 'surface_norm_dots', 'scope', 'type', 'other_mentions', 'prev_mentions', 'urn', 'urn_clean',
            'doc_id', 'doc_title', 'doc_title_mentions', 'doc_title_norm', 'doc_text', 'sentence_start',
            'sentence_end', 'mentions_in_context']
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
                df_data.loc[mention_data_id, 'surface_norm_dots'] = StringUtils.normalize(mention_surface, keep_dots=True)
                df_data.loc[mention_data_id, 'scope'] = mention_scope
                df_data.loc[mention_data_id, 'type'] = mention_type
                # TODO: MatteoF controlla se si puo rimuovere
                df_data.loc[mention_data_id, 'prev_mentions'] = mention_prev_entities
                df_data.loc[mention_data_id, 'doc_id'] = file_id
                df_data.loc[mention_data_id, 'doc_title'] = doc_title
                df_data.loc[mention_data_id, 'doc_title_mentions'] = doc_title_extracted_mentions
                df_data.loc[mention_data_id, 'doc_title_norm'] = doc_title_norm
                df_data.loc[mention_data_id, 'doc_text'] = doc_text
                df_data.loc[mention_data_id, 'urn'] = mention_urn
                df_data.loc[mention_data_id, 'urn_clean'] = clean_urn

            # Add successfully parsed mentions of the doc to other_mentions
            # field of each mention of the doc
            for m_id in prev_entities:
                other_mentions = list(prev_entities)
                other_mentions.remove(m_id)
                other_mentions = map(
                    lambda mid: (
                        df_data.loc[mid, 'type'],
                        df_data.loc[mid, 'surface_norm_dots'],
                        df_data.loc[m_id, 'scope']
                    ),
                    other_mentions
                )
                df_data.loc[m_id, 'other_mentions'] = other_mentions

            # by now `prev_entities` contains all entities/relations, sorted
            if context_window is not None:
                for m_id in prev_entities:
                    context_size_left, context_size_right = context_window
                    context_start = df_data.loc[m_id, 'sentence_start'] - context_size_left
                    context_end = df_data.loc[m_id, 'sentence_end'] + context_size_right

                    # filter out the entities/mentions outside of the context
                    mentions_in_context = list(df_data[(df_data["doc_id"]==file_id) & \
                                                        (df_data["sentence_start"] >= context_start) & \
                                                        (df_data["sentence_end"]<= context_end)].index)

                    logger.debug("Entity %s; start sentence = %i; end sentence = %i; context = %i, %i; entities in context: %s" % (m_id
                                                                           , df_data.loc[m_id, 'sentence_start']
                                                                           , df_data.loc[m_id, 'sentence_end']
                                                                           , context_start
                                                                           , context_end
                                                                           , mentions_in_context))

                    # remove the entity in focus
                    mentions_in_context.remove(m_id)
                    mentions_in_context = map(
                        lambda mid: (
                            df_data.loc[mid, 'type'],
                            df_data.loc[mid, 'surface_norm_dots'],
                            df_data.loc[m_id, 'scope']
                        ),
                        mentions_in_context
                    )
                    df_data.loc[m_id, 'mentions_in_context'] = mentions_in_context

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
