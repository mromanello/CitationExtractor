# -*- coding: utf-8 -*-
# author: Matteo Romanello, matteo.romanello@gmail.com

"""
Basic command line interface to run the reference extraction pipeline.

Usage:
    pipeline.py do (preproc | ner | relex | ned | all) --config=<file>

Options:
    -h, --help              Show this message.
    -V, --version           Show version.
    -c, --config=<file>     The configuration file.
"""

import os
import re
import sys
import logging
import codecs
import langid
if(sys.version_info < (3, 0)):
    from treetagger_python2 import TreeTagger
else:
    from treetagger import TreeTagger
from operator import itemgetter
import citation_extractor
from citation_extractor.Utils import IO
from citation_extractor.Utils.IO import read_ann_file, read_ann_file_new, init_logger, filter_IOB
from citation_extractor.Utils.sentencesplit import sentencebreaks_to_newlines # contained in brat tools
from citation_extractor.Utils.strmatching import StringUtils
import numpy as np

global logger
logger = logging.getLogger(__name__)

#TODO custom exception: invalid configuration


# Constants (remove?)
NIL_URN = 'urn:cts:GreekLatinLit:NIL'
LANGUAGES = ['en', 'es', 'de', 'fr', 'it']
AUTHOR_TYPE = 'AAUTHOR'
WORK_TYPE = 'AWORK'
REFAUWORK_TYPE = 'REFAUWORK'


def extract_entity_mentions(text, citation_extractor, postaggers, norm=False):
    if not text:
        return []

    try:
        # detect the language of the input string for starters
        lang = detect_language(text)

        # tokenise and do Part-of-Speech tagging
        postagged_string = postaggers[lang].tag(text)
    except Exception, e:
        logger.debug(u'Exception while tagging {} with lang={}'.format(text, lang))
        return []

    # convert to a list of lists; keep just token and PoS tag, discard lemma
    iob_data = [[token[:2] for token in sentence] for sentence in [postagged_string]]

    # put the PoS tags into a separate nested list
    postags = [[("z_POS", token[1]) for token in sentence] for sentence in iob_data if len(sentence) > 0]

    # put the tokens into a separate nested list
    tokens = [[token[0] for token in sentence] for sentence in iob_data if len(sentence) > 0]

    # invoke the citation extractor
    tagged_sents = citation_extractor.extract(tokens, postags)

    # convert the (verbose) output into an IOB structure
    output = [[(res[n]["token"].decode('utf-8'), postags[i][n][1], res[n]["label"])
               for n, d_res in enumerate(res)]
              for i, res in enumerate(tagged_sents)]

    logger.info(output)

    authors = map(lambda a: (AUTHOR_TYPE, a), filter_IOB(output, "AAUTHOR"))
    works = map(lambda w: (WORK_TYPE, w), filter_IOB(output, "AWORK"))
    mentions = authors + works

    if norm:
        mentions_norm = map(lambda (m_type, m_surface): (m_type, StringUtils.normalize(m_surface, lang=lang)), mentions)
        return mentions_norm

    return mentions


def recover_segmentation_errors(text, abbreviation_list, verbose=False):
    """

    Pretty straightforward heuristic here:
    if a line of text contains one token, which matches against a list of abbreviations
    assume that after this token there shouldn't be a sentence break; the same for
    the last token of a line consisting of more than one token.

    >> import  codecs
    >> abbrev_file = "data/abbreviations_all_in_one.txt"
    >> abbrev = codecs.open(abbrev_file).read().split('\n')
    >> text_file = 'data/txt/ocr_10.2307_40231021.txt'
    >> text = codecs.open(text_file,'r','utf-8').read()
    >> recover_segmentation_errors(text,abbrev,verbose=True)
    """
    def is_abbreviation(token,abbreviations):
        return token in abbreviations
    output = []
    text_lines = text.split('\n')
    if(verbose):
        print >> sys.stderr, "Input text has %i lines"%len(text_lines)
    for line in text_lines:
        tokens=line.split()
        if(len(tokens)==1):
            output+=tokens
            if(not is_abbreviation(tokens[0],abbreviation_list)):
                output.append('\n')
            else:
                if(verbose):
                    print >> sys.stderr,"%s is an abbreviation"%tokens[0]
        else:
            output+=tokens
            try:
                last_token = tokens[len(tokens)-1]
                if(not is_abbreviation(last_token,abbreviation_list)):
                    output.append('\n')
                else:
                    if(verbose):
                        print >> sys.stderr,"%s is an abbreviation"%last_token
            except Exception, e:
                pass
    output_text = " ".join(output)
    if(verbose):
        print >> sys.stderr, "Output text has %i lines"%len(output_text.split('\n'))
        print >> sys.stderr, "%i line were breaks recovered"%(len(text_lines)-len(output_text.split('\n')))
    return output_text


def get_taggers(treetagger_dir='/Applications/treetagger/cmd/', abbrev_file=None):
    """
    Initialises a set of treetaggers, one for each supported language (i.e. en, it, es, de, fr, nl).

    :param treetagger_dir: directory containing TreeTagger's executable. If the environment variable
                            `TREETAGGER_HOME` is not set, it takes the value of this parameter.
    :param abbrev_file: an abbreviation file to be passed to TreeTagger.
    :return: a dictionary where keys are language codes and values are `TreeTagger` instances.

    .. note::This function won't work (as it is) in Python 3, since the TreeTagger for py3 does not want
            the `encoding` parameter at initialisation time.
    """
    try:
        assert os.environ["TREETAGGER_HOME"] is not None
    except Exception, e:
        os.environ["TREETAGGER_HOME"] = treetagger_dir
    logger.info("Env variable $TREETAGGER_HOME == %s"%os.environ["TREETAGGER_HOME"])
    lang_codes = {
        'en':('english','utf-8'),
        'it':('italian','utf-8'),
        'es':('spanish','utf-8'),
        'de':('german','utf-8'),
        'fr':('french','utf-8'),
        #'la':('latin','latin-1'), #TODO: do it via CLTK
        'nl':('dutch','utf-8'),
        #'pt':('portuguese','utf-8') # for this to work one needs to add the language
                                     # to the dict _treetagger_languages in TreeTagger
    }
    taggers = {}
    for lang in lang_codes.keys():
        try:
            taggers[lang]=TreeTagger(language=lang_codes[lang][0]
                                    , encoding=lang_codes[lang][1]
                                    , abbreviation_list=abbrev_file)
            logger.info("Treetagger for %s successfully initialised"%lang)
        except Exception, e:
            logger.error("initialising Treetagger for language %s raised error: \"%s\""%(lang_codes[lang][0],e))
            raise e
    return taggers


def get_extractor(settings):
    """
    Instantiate, train and return a Citation_Extractor.
    """
    import sys
    import citation_extractor as citation_extractor_module
    from citation_extractor.core import citation_extractor
    from citation_extractor.Utils import IO
    ce = None
    try:
        logger.info("Using CitationExtractor v. %s"%citation_extractor_module.__version__)
        train_instances = []
        for directory in settings.DATA_DIRS:
            train_instances += IO.read_iob_files(directory,extension=".txt")
        logger.info("Training data: found %i directories containing %i  sentences and %i tokens"%(len(settings.DATA_DIRS),len(train_instances),IO.count_tokens(train_instances)))

        if(settings.CLASSIFIER is None):
            ce = citation_extractor(settings)
        else:
            ce = citation_extractor(settings, settings.CLASSIFIER)

    except Exception, e:
        print e
    finally:
        return ce


def detect_language(text, return_probability=False):
    """
    Detect language of a notice by using the module `langid`.

    : param text: the text whose language is to be detected
    :return: if `return_probability` == False, returns the language code (string);
                if `return_probability` == False, returns a tuple where tuple[0] is
                the language code and tuple[1] its probability.
    """
    try:
        language, classification_probability = langid.classify(text)
        logger.debug("Language detected => %s (%s)"%(language, classification_probability))
        if(return_probability):
            return language, classification_probability
        else:
            return language
    except Exception,e:
        print "lang detection raised error \"%s\""%str(e)


def compact_abbreviations(abbreviation_dir):
    """
    process several files with abbreviations
    chain them together and write them to a file
    """
    fname = "%s%s"%(abbreviation_dir,"kb_abbrevs.txt")
    import codecs
    f = codecs.open(fname,'w','utf-8')
    abbrvs = get_abbreviations_from_knowledge_base()
    f.write("\n".join(abbrvs))
    f.close()
    abbreviations = []
    files = [
        fname
        ,"/Applications/TextPro1.5.2/SentencePro/bin/dict/ita/abbreviations.txt"
        ,"/Applications/TextPro1.5.2/SentencePro/bin/dict/eng/abbreviations.txt"
        ,"/Applications/TextPro1.5.2/SentencePro/bin/dict/ita/no_split_abbreviations.txt"
        ,"/Applications/TextPro1.5.2/SentencePro/bin/dict/eng/no_split_abbreviations.txt"
    ]
    for fn in files:
        f = codecs.open(fn,'r','utf-8')
        print >> sys.stderr, "getting abbreviations from %s"%fn
        abbreviations = abbreviations + [line for line in f.readlines() if not line.startswith("#") and line !=""]
    abbreviations = sorted(list(set(abbreviations)))
    fname = "%s%s"%(abbreviation_dir,"abbreviations_all_in_one.txt")
    f = codecs.open(fname,'w','utf-8')
    f.write("".join(abbreviations))
    f.close()
    return fname,abbreviations


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
                logger.debug("Detected relation %s"%str(relations[rel_id]))

    return relations


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
        logger.info("Written %i relations to %s"%(len(relations),ann_file))
    except Exception, e:
        raise e
    return result


def clean_relations_annotation(fileid, ann_dir, entities):
    """
    overwrites relationships (type=scope) to an .ann file.
    """
    ann_file = "%s%s-doc-1.ann"%(ann_dir,fileid)
    keys = entities.keys()
    keys.sort(key=lambda k:(k[0], int(k[1:])))
    result = "\n".join([
        "%s\t%s %s %s\t%s" % (
            ent,
            entities[ent][0],
            entities[ent][2],
            entities[ent][3],
            entities[ent][1]
        )
        for ent in keys
    ])
    try:
        f = codecs.open(ann_file,'w','utf-8')
        f.write(result)
        f.close()
        logger.info("Cleaned relations annotations from %s"%ann_file)
    except Exception, e:
        raise e
    return result


def remove_all_annotations(fileid, ann_dir):
    """Remove all free-text annotations from a brat file."""
    ann_file = "%s%s-doc-1.ann" % (ann_dir, fileid)
    entities, relations, annotations = read_ann_file(fileid, ann_dir)

    entity_keys = entities.keys()
    entity_keys.sort(key=lambda k: (k[0], int(k[1:])))
    entities_string = "\n".join(
        [
            "%s\t%s %s %s\t%s" % (
                ent,
                entities[ent][0],
                entities[ent][2],
                entities[ent][3],
                entities[ent][1]
            )
            for ent in entity_keys
        ]
    )

    relation_keys = relations.keys()
    relation_keys.sort(key=lambda k: (k[0], int(k[1:])))
    relation_string = "\n".join(
        [
            "%s\tScope Arg1:%s Arg2:%s" % (
                rel,
                relations[rel][1].replace('Arg1:', ''),
                relations[rel][2].replace('Arg2:', '')
            )
            for rel in relation_keys
        ]
    )

    try:
        with codecs.open(ann_file,'w','utf-8') as f:
            f.write(entities_string)
            f.write("\n")
            f.write(relation_string)
        print >> sys.stderr, "Cleaned all relations annotations from %s"%ann_file
    except Exception, e:
        raise e
    return


def save_scope_annotations(fileid, ann_dir, annotations):
    """
    :param fileid: the file name (prefix added by brat is removed)
    :param ann_dir: the directory containing brat standoff annotations
    :param annotations: a list of tuples where: t[0] is the ID of the entity/relation the annotation is about;
                        t[1] is the label (it doesn't get written to the file); t[2] is the URN, i.e. the content
                        of the annotation. If t[2] is None the annotation is skipped
    :return: True if annotations were successfully saved to file, False otherwise.
    """
    try:
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
        return True
    except Exception as e:
        logger.error("Saving annotations to file %s%s failed with error: %s"%(ann_dir, fileid, e))
        return False


def tostandoff(iobfile,standoffdir,brat_script):
    """
    Converts the .iob file with NE annotation into standoff markup.
    """
    import sys
    import os
    try:
        cmd = "python %s -o %s %s"%(brat_script,standoffdir,iobfile)
        os.popen(cmd).readlines()
        logger.info("Document %s: .ann output written successfully."%iobfile)
    except Exception, e:
        raise e


def preproc_document(doc_id, inp_dir, interm_dir, out_dir, abbreviations, taggers, split_sentences=True):
    """
    :param doc_id: the input filename
    :param inp_dir: the input directory
    :param interm_dir: the directory where to store intermediate outputs
    :param out_dir: the directory where to store the PoS-tagged and tokenised text
    :param abbreviations:
    :param taggers: the dictionary returned by `get_taggers`
    :param split_sentences: (boolean) whether to slit text into sentences or not.
                            If `False`, text is split on newline characters `\n`.

    Returns:

    language, number of sentences, number of tokens

    """
    lang, no_sentences, no_tokens = np.nan,np.nan,np.nan
    try:
        intermediate_out_file = "%s%s"%(interm_dir,doc_id)
        iob_out_file = "%s%s"%(out_dir,doc_id)
        text = codecs.open("%s%s"%(inp_dir,doc_id),'r','utf-8').read()
        if(split_sentences):
            intermediate_text = sentencebreaks_to_newlines(text)
            text = recover_segmentation_errors(intermediate_text, abbreviations, verbose=False)
        else:
            logger.info("Document %s: skipping sentence splitting"%doc_id)
        sentences = text.split('\n')
        logger.info("Document \"%s\" has %i sentences"%(doc_id,len(sentences)))
        codecs.open(intermediate_out_file,'w','utf-8').write(text)
        logger.info("Written intermediate output to %s"%intermediate_out_file)
        lang = detect_language(text)
        logger.info("Language detected=\"%s\""%lang)
        tagged_sentences = taggers[lang].tag_sents(sentences)
        tokenised_text = [[token for token in line] for line in tagged_sentences]
        IO.write_iob_file(tokenised_text,iob_out_file)
        logger.info("Written IOB output to %s"%iob_out_file)
        no_sentences = len(text.split('\n'))
        no_tokens = IO.count_tokens(tokenised_text)
    except Exception, e:
        logger.error("The pre-processing of document %s (lang=\'%s\') failed with error \"%s\""%(doc_id,lang,e))
    finally:
        return doc_id, lang, no_sentences, no_tokens


def do_ner(doc_id, inp_dir, interm_dir, out_dir, extractor, so2iob_script):
    # TODO:
    # wrap with a try/except/finally
    # return doc_id and a boolean
    from citation_extractor.Utils import IO
    try:
        data = IO.file_to_instances("%s%s"%(inp_dir,doc_id))
        postags = [[("z_POS",token[1]) for token in instance] for instance in data if len(instance)>0]
        instances = [[token[0] for token in instance] for instance in data if len(instance)>0]
        result = extractor.extract(instances,postags)
        output = [[(res[n]["token"].decode('utf-8'), postags[i][n][1], res[n]["label"]) for n,d_res in enumerate(res)] for i,res in enumerate(result)]
        out_fname = "%s%s"%(interm_dir,doc_id)
        IO.write_iob_file(output,out_fname)
        logger.info("Output successfully written to file \"%s\""%out_fname)
        tostandoff(out_fname,out_dir,so2iob_script)
        return (doc_id,True)
    except Exception, e:
        logger.error("The NER of document %s failed with error \"%s\""%(doc_id,e))
        return (doc_id,False)
    finally:
        logger.info("Finished processing document \"%s\""%doc_id)
    return


def do_ned(
        doc_id,
        inp_dir,
        citation_matcher,
        clean_annotations=False
):
    """Perform named entity and relation disambiguation on a brat file."""
    try:
        if(clean_annotations):
            remove_all_annotations(doc_id, inp_dir)

        annotations = []
        pass

        return (doc_id, True, len(annotations))
    except Exception, e:
        logger.error("The NED of document %s failed with error \"%s\"" % (
            doc_id, e
        ))
        return (doc_id, False, None)
    finally:
        logger.info("Finished processing document \"%s\"" % doc_id)


def do_relex(doc_id, inp_dir, clean_relations=False):
    try:
        entities, relations, disambiguations = read_ann_file(doc_id,inp_dir)
        logger.info("%s: %i entities; %i relations; %i disambiguations"%(doc_id
                                                                        , len(entities)
                                                                        , len(relations)
                                                                        , len(disambiguations)))
        if(clean_relations):
            clean_relations_annotation(doc_id,inp_dir,entities)
        relations = extract_relationships(entities)
        for r in relations:
            logger.debug("%s %s -> %s"%(r,entities[relations[r][0]][1],entities[relations[r][1]][1]))
        if(len(relations)>0):
            save_scope_relationships(doc_id,inp_dir,relations,entities)
        return (doc_id,True,({"n_entities":len(entities),"n_relations":len(relations)}))
    except Exception, e:
        logger.error("The RelationExtraction from document %s failed with error \"%s\""%(doc_id,e))
        return (doc_id,False,{})
    finally:
        logger.info("Finished processing document \"%s\""%doc_id)


def validate_configuration(configuration_parameters, task="all"): #TODO finish
    """TODO"""
    def is_valid_configuration_ner(configuration_parameters):
        pass
    def is_valid_configuration_relex(configuration_parameters):
        pass
    def is_valid_configuration_ned(configuration_parameters):
        pass
    valid_tasks = ["all", "ner", "ned", "relex"]
    try:
        task in valid_tasks
    except Exception, e:
        raise e # custom exception
    if task == "all":
        assert is_valid_configuration_ner(configuration_parameters) and is_valid_configuration_relex(configuration_parameters) and is_valid_configuration_ned(configuration_parameters)
    elif task  == "ner":
        pass
    elif task == "relex":
        pass
    elif task == "ned":
        pass


def run_pipeline(configuration_file): #TODO: implement
    pass

if __name__ == "__main__":
    from docopt import docopt
    arguments = docopt(__doc__, version=citation_extractor.__version__)
    logger = init_logger()
    logger.info(arguments)
    # TODO: validate configuration file based on task at hand
