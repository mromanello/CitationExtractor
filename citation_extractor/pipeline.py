# -*- coding: utf-8 -*-
# author: Matteo Romanello, matteo.romanello@gmail.com

"""
Basic command line interface to run the reference extraction pipeline.

Usage:
    citedloci-pipeline init --config=<file>
    citedloci-pipeline do (preproc | ner | relex | ned | all) --config=<file> [--doc=<name> --overwrite]

Options:
    -h, --help              Show this message.
    -V, --version           Show version.
    -c, --config=<file>     The configuration file.
    -o, --overwrite         Whether existing subdirs within `working-dir` should be kept

Examples:
     citedloci-pipeline  do preproc --config=config/project.ini
"""  # noqa: E501

import codecs
import importlib
import json
import logging
import os
import re
import shutil
import sys
from operator import itemgetter

import numpy as np
from docopt import docopt

import langid
from citation_extractor import __version__ as VERSION
from citation_extractor.core import citation_extractor
from citation_extractor.io.converters import DocumentConverter
from citation_extractor.io.iob import (count_tokens, file_to_instances,
                                       filter_IOB, read_iob_files,
                                       write_iob_file)
from citation_extractor.ned.matchers import CitationMatcher
from citation_extractor.relex.rb import RBRelationExtractor
from citation_extractor.Utils.IO import init_logger
from citation_extractor.Utils.strmatching import StringUtils
from citation_extractor.Utils.sentencesplit import sentencebreaks_to_newlines
from citation_parser import CitationParser
from knowledge_base import KnowledgeBase

try:
    import configparser
except ImportError:
    import ConfigParser as configparser


if(sys.version_info < (3, 0)):
    from treetagger_python2 import TreeTagger
else:
    from treetagger import TreeTagger

global logger
logger = logging.getLogger(__name__)
# TODO custom exception: invalid configuration


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

    logger.debug("Entity mentions extracted from \"%s\": %s" % (text, output))

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
            train_instances += read_iob_files(directory,extension=".txt")
        logger.info("Training data: found %i directories containing %i  sentences and %i tokens"%(len(settings.DATA_DIRS),len(train_instances),count_tokens(train_instances)))

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


def split_sentences(filename, outfilename=None):
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
        intermediate_out_file = os.path.join(interm_dir,doc_id)
        iob_out_file = os.path.join(out_dir,doc_id)
        text = codecs.open(os.path.join(inp_dir,doc_id),'r','utf-8').read()
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
        write_iob_file(tokenised_text,iob_out_file)
        logger.info("Written IOB output to %s"%iob_out_file)
        no_sentences = len(text.split('\n'))
        no_tokens = count_tokens(tokenised_text)
    except Exception, e:
        logger.error("The pre-processing of document %s (lang=\'%s\') failed with error \"%s\""%(doc_id,lang,e))
    finally:
        return doc_id, lang, no_sentences, no_tokens


# TODO: finish implementation
def validate_configuration(configuration_parameters, task="all"):
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


def initialize(configuration):
    """
    Validate the configuration file + initialize and persist objects (extractor,
    matcher, etc.) + initialize the working directory with subfolders.
    """
    pass


def init_working_dir(path, overwrite=False):
    """Initializes the wowrking directory with required structure."""
    working_directories = {}
    subdirs = ["orig", "txt", "iob", "iob_ne", "json", "xmi"]

    if os.path.exists(path) and overwrite:
        shutil.rmtree(path)

    for subdir in subdirs:

        newdir = os.path.join(path, subdir)
        if overwrite or not os.path.exists(newdir):
            os.makedirs(newdir)
        working_directories[subdir] = newdir

    return working_directories


# TODO: move to the codebase
def find_input_documents(task, processing_dirs, doc_id):
    """Locates input documents based on the current task.

    :param str task: Description of parameter `task`.
    :param list processing_dirs: Description of parameter `processing_dirs`.
    :param str doc_id: Id of a single document to process (if None all
        documents in the input folder are processed).
    :return: A list of file names.
    :rtype: list

    """
    docs_to_process = []
    if task == 'all':
        extension = '.txt'
        input_dir = 'orig'
    elif task == 'preproc':
        extension = '.txt'
        input_dir = 'orig'
    elif task == 'ner':
        extension = ".txt"
        input_dir = "iob"
    elif task == 'relex' or task == 'ned':
        extension = '.json'
        input_dir = 'json'

    if doc_id is None:
        docs_to_process = [
            file
            for file in os.listdir(processing_dirs[input_dir])
            if extension in file
        ]
    else:
        docs_to_process.append(doc_id)

    logger.info(
        "There are {} docs to process".format(len(docs_to_process))
    )
    logger.info(
        'Following documents will be processed: {}'.format(
            docs_to_process
        )
    )
    return docs_to_process


def do_ner(doc_id, inp_dir, interm_dir, out_dir, extractor):

    try:
        data = file_to_instances(os.path.join(inp_dir, doc_id))
        # store pos tags in a separate list of lists
        postags = [
            [
                ("z_POS", token[1])
                for token in instance
            ]
            for instance in data
            if len(instance) > 0
        ]

        # store tokens in a separate list of lists
        instances = [
            [
                token[0]
                for token in instance
            ]
            for instance in data
            if len(instance) > 0
        ]

        # extract entities from the input document
        result = extractor.extract(instances, postags)
        output = [
            [
                (
                    res[n]["token"].decode('utf-8'),
                    postags[i][n][1],
                    res[n]["label"]
                )
                for n, d_res in enumerate(res)
            ]
            for i, res in enumerate(result)
        ]

        # first write the IOB
        out_fname = os.path.join(interm_dir, doc_id)
        write_iob_file(output, out_fname)
        logger.info('Output successfully written to file'.format({out_fname}))

        # then convert to JSON
        dc = DocumentConverter()
        dc.load(iob_file_path=os.path.join(interm_dir, doc_id))
        dc.to_json(output_dir=out_dir)
        return (doc_id, True)
    except Exception, e:
        logger.error(
            'The NER of document {} failed with error {}'.format(doc_id, e)
        )
        return (doc_id, False)
    finally:
        logger.info('Finished processing document {}'.format(doc_id))
    return


def do_relex(doc_id, input_dir, output_dir, prefix=None):

    inp_file_path = os.path.join(input_dir, doc_id)
    if prefix is None:
        out_file_name = doc_id
    else:
        # this is horrible and hard coded
        out_file_name = doc_id.replace('stage1_', prefix)
    outp_file_path = os.path.join(output_dir, out_file_name)

    # read input file
    with codecs.open(inp_file_path, 'r', 'utf-8') as inpfile:
        doc = json.load(inpfile)

    assert doc is not None
    logger.info('Document {} contains {} entities.'.format(
        doc_id, len(doc['entities'])
    ))

    rel_extractor = RBRelationExtractor()
    doc['relations'] = rel_extractor.extract(doc)

    # debug
    """
    for rel_id in doc['relations']:
        arg1, arg2 = doc['relations'][rel_id]
        print("{} {}".format(
            doc['entities'][arg1]["surface"].encode('utf-8'),
            doc['entities'][arg2]["surface"].encode('utf-8')
        ))
    """

    # write output
    with codecs.open(outp_file_path, 'w', 'utf-8') as outpfile:
        json.dump(doc, outpfile)
        logger.info(
            "Document {} ({} entities, {} relations) written to {}".format(
                doc_id,
                len(doc["entities"]),
                len(doc["relations"]),
                outp_file_path
            )
        )


def do_ned(doc_id, input_dir, output_dir, matcher):

    inp_file_path = os.path.join(input_dir, doc_id)
    outp_file_path = os.path.join(output_dir, doc_id)

    # initialize the citation parser
    parser = CitationParser()
    cleaning_regex = r'([\.,\(\);]+)$'

    # read input file
    with codecs.open(inp_file_path, 'r', 'utf-8') as inpfile:
        doc = json.load(inpfile)

    relations = doc['relations']
    entities = doc['entities']

    # let's create an inverted index entity_id => relation_id
    entity2relations = {}

    for relation_id in relations:

        for e_id in relations[relation_id]:

            if e_id not in entity2relations:
                entity2relations[e_id] = relation_id

    # iterate through all entities
    for entity_id in doc['entities']:
        if int(entity_id) not in entity2relations:
            continue

        entity = doc['entities'][entity_id]
        e_type = str(entity["entity_type"])
        surface = entity["surface"]

        if e_type == 'REFSCOPE':
            # TODO: call the citation parser and normalize
            cleaned_scope = re.sub(
                cleaning_regex,
                "",
                surface
            ).replace(u"–", u"-")

            try:
                parsed_scope = parser.parse(cleaned_scope)
                norm_scope = CitationParser.scope2urn('', parsed_scope)[0]
                entity['norm_scope'] = norm_scope.replace(':', '')

                logger.info("{} => {}".format(
                    surface.encode('utf-8'),
                    entity['norm_scope'].encode('utf-8')
                ))
            except Exception:
                logger.warning("Unable to parse {} {}".format(
                    doc_id, surface.encode('utf-8')
                ))

        else:
            rel_id = entity2relations[int(entity_id)]
            arg1, arg2 = relations[rel_id]
            scope = entities[str(arg2)]['surface']
            try:
                result = matcher.disambiguate(surface, e_type, scope)
                entity['urn'] = str(result.urn)

                # if matched work is not a NIL entity, we map the URN into a URI
                if result.urn != 'urn:cts:GreekLatinLit:NIL':
                    work_uri = matcher._kb.get_resource_by_urn(result.urn).subject
                    entity['work_uri'] = work_uri
            except Exception as e:
                logger.error("There was a problem disambiguating {}".format(surface))
                logger.error("({}) {}".format(doc_id, e))

    # write output
    with codecs.open(outp_file_path, 'w', 'utf-8') as outpfile:
        json.dump(doc, outpfile)
        logger.info(
            "Document {}: total entities {}; disambiguated: {}".format(
                doc_id,
                len(doc['entities']),
                len(
                    [
                        entity
                        for ent_id in entities
                        if ('work_uri' in entities[ent_id] or
                            'author_uri' in entities[ent_id])
                    ]
                )
            )
        )
        logger.info(
            "Document {} written to {}".format(
                doc_id,
                outp_file_path
            )
        )
    return


def main():

    args = docopt(__doc__, version=VERSION)

    global logger
    logger = init_logger(loglevel=logging.INFO)

    if args['init']:
        print('Not yet implemented, sorry.')
    elif args['do']:
        # load the configuration file
        config = configparser.ConfigParser()
        config.readfp(open(args['--config']))

        clear_working_dir = args["--overwrite"]
        doc_id = args["--doc"]

        working_dir = os.path.abspath(config.get('general', 'working_dir'))
        dirs = init_working_dir(working_dir, overwrite=clear_working_dir)
        logger.info('Current working directory: {}'.format(working_dir))

        try:
            cfg_storage = os.path.abspath(config.get('general', 'storage'))
        except Exception:
            cfg_storage = None

        if args['preproc']:
            cfg_split_sentences = config.get('preproc', 'split_sentences')
            cfg_abbrevations_file = config.get('preproc', 'abbreviation_list')
            cfg_treetagger_home = config.get('preproc', 'treetagger_home')

            docs_to_process = find_input_documents('preproc', dirs, doc_id)

            # initialize TreeTagger PoSTaggers
            try:
                pos_taggers = get_taggers(
                    treetagger_dir=cfg_treetagger_home,
                    abbrev_file=cfg_abbrevations_file
                )
            except Exception as e:
                raise e

            # read the list of abbreviations if provided in config file
            try:
                abbreviations = codecs.open(
                    cfg_abbrevations_file
                ).read().split('\n')
            except Exception as e:
                # no big deal: if abbraviations are not there we simply
                # won't use them
                print(e)
                abbreviations = None

            for doc_id in docs_to_process:
                preproc_document(
                    doc_id,
                    inp_dir=dirs['orig'],
                    interm_dir=dirs['txt'],
                    out_dir=dirs["iob"],
                    abbreviations=abbreviations,
                    taggers=pos_taggers,
                    split_sentences=cfg_split_sentences
                )
            return
        elif args['ner']:

            cfg_model_name = config.get('ner', 'model_name')
            cfg_settings_dir = config.get('ner', 'model_settings_dir')

            if cfg_settings_dir not in sys.path:
                sys.path.append(str(cfg_settings_dir))

            docs_to_process = find_input_documents('ner', dirs, doc_id)

            # use importlib to import the settings
            ner_settings = importlib.import_module(cfg_model_name)

            # check whether a pickle exists
            if cfg_storage:
                pkl_extractor_path = os.path.join(cfg_storage, 'extractor.pkl')

            # if exists load from memory
            if cfg_storage is not None and os.path.exists(pkl_extractor_path):
                extractor = citation_extractor.from_pickle(pkl_extractor_path)
                logger.info(
                    "Extractor loaded from pickle {}".format(
                        pkl_extractor_path
                    )
                )
            # otherwise initialize and train
            else:
                extractor = get_extractor(ner_settings)
                assert extractor is not None
                extractor.to_pickle(pkl_extractor_path)

            for doc_id in docs_to_process:
                do_ner(
                    doc_id,
                    inp_dir=dirs["iob"],
                    interm_dir=dirs["iob_ne"],
                    out_dir=dirs["json"],
                    extractor=extractor
                )
            return
        elif args['relex']:
            # if no --doc is passed at cli, then all documents in folder
            # are processed, otherwise only that document
            docs_to_process = find_input_documents('relex', dirs, doc_id)

            for doc_id in docs_to_process:
                do_relex(
                    doc_id,
                    input_dir=dirs['json'],
                    output_dir=dirs['json'],
                )
        elif args['ned']:

            cfg_kb = config.get('ned', 'kb_config')

            # check whether a pickle exists
            if cfg_storage:
                pkl_matcher_path = os.path.join(cfg_storage, 'matcher.pkl')

            # if exists load from memory
            if cfg_storage is not None and os.path.exists(pkl_matcher_path):
                matcher = CitationMatcher.from_pickle(pkl_matcher_path)
                logger.info(
                    "CitationMatcher loaded from pickle {}".format(
                        pkl_matcher_path
                    )
                )
            # otherwise initialize and train
            else:
                kb = KnowledgeBase(cfg_kb)
                logger.info("Pickled CitationMatcher not found.")
                matcher = CitationMatcher(
                    kb,
                    fuzzy_matching_entities=True,
                    fuzzy_matching_relations=False,
                    min_distance_entities=4,
                    max_distance_entities=7,
                    distance_relations=2
                )
                assert matcher is not None

                if cfg_storage:
                    matcher.to_pickle(pkl_matcher_path)

            docs_to_process = find_input_documents('ned', dirs, doc_id)

            for doc_id in docs_to_process:
                do_ned(
                    doc_id,
                    input_dir=dirs['json'],
                    output_dir=dirs['json'],
                    matcher=matcher
                )
