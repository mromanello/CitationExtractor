#!/usr/bin/python
# -*- coding: utf-8 -*-
# author: Matteo Romanello, matteo.romanello@gmail.com

# Example of usage:
#	python align_IOB_to_ST_annotations.py /Users/rromanello/Documents/APh_Corpus/goldset/iob/75-06684.txt
#	--standoff-dir=/Users/rromanello/Documents/APh_Corpus/goldset/ann/
#	--output-dir=/Users/rromanello/Downloads/updated_iob/

from __future__ import with_statement

import sys
import re
import os
import codecs
from citation_extractor.Utils import IO
from citation_extractor.pipeline import read_ann_file

INPUT_ENCODING = "UTF-8"
OUTPUT_ENCODING = "UTF-8"

def quote(s):
	# function borrowed from https://github.com/nlplab/brat/blob/master/tools/conll02tostandoff.py 
    return s in ('"', )

def space(t1, t2, quote_count = None):
	# function borrowed from https://github.com/nlplab/brat/blob/master/tools/conll02tostandoff.py
    # Helper for reconstructing sentence text. Given the text of two
    # consecutive tokens, returns a heuristic estimate of whether a
    # space character should be placed between them.

    if re.match(r'^[\(]$', t1):
        return False
    if re.match(r'^[.,\)\?\!]$', t2):
        return False
    if quote(t1) and quote_count is not None and quote_count % 2 == 1:
        return False
    if quote(t2) and quote_count is not None and quote_count % 2 == 1:
        return False
    return True

def tagstr(start, end, ttype, idnum, text):
	# function borrowed from https://github.com/nlplab/brat/blob/master/tools/conll02tostandoff.py
    # sanity checks
    assert '\n' not in text, "ERROR: newline in entity '%s'" % (text)
    assert text == text.strip(), "ERROR: tagged span contains extra whitespace: '%s'" % (text)
    return "T%d\t%s %d %d\t%s" % (idnum, ttype, start, end, text)

def process(fn):
	# function adapted from https://github.com/nlplab/brat/blob/master/tools/conll02tostandoff.py
    docnum = 1
    sentences = []

    with codecs.open(fn, encoding=INPUT_ENCODING) as f:

        # store (token, BIO-tag, type) triples for sentence
        current = []

        lines = f.readlines()

        for ln, l in enumerate(lines):
            l = l.strip()

            if re.match(r'^\s*$', l):
                # blank lines separate sentences
                if len(current) > 0:
                    sentences.append(current)
                current = []
                continue
            elif (re.match(r'^===*\s+O\s*$', l) or
                  re.match(r'^-DOCSTART-', l)):
                # special character sequence separating documents
                if len(sentences) > 0:
                    output(fn, docnum, sentences)
                sentences = []
                docnum += 1
                continue

            if (ln + 2 < len(lines) and 
                re.match(r'^\s*$', lines[ln+1]) and
                re.match(r'^-+\s+O\s*$', lines[ln+2])):
                # heuristic match for likely doc before current line
                if len(sentences) > 0:
                    output(fn, docnum, sentences)
                sentences = []
                docnum += 1
                # go on to process current normally

            # Assume it's a normal line. The format for spanish is
            # is word and BIO tag separated by space, and for dutch
            # word, POS and BIO tag separated by space. Try both.
            m = re.match(r'^(\S+)\s(\S+)$', l)
            if not m:
                m = re.match(r'^(\S+)\s\S+\s(\S+)$', l)
            assert m, "Error parsing line %d: %s" % (ln+1, l)
            token, tag = m.groups()

            # parse tag
            m = re.match(r'^([BIO])((?:-[A-Za-z_]+)?)$', tag)
            assert m, "ERROR: failed to parse tag '%s' in %s" % (tag, fn)
            ttag, ttype = m.groups()
            if len(ttype) > 0 and ttype[0] == "-":
                ttype = ttype[1:]

            current.append((token, ttag, ttype))

        # process leftovers, if any
        if len(current) > 0:
            sentences.append(current)
    return sentences

def get_start_end(sentences):
	# function borrowed from https://github.com/nlplab/brat/blob/master/tools/conll02tostandoff.py
	# this function returns the start and end index for each token of an IOB file.
	# in order to do so, it uses the same logic of brat's algorithm (to convert from
	# IOB to StandOff format. The tokenisation of reference is the one contained in the
	# IOB file

	# output (list of lists): 
	#		[[(u'Hesiod', 0, 6), (u'Th.', 7, 10),...]

    offset, idnum = 0, 1

    doctext = ""
    sents = []
    for si, sentence in enumerate(sentences):

        prev_token = None
        prev_tag = "O"
        curr_start, curr_type = None, None
        quote_count = 0
        sent = []
        for token, ttag, ttype in sentence:
            
            
            if prev_token is not None and space(prev_token, token, quote_count):
                doctext = doctext + ' '
                offset += 1
                
            curr_start = offset

            if curr_type is None and ttag != "O":
                # a new tagged sequence begins here
                curr_start, curr_type = offset, ttype

            doctext = doctext + token
            offset += len(token)

            if quote(token):
                quote_count += 1

            prev_token = token
            prev_tag = ttag
            assert token == doctext[curr_start:offset]
            sent.append((token,curr_start,offset))
        sents.append(sent)

        if si+1 != len(sentences):
            doctext = doctext + '\n'        
            offset += 1
    return sents

def get_tag(tok_start,tok_end,entities):
	# this function search through a list of stand-off entities
	# given a start and end index for a token, the matching tag (formatted as IOB style)
	# is returned

    tag = None
    for entity in entities:
        entity_start = entity[2]
        entity_end = entity[3]
        if(tok_start == entity_start):
            tag = "B-%s"%entity[1]
        elif(tok_end == entity_end):
            tag = "I-%s"%entity[1]
        elif(tok_start >= entity_start and tok_end <= entity_end):
            tag = "I-%s"%entity[1]
    if(tag is None):
        return "O"
    else:
        return tag

def update(iob_startend,iob_instances,so_entities):
	out_sentences = []
	for si,sentence in enumerate(iob_startend):
	    out_sentence = []
	    for ti,token in enumerate(sentence):
	        tag = get_tag(token[1],token[2],so_entities)
	        old_tag = iob_instances[si][ti][2]
	        pos_tag = iob_instances[si][ti][1]
	        if(tag != old_tag):
	            # tell me if the tag changed
	            print >> sys.stderr,"UPDATE: sentence %i, token %i: \'%s\'[%i:%i] %s => %s "%(si,ti,token[0],token[1],token[2],old_tag,tag)
	        out_sentence.append((token[0], pos_tag, tag))
	    out_sentences.append(out_sentence)
	return out_sentences

def main():
	import argparse
	parser = argparse.ArgumentParser(description="")
	parser.add_argument("input", type=str, help="IOB input file")
	parser.add_argument("--standoff-dir", help="Stand-off directory",type=str,required=True)
	parser.add_argument("--output-dir", help="IOB output file",type=str,required=True)
	args = parser.parse_args()

	print >> sys.stderr, "IOB Input:", args.input
	print >> sys.stderr, "Stand-off input folder: ",args.standoff_dir
	print >> sys.stderr, "IOB output dir:", args.output_dir

	fname = os.path.split(args.input)[1].split(".")[0]

	# read the correspondant .ann file with stand-off annotation
	so_entities,so_relations,so_annotations = read_ann_file("%s.txt"%fname,args.standoff_dir)

	# extract for each token the start and end
	sentences = process(args.input)
	token_start_end = get_start_end(sentences)
	
	# read IOB from file
	iob_data = IO.file_to_instances(args.input)
	# make sure that data is consistent
	assert [len(sentence) for sentence in iob_data] == [len(sentence) for sentence in token_start_end]

	so_entities = [(so_entities[ent][1],so_entities[ent][0],int(so_entities[ent][2]),int(so_entities[ent][3])) for ent in so_entities.keys()]
	updated_iob_instances = update(token_start_end,iob_data,so_entities)
	try:
		destination = "%s%s.txt"%(args.output_dir,fname)
		IO.write_iob_file(updated_iob_instances,destination)
		print >> sys.stderr, "IOB output written to \'%s\'"%destination
	except Exception, e:
		print >> sys.stderr, "Writing ouput to \'%s\' failed with error \'%s\'"%(destination,e)
	
if __name__ == "__main__":
	main()
