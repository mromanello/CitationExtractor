#!/usr/bin/python
# -*- coding: utf-8 -*-
# author: Matteo Romanello, matteo.romanello@gmail.com

from __future__ import with_statement, print_function

import sys
import re
import os
import codecs
from citation_extractor.Utils import IO
from citation_extractor.pipeline import read_ann_file_new

INPUT_ENCODING = "UTF-8"
OUTPUT_ENCODING = "UTF-8"


class DocumentConverter(object):
    """DocumentConverter transforms legacy data into UIMA/XML format.

    TODO: explain assumptions about data format.

    Usage:
    >>> iob_file = ''
    >>> standoff_dir = ''
    >>> doc_conv = DocumentConverter(iob_file, standoff_dir)
    """
    def __init__(self, iob_file_path, standoff_dir):
        self.document_id = os.path.basename(iob_file_path)
        self.document_name = os.path.basename(iob_file_path).split('.')[0]
        self._iob_file_path = iob_file_path
        self._standoff_dir = standoff_dir

        self._iob_data = self._parse_iob(self._iob_file_path)
        self._standoff_data = self._get_start_end(self._iob_data)

    def _parse_iob(self, fn):
        """TODO."""
        docnum = 1
        sentences = []

        assert fn is not None

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
                    #if len(sentences) > 0:
                    #    output(fn, docnum, sentences)
                    sentences = []
                    docnum += 1
                    continue

                if (ln + 2 < len(lines) and
                    re.match(r'^\s*$', lines[ln+1]) and
                    re.match(r'^-+\s+O\s*$', lines[ln+2])):
                    # heuristic match for likely doc before current line
                    #if len(sentences) > 0:
                    #    output(fn, docnum, sentences)
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

    def _get_start_end(sentences):
        """TODO."""
        pass

    def write_xml(output_dir):
        pass
