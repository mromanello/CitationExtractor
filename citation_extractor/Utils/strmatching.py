# -*- coding: utf-8 -*-
# author: Matteo Filipponi

"""

Module containing functions/classes for cleaning and matching of strings.

"""

import re
import sys
import unicodedata
from stop_words import safe_get_stop_words

global punct_codes, punct_codes_nodot, symbol_codes, numbers_codes

punct_codes = [i for i in xrange(sys.maxunicode) if unicodedata.category(unichr(i)).startswith('P')]
punct_codes_nodot = [i for i in xrange(sys.maxunicode) if unicodedata.category(unichr(i)).startswith('P') and i != 46]
symbol_codes = [i for i in xrange(sys.maxunicode) if unicodedata.category(unichr(i)).startswith('S')]
numbers_codes = [i for i in xrange(sys.maxunicode) if unicodedata.category(unichr(i)).startswith('N')]

global p_codes_to_space, p_codes_nodot_to_space, s_codes_to_space, n_codes_to_space

p_codes_to_space = dict.fromkeys(punct_codes, 32)
p_codes_nodot_to_space = dict.fromkeys(punct_codes_nodot, 32)
s_codes_to_space = dict.fromkeys(symbol_codes, 32)
n_codes_to_space = dict.fromkeys(numbers_codes, 32)

class StringUtils:

    @staticmethod
    def remove_symbols(s):
        return s.translate(s_codes_to_space)

    @staticmethod
    def remove_numbers(s):
        return s.translate(n_codes_to_space)

    @staticmethod
    def strip_accents(s):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

    @staticmethod
    def remove_apostrophed_words(s):
        s = re.sub(ur"[a-zA-Z]+'", '', s) # xx..'
        s = re.sub(ur"'[a-zA-Z]", '', s)  # 'x
        return s

    @staticmethod
    def remove_iiis(s):
        s = re.sub(ur" [i]+ ", ' ', s) # in the middle
        s = re.sub(ur"^[i]+ ", ' ', s) # at the beginning
        s = re.sub(ur" [i]+$", ' ', s) # at the end
        return s
        
    @staticmethod
    def strip_single_letters_without_dot(s):
        s = re.sub(ur" [a-zA-Z] ", ' ', s) # in the middle
        s = re.sub(ur"^[a-zA-Z] ", ' ', s) # at the beginning
        s = re.sub(ur" [a-zA-Z]$", ' ', s) # at the end
        return s

    @staticmethod
    def remove_punctuation(s, keep_dots=False):
        if keep_dots:
            return s.translate(p_codes_nodot_to_space)
        else:
            return s.translate(p_codes_to_space)

    @staticmethod
    def trim_spaces(s):
        return u' '.join(s.split())
    
    @staticmethod
    def remove_stop_words(string, language):
        tokens = string.split()
        clean_tokens = [token for token in tokens if token not in safe_get_stop_words(language)]
        return u' '.join(clean_tokens)

    @staticmethod
    def remove_words_shorter_than(name, k):
        filtered = filter(lambda w: len(w) > k, name.split())
        return u' '.join(filtered)

    @staticmethod
    def normalize(text, lang=None, keep_dots=False):
        if not text:
            return u'None'
        
        text = StringUtils.remove_symbols(text)
        text = StringUtils.remove_numbers(text)
        text = StringUtils.strip_accents(text)
        text = StringUtils.remove_apostrophed_words(text)
        text = StringUtils.remove_iiis(text)
        text = StringUtils.strip_single_letters_without_dot(text)
        text = StringUtils.remove_punctuation(text, keep_dots=keep_dots)
        text = StringUtils.trim_spaces(text)
        text = text.lower()

        if lang:
            text = StringUtils.remove_stop_words(text, lang)

        return text
