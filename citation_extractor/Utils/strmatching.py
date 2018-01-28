# -*- coding: utf-8 -*-
# author: Matteo Filipponi

"""

Module containing functions/classes for cleaning and matching of strings.

"""

import re
import sys
import unicodedata
import jellyfish
from stop_words import safe_get_stop_words

from citation_extractor.ned import PREPS

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


class DictUtils:
    @staticmethod
    def _dict_contains_match(self, dictionary):
        """Find if at least one boolean value of a dictionary is True

        :param dictionary: the target dictionary
        :type dict

        :return: True if at least one boolean value of a dictionary is True, False otherwise
        :rtype: bool
        """
        return any(filter(lambda v: type(v) == bool, dictionary.values()))


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
        s = re.sub(ur"[a-zA-Z]+'", '', s)  # xx..'
        s = re.sub(ur"'[a-zA-Z]", '', s)  # 'x
        return s

    @staticmethod
    def remove_iiis(s):
        s = re.sub(ur" [i]+ ", ' ', s)  # in the middle
        s = re.sub(ur"^[i]+ ", ' ', s)  # at the beginning
        s = re.sub(ur" [i]+$", ' ', s)  # at the end
        return s

    @staticmethod
    def strip_single_letters_without_dot(s):
        s = re.sub(ur" [a-zA-Z] ", ' ', s)  # in the middle
        s = re.sub(ur"^[a-zA-Z] ", ' ', s)  # at the beginning
        s = re.sub(ur" [a-zA-Z]$", ' ', s)  # at the end
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

    @staticmethod
    def split_names(names):
        """Split a list of names into a list of unique words.

        :param names: a list of names
        :type names: list of unicode

        :return: a list of unique words
        :rtype: list of unicode
        """
        splitted_names = set(u' '.join(names).split())
        return list(splitted_names)

    @staticmethod
    def remove_initial_word(word, name):
        """Remove a word from a name if the name starts with that word

        :param word: the word to be removed if present
        :type word: unicode
        :param name: a target name
        :type name: unicode

        :return: the name without the initial word if present, the name otherwise
        :rtype: unicode
        """
        pattern = u'^' + word + u' (.+)$'
        matched = re.match(pattern, name)
        if matched:
            return matched.group(1)
        return name

    @staticmethod
    def remove_words_shorter_than(name, k):
        """Remove words shorter than a specified length from a name

        :param name: the target name
        :type name: unicode
        :param k: the lenght threshold
        :type k: int

        :return: the name without the words shorter than k
        :rtype: unicode
        """
        filtered = filter(lambda w: len(w) > k, name.split())
        return u' '.join(filtered)

    @staticmethod
    def remove_preps(name):
        """Remove prepositions from a name

        :param name: the target name
        :type name: unicode

        :return: the name without prepositions
        :rtype: unicode
        """
        name_words = filter(lambda w: w not in PREPS, name.split())
        return u' '.join(name_words)

    @staticmethod
    def clean_surface(surface):
        """Clean the surface form of a mention

        :param surface: the surface form a mention
        :type: unicode

        :return: the surface form of a mention without prepositions and the words 'de', 'in' if present at the beginning
        :rtype: unicode
        """
        if len(surface.split()) > 1:
            surface = StringUtils.remove_initial_word(u'de', surface)
        if len(surface.split()) > 1:
            surface = StringUtils.remove_initial_word(u'in', surface)
        if len(surface.split()) > 2:
            surface = StringUtils.remove_preps(surface)
        return surface


class StringSimilarity:
    @staticmethod
    def levenshtein_distance_norm(s1, s2):
        return 1 - float(jellyfish.levenshtein_distance(s1, s2)) / max(len(s1) + len(s2), 1)

    @staticmethod
    def exact_match(s, names):
        return s in names

    @staticmethod
    def exact_match_swords(ss, names):
        matched = map(lambda s: s in names, ss)
        return all(matched)

    @staticmethod
    def qexact_match(s1, s2):
        return s1 == s2 or StringSimilarity.levenshtein_distance_norm(s1, s2) >= 0.9

    @staticmethod
    def exact_match_swords_any(ss, names):
        matched = map(lambda s: s in names, ss)
        return any(matched)

    @staticmethod
    def fuzzy_match(s, names):
        for n in names:
            if StringSimilarity.levenshtein_distance_norm(s, n) >= 0.9:
                return True
        return False

    @staticmethod
    def fuzzy_match_max(s, names):
        scores = [0.0]
        for n in names:
            scores.append(StringSimilarity.levenshtein_distance_norm(s, n))
        return max(scores)

    @staticmethod
    def fuzzy_match_swords(ss, names):
        matched = map(lambda s: StringSimilarity.fuzzy_match(s, names), ss)
        return all(matched)

    @staticmethod
    def fuzzy_match_swords_any(ss, names):
        matched = map(lambda s: StringSimilarity.fuzzy_match(s, names), ss)
        return any(matched)

    @staticmethod
    def _common_initial_letters(s1, s2):
        l = min(len(s1), len(s2))
        n = 0
        for i in range(l):
            if s1[i] == s2[i]:
                n += 1
        return float(n) / max(l, 1)

    @staticmethod
    def fuzzy_initial_match(s, names):
        for n in names:
            if StringSimilarity._common_initial_letters(s, n) >= 0.75:
                return True
        return False

    @staticmethod
    def fuzzy_initial_match_max(s, names):
        scores = [0.0]
        for n in names:
            scores.append(StringSimilarity._common_initial_letters(s, n))
        return max(scores)

    @staticmethod
    def fuzzy_initial_match_swords(ss, names):
        matched = map(lambda s: StringSimilarity.fuzzy_initial_match(s, names), ss)
        return all(matched)

    @staticmethod
    def _phonetic_similarity(s1, s2):
        sp1 = jellyfish.nysiis(s1)
        sp2 = jellyfish.nysiis(s2)
        return StringSimilarity.levenshtein_distance_norm(sp1, sp2)

    @staticmethod
    def fuzzy_phonetic_match(s, names):
        for n in names:
            if StringSimilarity._phonetic_similarity(s, n) >= 0.8:
                return True
        return False

    @staticmethod
    def fuzzy_phonetic_match_max(s, names):
        scores = [0.0]
        for n in names:
            scores.append(StringSimilarity._phonetic_similarity(s, n))
        return max(scores)

    @staticmethod
    def fuzzy_phonetic_match_swords(ss, names):
        matched = map(lambda s: StringSimilarity.fuzzy_phonetic_match(s, names), ss)
        return all(matched)

    @staticmethod
    def fuzzy_mrc(s, names):
        for n in names:
            if jellyfish.match_rating_comparison(s, n):
                return True
        return False

    @staticmethod
    def _is_exact_acronym(acronym, name):
        return acronym == u''.join(map(lambda w: w[0], name.split()))

    @staticmethod
    def acronym_match(s, names):
        for n in names:
            if StringSimilarity._is_exact_acronym(s, n):
                return True
        return False

    @staticmethod
    def abbreviation_match(s, names):
        for n in names:
            if n.startswith(s):
                return True
        return False

    @staticmethod
    def _is_sparse_abbreviation(abbr, name):
        pattern = u'^' + u'.*'.join(abbr) + u'.*$'
        return re.match(pattern, name) is not None

    @staticmethod
    def abbreviation_sparse_match(s, names):
        for n in names:
            if StringSimilarity._is_sparse_abbreviation(s, n):
                return True
        return False

    @staticmethod
    def _is_abbreviation_sequence(seq, name):
        pattern1 = u'^' + u'.* '.join(seq) + u'.*$'
        pattern2 = u'^.* ' + u'.* '.join(seq) + u'.*$'
        return re.match(pattern1, name) is not None or re.match(pattern2, name) is not None

    @staticmethod
    def abbreviation_sequence_match(seq, names):
        for n in names:
            if StringSimilarity._is_abbreviation_sequence(seq, n):
                return True
        return False
