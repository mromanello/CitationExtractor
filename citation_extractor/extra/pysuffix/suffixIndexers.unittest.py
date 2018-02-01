#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
import tools_karkkainen_sanders as tks
import array
import suffixIndexers as si
import random
import re

def get_all_pos(w, pattern) :
  i = 0
  lt = len(w)
  res = []
  while i < lt :
    try :
      pos = w.index(pattern, i)
      res.append(pos)
      i = pos + 1
    except Exception, e :
      return res
  return res

def naive_get_all_list(l, pattern) :
  res = []
  for i in xrange(len(l)) :
    w = l[i]
    if is_pattern_in(pattern, w) :
      res.append(i)
  return res

def naive_get_all_dict(d, pattern) :
  res = []
  for i,w in d.iteritems() :
    if is_pattern_in(pattern, w) :
      res.append(i)
  return res

def naive_get_all_with_pos_list(l, pattern) :
  res = []
  for i in xrange(len(l)) :
    w = l[i]
    r = get_all_pos(w, pattern)
    for o in r :
      res.append((i,o))
  return res

def naive_get_all_with_pos_dict(d, pattern) :
  res = []
  for i,w in d.iteritems() :
    r = get_all_pos(w, pattern)
    for o in r :
      res.append((i,o))
  return res

class Test_list_indexer :
  def setUp(self) :
    self.l,self.pattern = self.getData()
    self.index = si.ListIndexer(self.l)

  def test_one_word(self) :
    res = self.index.searchOneWord(self.pattern)
    naive_res = naive_get_all_list(self.l, self.pattern)
    if res == None :
      self.assertTrue(naive_res == [])
    else :
      self.assertTrue(res in naive_res)

  def test_all_words(self) :
    res = self.index.searchAllWords(self.pattern).sort()
    naive_res = naive_get_all_list(self.l, self.pattern).sort()
    self.assertTrue(res == naive_res)

  def test_one_word_and_pos(self) :
    res = self.index.searchOneWordAndPos(self.pattern)
    naive_res = naive_get_all_with_pos_list(self.l, self.pattern)
    if res == None :
      self.assertTrue(naive_res == [])
    else :
      self.assertTrue(res in naive_res)

  def test_all_words_and_pos(self) :
    res = self.index.searchAllWordsAndPos(self.pattern).sort()
    naive_res = naive_get_all_with_pos_list(self.l, self.pattern).sort()
    self.assertTrue(res == naive_res)

class Test_dict_values_indexer :
  def setUp(self) :
    self.d, self.pattern = self.getData()
    self.index = si.DictValuesIndexer(self.d)

  def test_one_word(self) :
    res = self.index.searchOneWord(self.pattern)
    naive_res = naive_get_all_dict(self.d, self.pattern)
    if res == None :
      self.assertTrue(naive_res == [])
    else :
      self.assertTrue(res in naive_res)

  def test_all_words(self) :
    res = self.index.searchAllWords(self.pattern).sort()
    naive_res = naive_get_all_dict(self.d, self.pattern).sort()
    self.assertTrue(res == naive_res)

  def test_one_word_and_pos(self) :
    res = self.index.searchOneWordAndPos(self.pattern)
    naive_res = naive_get_all_with_pos_dict(self.d, self.pattern)
    if res == None :
      self.assertTrue(naive_res == [])
    else :
      self.assertTrue(res in naive_res)

  def test_all_words_and_pos(self) :
    res = self.index.searchAllWordsAndPos(self.pattern).sort()
    naive_res = naive_get_all_with_pos_dict(self.d, self.pattern).sort()
    self.assertTrue(res == naive_res)

class Test_list_a_b(Test_list_indexer, unittest.TestCase) :
  def getData(self) :
    l = ['a'*100 for i in xrange(100)]
    return l,'bbb'

class Test_list_a_a(Test_list_indexer, unittest.TestCase) :
  def getData(self) :
    l = ['a'*100 for i in xrange(100)]
    return l,'aaa'

#class Test_list_python(Test_list_indexer, unittest.TestCase) :
#  def getData(self) :
#    s = open('Python.htm','r').read()
#    s_unicode = unicode(s,'utf-8','replace')[20000:25000]
#    l = re.split('\s', s_unicode)
#    return l, 'ython'

class Test_dict_a_a(Test_dict_values_indexer, unittest.TestCase) :
  def getData(self) :
    d = dict([(i,'a'*10) for i in xrange(10)])
    return d,'aaa' 

class Test_dict_a_b(Test_dict_values_indexer, unittest.TestCase) :
  def getData(self) :
    d = dict([(i,'a'*10) for i in xrange(10)])
    return d,'bbb' 

#class Test_dict_python(Test_dict_values_indexer, unittest.TestCase) :
#  def getData(self) :
#    s = open('Python.htm','r').read()
#    s_unicode = unicode(s,'utf-8','replace')[20000:25000]
#    l = re.split('\s', s_unicode)
#    d = dict((i, l[i]) for i in xrange(len(l))) 
#    return d, 'ython'

if (__name__ == '__main__') :
  unittest.main() 
