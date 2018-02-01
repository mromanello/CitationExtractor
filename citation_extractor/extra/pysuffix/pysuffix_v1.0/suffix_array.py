#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tools import *
from string import *

# su = suffixe, c-a-d un couple (offset,id_str)

class Suffix_array :

  def __init__(self) :
    self.path_array = []
    self.str_array = []
    self.suffix_array = []
    self.fusion = ''
    self.equiv = []

  def _get_dict(self) :
    return {
      "path_array" : self.path_array,
      "str_array" : self.str_array,
      "suffix_array" : self.suffix_array
    }

  def _print_dico(self) :
    str_utf8_array = []
    for elt in self.str_array:
      str = utf82unicode(elt)
      str_utf8_array.append('"'+unicode2utf8(str).replace("\n","\\n")+'"')
    res  = "{"
    res += '"path_array" :' + print_list_recursiv(self.path_array) + ','
    res += '"str_array" :' + print_list_recursiv(str_utf8_array) + ','
    res += '"suffix_array" : ' + print_list_recursiv(self.suffix_array)
    res += "}"
    return res
  
  def _add_str(self,str_unicode) :
    taille = len(self.fusion)
    self.str_array.append(str_unicode)
    if taille != 0 :
      self.fusion += unichr(2)
      self.equiv.append(taille+1)
    else :
      self.equiv.append(0)
    self.fusion += str_unicode

  def _add_path(self,path) :
    str = lire(path)
    str_unicode = utf82unicode(str)
    self.path_array.append(path)
    self._add_str(str_unicode)

  def decode(self,id_str) :
    sa = []
    debut = self.equiv[id_str]
    fin = len(self.str_array[id_str]) + debut
    for el in self.suffix_array :
      if el >= debut and el < fin :
        sa.append(el-debut)
    return sa   

  def offset_fusion2offset_normal(self,offset_fusion) :
    for id_str , start in enumerate(self.equiv) :
      end = start + len(self.str_array[id_str])
      if(offset_fusion >= start and offset_fusion < end) :
        return offset_fusion - start
    return -1 
 
  def offset_sa2id_str(self , offset_sa) : 
    for i , el in enumerate(self.equiv) :
      fin = el + len(self.str_array[i])
      if offset_sa < fin and offset_sa >= el :
        return i
    return -1

  def _verif_suffix_array(self) :
    i = 0
    verif = []
    while i < len(self.suffix_array) - 4 :
      offset1 = self.suffix_array[i]
      id_str1 = self.offset_sa2id_str(offset1)
      str1 = self.str_array[id_str1]
      taille1 = len(str1)
      offset_normal1  = self.offset_fusion2offset_normal(offset1)

      offset2 = self.suffix_array[i+1]
      id_str2 = self.offset_sa2id_str(offset2)
      str2 = self.str_array[id_str2]
      taille2 = len(str2)
      offset_normal2  = self.offset_fusion2offset_normal(offset2)

      if offset_normal2 > -1 and offset_normal1 > -1 :
        pstr1 = str1[offset_normal1:taille1]
        pstr2 = str1[offset_normal2:taille2]
        if pstr1 > pstr2 :
          pstr1 = unicode2utf8(pstr1[:20].replace("\n",'NL'))
          pstr2 = unicode2utf8(pstr2[:20].replace("\n",'NL'))
          print "********************"
          print "["+pstr1+"]"
          print "["+pstr2+"]"
          print "********************"
          verif.append((i,i+1))
      i += 1
    return verif

    
  def _write_su_n(self,n) :
    i = 0
    while i < len(self.suffix_array) - 3 :
      offset = self.suffix_array[i]
      id_str = self.offset_sa2id_str(offset)
      str = self.str_array[id_str]
      taille = len(str)
      offset_normal  = self.offset_fusion2offset_normal(offset)
      if offset_normal >= 0 :
        pstr = unicode2utf8(str[offset_normal:taille])
        print '%i %i %s' % (id_str,offset_normal,pstr)
      i += 1
  
  def karkkainen_sort(self) :
    n = len(self.fusion)
    s1 = self.fusion + unichr(1) + unichr(1) + unichr(1)
    b = lst_char(s1)
    s2 = [0]*len(s1)
    kark_sort(s1,s2,n,b)
    self.suffix_array = s2


if (__name__ == '__main__') :
  import sys
  import json
  from StringIO import StringIO
  import copy
  import time
  import string
  str = 'abc'
  str_unicode = utf82unicode(str)
  sa1 = Suffix_array()
  sa1._add_str(str_unicode)
  sa1._add_str(str_unicode)
  sa1.karkkainen_sort()
  sa1._verif_suffix_array()

