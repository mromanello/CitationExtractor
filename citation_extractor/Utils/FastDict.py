# -*- coding: utf-8 -*-

from citation_extractor.extra.pysuffix.suffixIndexers import DictValuesIndexer


class LookupDictionary:
	"""
	>>> auth_dict = "http://cwkb.webfactional.com/cwkb/dict/authors/all/json" #doctest: +SKIP
	>>> raw_data = LookupDictionary.fetch_data(auth_dict) #doctest: +SKIP
	>>> import codecs
	>>> fname = dir="../data/works.csv"
	>>> file = codecs.open(fname,"r","utf-8")
	>>> raw_data = file.read()
	>>> file.close()
	>>> d = LookupDictionary(raw_data.encode('utf-8'))
	>>> d.lookup("Od.")
	{'/cwkb/works/2816#abbr': 'Od. ', '/cwkb/works/1600#abbr': 'Od.'}

	"""
	def __init__(self,raw_data):
		#self.raw_data = self.fetch_data(source)
		self.keys = self.process_source(raw_data)
		self.indexer = self.index(self.keys)

	@staticmethod
	def fetch_data(source):
		"""
		Fetch the data given the source URL passed as parameter.

		Args:
			source:
				a string being the source URL of the data to be fetched

		Returns:
			a string with the resource content.
		"""
		import urllib
		try:
			return urllib.urlopen(source).read()
		except Exception, e:
			raise e

	def process_source(self,raw_data):
		"""
		Turn a list of raw CSV data into a dictionary.
		"""
		import csv
		import os
		result = {}
		try:
			dict_csv = csv.reader(raw_data.split(os.linesep))
			for n,l in enumerate(dict_csv):
				# TODO here could go any manipulation of the data, e.g. lowercasing etc.
				result["%s#%s"%(l[0],l[1])]=l[2]
			return result
		except Exception as e:
			raise e

	def index(self,dictionary):
		"""
		Indexes a dictionary for quick lookups.
		"""
		return DictValuesIndexer(dictionary)

	def lookup(self,search_key):
		"""
		Searches within the internal dictionary.
		For each matching entry its key is returned.

		Args:
			search_key:
				...

		Returns:
			a dictionary:

		"""
		result = {}
		temp = self.indexer.searchAllWords(search_key)
		for res in temp:
			result[res] = self.keys[res]
		return result


if __name__ == '__main__':
	import doctest
	doctest.testmod(verbose=True)
