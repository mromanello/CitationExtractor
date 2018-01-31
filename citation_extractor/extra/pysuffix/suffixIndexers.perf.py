import suffixIndexers as si
import time

l = open('suffixIndexers.py').readlines()
l = l * 100
word = '&'
runs = 10000

def naive_get_all_list(l, pattern) :
  return list((i for i in xrange(len(l)) if pattern in l[i]))


def test_naive(l, word, runs):
  t0 = time.time()
  for i in xrange(runs):
    naive_get_all_list(l, word)
  t1 = time.time()
  return t1 - t0

def test_indexer(l, word, runs):
  indexedList = si.ListIndexer(l)
  indexedList.computeLCP()
  t0 = time.time()
  for i in xrange(runs):
    indexedList.searchAllWords(word)
  t1 = time.time()
  return t1 - t0

print 'test naive'
print test_naive(l, word, 1)
print test_naive(l, word, 10)
print test_naive(l, word, 100)
print test_naive(l, word, 1000)
#print test_naive(l, word, 10000)  
  
print 'test indexer'
print test_indexer(l, word, 1)
print test_indexer(l, word, 10)
print test_indexer(l, word, 100)
print test_indexer(l, word, 1000)
#print test_indexer(l, word, 10000)