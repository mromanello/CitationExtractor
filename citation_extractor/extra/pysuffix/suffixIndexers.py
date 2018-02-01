import tools_karkkainen_sanders as tks
import array

class SuffixIndexer :
  def __init__(self, data) :
    self.reset()
    self.buildWord(data)

  def buildWord(self, lstWords):
    raise NotImplementedError  
      
  def getWordAt(self, idx):
    raise NotImplementedError                           
        
  def reset(self):
    self.word = None
    self.sortedSuffixes = None
    self.lcp = None
    
  def sortSuffixes(self) :
    if self.sortedSuffixes != None:
      return
    self.sortedSuffixes = tks.direct_kark_sort(self.word)

  def computeLCP(self) :
    if self.lcp != None:
      return
    self.sortSuffixes()
    self.lcp = tks.LCP(self.word, self.sortedSuffixes)   
          
  def _search(self, word):
    self.sortSuffixes()
    min_ = 0
    max_ = len(self.sortedSuffixes) - 1 
    len_word = len(word)
    it_word = range(len(word))
    mid = 0
    while 1:
      mid = (max_ + min_) / 2
      start = self.sortedSuffixes[mid]
      for i in it_word:
        c1 = self.word[start + i]
        c2 = word[i]
        if c1 > c2: 
          if mid == max_:
            return None
          max_ = mid
          break
        elif c1 < c2:
          if mid == min_:
            return None
          min_ = mid
          break
      else:
        return mid

  def _searchAll(self, word):
    self.computeLCP()
    len_word = len(word)
    idx = self._search(word)
    if idx == None: 
      return None, None
    lcp = self.lcp
    words = set()
    sup = inf = idx
    while 1: 
      inf -= 1 
      if self.lcp[inf] < len_word:
        break
    inf += 1
    while 1: 
      if self.lcp[sup] < len_word:
        break
      sup += 1 
    return (inf, sup)

  def searchOneWord(self, word):
    idx = self._search(word)
    if idx == None:
      return None
    pos = self.sortedSuffixes[idx]
    return self.getWordAt(pos)

  def searchAllWords(self, word):
    inf, sup = self._searchAll(word)
    if inf == None: 
      return []
#    result = [] 
    result = []
    for idx in xrange(inf, sup+1):
      pos = self.sortedSuffixes[idx]
#      result.append(self.getWordAt(pos))
      result.append(self.getWordAt(pos))
    return list(set(result))
        
  def searchOneWordAndPos(self, word):
    idx = self._search(word)
    if idx == None:
      return None
    pos = self.sortedSuffixes[idx]
    return self.getWordAt(pos), self.getPosition(pos)
        
  def searchAllWordsAndPos(self, word):
    inf, sup = self._searchAll(word)
    if inf == None: 
      return []
    result = [] 
    for idx in xrange(inf, sup+1):
      pos = self.sortedSuffixes[idx]
      result.append((self.getWordAt(pos), self.getPosition(pos)))
    return result
    
    
class ListIndexer(SuffixIndexer):
  def buildWord(self, lstWords):
    if self.word != None:
      return
    self.array_str = lstWords
    charFrontier = chr(2)
    self.word = charFrontier.join(self.array_str)
  
    self.indexes = array.array('i', [-1]*len(self.word))
    self.wordStarts = array.array('i', [0]*len(self.array_str))
    idx_w = k = 0
    for w in self.array_str:
      self.wordStarts[idx_w] = k
      for _ in w :
        self.indexes[k] = idx_w
        k += 1
      idx_w += 1
      k += 1

  def getWordAt(self, pos):
#    return pos
    return self.indexes[pos]
#    return self.array_str[self.indexes[pos]]         
    
  def getPosition(self, pos):
    return pos - self.wordStarts[self.indexes[pos]]

class DictValuesIndexer(SuffixIndexer):
  def buildWord(self, dictWords):
    if self.word != None:
      return
#    self.array_str = lstWords
    charFrontier = chr(2)
    self.word = charFrontier.join(dictWords.itervalues())

    self.indexes = {}
    self.wordStarts = {}
    idx_w = i = 0
    for k, v in dictWords.iteritems():
      self.wordStarts[k] = i
      for _ in v :
        self.indexes[i] = k
        i += 1
      i += 1

  def getWordAt(self, pos):
#    return pos
    return self.indexes[pos]
      
  def getPosition(self, pos):
    return pos - self.wordStarts[self.indexes[pos]]
    
if __name__ == '__main__':                             
  data = [
    'azerty',
    'ayerty',
    'axxxty',
    'azeyyy',
  ]
  
  m = ListIndexer(data)

  s = 'rty'
  print data
  print 'sow', s, m.searchOneWord(s)
  print 'saw', s, m.searchAllWords(s)
  print 'sowap', s, m.searchOneWordAndPos(s)
  print 'sawap', s, m.searchAllWordsAndPos(s)

  data = {
    'a':'azerty',
    'b':'ayerty',
    'c':'axxxty',
    'd':'azeyyy',
  }
  
  m = DictValuesIndexer(data)

#  s = 'y'
  print data
  print 'sow', s, m.searchOneWord(s)
  print 'saw', s, m.searchAllWords(s)
  print 'sowap', s, m.searchOneWordAndPos(s)
  print 'sawap', s, m.searchAllWordsAndPos(s)
