#
#   Code from the article: Cross Validation Module for Python
#
#   (c) 2006-2008 Michael G. Noll <http://www.michael-noll.com/>
#
#   Original article:
#   http://www.michael-noll.com/blog/2006/08/03/cross-validation-module-for-python/
#
import random
import unittest

class PartitionerError(Exception): pass
class TooManyPartitionsError(PartitionerError): pass

class Partitioner(object):
    """This class can be used to create k-fold, randomized partition sets."""
    
    def __init__(self, numPartitions=10, randomize=True):
        self._numPartitions = numPartitions
        self._randomize = randomize
    
    def partition(self, seq):
        """Splits the supplied sequence into (almost) equally-sized partitions. When it is not
        possible to create completely equally-sized partitions, the 'first' partitions are
        slightly extended so that all list elements fit into the partitions."""
        
        if self._numPartitions > len(seq):
            raise TooManyPartitionsError, "Number of desired partitions exceeds number of items to be partioned"
        
        partitionSizes = self._getPartitionSizes(seq)

        workList = list(seq)
        # randomize the list of items if needed
        if self._randomize:
            workList = random.sample(workList, len(workList))
        
        index = 0
        for size in partitionSizes:
            yield workList[index:index+size]
            index += size
        return

    def _getPartitionSizes(self, seq):
        """Returns a list of integer numbers, representing the partition sizes
        for cross validation. The algorithm tries to create equally large partitions;
        if needed, it will extend the first partitions so that all input elements fit
        into a partition
        
        For example:
        >>> p = Partitioner(3)
        >>> p._getPartitionSizes(range(8))
        [3, 3, 2]

        """
        total = len(seq)
        size = int(total / self._numPartitions)
        mod = total % self._numPartitions
        
        partitionSizes = []
        for i in xrange(self._numPartitions):
            if i < mod:
                partitionSizes.append(size + 1)
            else:
                partitionSizes.append(size)
        return partitionSizes

class PartitionTester(unittest.TestCase):
    
    def testPartitionSize(self):
        """Testing for correct partition sizes and number of partitions"""
        p = Partitioner(3)
        self.assertEqual(p._getPartitionSizes(xrange(8)), [3,3,2])
        p = Partitioner(10)
        self.assertEqual(p._getPartitionSizes(xrange(105)), [11, 11, 11, 11, 11, 10, 10, 10, 10, 10])

    def testNonrandomizedPartition(self):
        """Testing whether each list element is assigned to only one partition, and not more than once"""
        p = Partitioner(4, randomize=False)
        self.assertEqual([partition for partition in p.partition(xrange(1, 30))], \
                [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15], \
                 [16, 17, 18, 19, 20, 21, 22], [23, 24, 25, 26, 27, 28, 29]])

    def testRandomizedPartition(self):
        """(Basic) Testing whether randomization assigns each list element to a partition, and only once"""
        p = Partitioner(4, randomize=True)
        partitions = p.partition(xrange(14))
        
        sum = 0
        for partition in partitions:
            sum += reduce(lambda x,y: x+y, partition)
        self.assertEqual(sum, 91)
        
    def testTooManyPartitions(self):
        """Test whether an error is raised when too many partitions are requested"""
        p = Partitioner(10)
        # don't know why a simple call to assertRaises() does not work when
        # c._getPartitionSizes() is a generator; so we use a workaround
        exception_raised = False
        try:
            # we try to loop through the iterator returned by
            # partition(); this should fail at first try!
            for partition in p.partition(xrange(1)):
                pass
        except TooManyPartitionsError:
            exception_raised = True
        self.assertEqual(exception_raised, True)


__all__ = ['Partitioner', 'PartitionerError', 'TooManyPartitionsError']


if __name__ == "__main__":
    unittest.main()
