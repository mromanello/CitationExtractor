#
#   Code from the article: Cross Validation Module for Python
#
#   (c) 2006-2008 Michael G. Noll <http://www.michael-noll.com/>
#
#   Original article:
#   http://www.michael-noll.com/blog/2006/08/03/cross-validation-module-for-python/
#
import unittest
from itertools import izip

from partitioner import *

class CrossValidationDataConstructorError(Exception): pass
class DuplicateItemError(CrossValidationDataConstructorError): pass

class CrossValidationDataConstructor(object):
    """This class is used to construct data sets for cross validation. It accepts two input lists,
    positiveList and negativeList, splits them into a user-specifiable number of partitions and
    pairs the good and bad partitions item-wise, i.e. (positivePartitions[i], negativePartitions[i]),
    which it returns as a list. This result list of good/bad tuples can be used for various
    statistical evaluations.
    """
    
    def __init__(self, positiveList=None, negativeList=None, numPartitions=10, randomize=True, partitioner=None):

        self._positiveList = positiveList or []
        self._negativeList = negativeList or []
        self.numPartitions = numPartitions
        self.randomize = randomize
        self._partitioner = partitioner or \
                Partitioner(numPartitions=self.numPartitions, randomize=self.randomize)

    def getDataSets(self):
        """Generate #{numPartitions} data sets for cross validation purposes.

        Returns a list of length numPartitions. Each list element is a tuple of
        training and testing data sets, which in turn are tuples consisting of
        a positive sample partition and a negative sample partition. The sample
        partitions are represented as lists.
        
        Example:
                |--- training ---|   |--- testing ----| 
            [ ( (posList, negList) , (posList, negList) ), ... ]

        """
        return self._foldPartitionPairs(self._getPartitionPairs())
    
    def _getPartitionPairs(self):
        """Combines the partitions of good and bad lists into a list (with
        length numPartitions) of tuples of the form (positivePartitions[i],
        negativePartitions[i]). This list of good/bad pairs can be used for
        training and testing classifiers.

        """
        positivePartitions = self._partitioner.partition(self._positiveList)
        negativePartitions = self._partitioner.partition(self._negativeList)
        return izip(positivePartitions, negativePartitions)

    def _foldPartitionPairs(self, partitionPairs):
        """For each partition type, i.e. positive and negative, the algorithm
        should merge n-1 partitions into one and designate it for training, and
        use the remaining partition for testing.
        
        """
        # join n-1 partition pairs to a single partition pair for training,
        # and use the remaining partition pair for testing
        crossValidationDataSets = []
        for i in range(0, self.numPartitions):
            index = 0
            # first list is positive, second list is negative
            training = ([], [])
            # first list is positive, second list is negative
            testing = ([], [])
            for (positivePartition, negativePartition) in self._getPartitionPairs():
                if index == i:
                    # testing partition pair
                    testing[0].extend(positivePartition)
                    testing[1].extend(negativePartition)
                else:
                    # training partition pair
                    training[0].extend(positivePartition)
                    training[1].extend(negativePartition)
                index += 1
            # add it to the cross validation data sets
            yield (training, testing)
        return
            
class CrossValidationDataConstructorTester(unittest.TestCase):
    """Tests whether folding of partition pairs into two tuples for training
    and testing works correctly. For each partition type, i.e. positive and
    negative, the algorithm should merge n-1 partitions into one and designate
    it for training, and use the remaining partition for testing.
    
    """
    
    def testEachItemOnlyOnce(self):
        """Tests whether an item is only element of a single partition, and only once therein."""
        numbers = 66
        positiveList = range(0, numbers/2)
        negativeList = range(numbers/2, numbers)
        numPartitions = 9
        
        c = CrossValidationDataConstructor(positiveList, negativeList, numPartitions=numPartitions)
        dataSets = c.getDataSets()
        for (trainingSet, testingSet) in dataSets:
            (positiveTraining, negativeTraining) = trainingSet
            (positiveTesting, negativeTesting) = testingSet

            # merge all partitions into one list for the test
            all = []
            all.extend(positiveTraining)
            all.extend(negativeTraining)
            all.extend(positiveTesting)
            all.extend(negativeTesting)
            
            foundNumbers = []
            for number in all:
                if number in foundNumbers:
                    raise DuplicateItemError, "Found a duplicate item in a data set."
                else:
                    foundNumbers.append(number)
    
    def testAllItemsIncluded(self):
        """Tests whether every item has been assigned to at least one partition."""
        numbers = 57
        positiveList = range(0, numbers/2)
        negativeList = range(numbers/2, numbers)
        numPartitions = 8
        
        c = CrossValidationDataConstructor(positiveList, negativeList, numPartitions=numPartitions)
        for (trainingSet, testingSet) in c.getDataSets():
            (positiveTraining, negativeTraining) = trainingSet
            (positiveTesting, negativeTesting) = testingSet

            # merge all partitions into one list for the test
            all = []
            all.extend(positiveTraining)
            all.extend(negativeTraining)
            all.extend(positiveTesting)
            all.extend(negativeTesting)
            
            foundNumbers = {}
            for number in all:
                foundNumbers[number] = 1
            
            for number in range(0, numbers):
                assert number in foundNumbers, "Found a missing number, which was not included in the generated data set."

    
    def testNumberOfPartitionPairs(self):
        """Tests whether as many partition pairs are returned \
        as number of requested partitions."""
        positiveList = range(1,10)
        negativeList = range(-9, 0)
        numPartitions = 3

        c = CrossValidationDataConstructor(positiveList, negativeList, \
                numPartitions=numPartitions)
        num_created_partitions = 0
        for training, testing in c._getPartitionPairs():
            num_created_partitions += 1
        self.assertEqual(num_created_partitions, numPartitions)


__all__ = ['CrossValidationDataConstructor', 'CrossValidationDataConstructorError', \
                   'DuplicateItemError']


if __name__ == "__main__":
    unittest.main()
