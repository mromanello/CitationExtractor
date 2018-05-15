"""Machine learning code related to the NED step."""

# -*- coding: utf-8 -*-
# author: Matteo Filipponi

from __future__ import print_function

import logging
import time
import pdb

import numpy as np
from dask import delayed, compute

from citation_extractor.Utils.extra import avg
from sklearn import (model_selection, preprocessing, svm)
from sklearn.feature_extraction import DictVectorizer

LOGGER = logging.getLogger(__name__)


class LinearSVMRank(object):
    """Implementation of the SVMRank algorithm that uses a linear SVM."""

    def __init__(self, classifier=None):
        """
        Initialize an instance of LinearSVMRank.

        :param classifier: specify to use this SVM classifier.
        :type classifier: sklearn.svm.SVC
        """
        LOGGER.info('Initializing SVM Rank')
        self._classifier = classifier
        self._dv = DictVectorizer(sparse=False)
        # TODO: add feature to use sparse (MF)

    @staticmethod
    def pairwise_transformation(X, y, groups):
        """Apply the pairwise transformation to groups of labeled vectors.

        :param X: the matrix describing the vectors
        :type X: numpy.ndarray
        :param y: the labels of the vectors
        :type y: numpy.ndarray
        :param groups: the labels of the groups
        :type groups: numpy.ndarray

        :return: the pairwise-transformed vectors with their labels
        :rtype: numpy.ndarray, numpy.ndarray
        """
        LOGGER.debug('Applying pairwise transformation')
        nb_groups = len(set(groups))
        Xp, yp = [], []
        k = 0
        for i in set(groups):
            group_idx = (groups == i)
            X_group = X[group_idx]
            y_group = y[group_idx]

            y_group_true_idx = (y_group == 1)
            X_group_true = X_group[y_group_true_idx][0]
            X_group_false = X_group[~y_group_true_idx]

            for x in X_group_false:
                Xp.append(X_group_true - x)
                yp.append(1)

                # output balanced classes
                if yp[-1] != (-1) ** k:
                    yp[-1] *= -1
                    Xp[-1] *= -1
                k += 1

        Xp, yp = map(np.asanyarray, (Xp, yp))
        return Xp, yp

    def _pairwise_transformation(self, X, y, groups):
        """Apply the pairwise transformation to groups of labeled vectors.

        :param X: the matrix describing the vectors
        :type X: numpy.ndarray
        :param y: the labels of the vectors
        :type y: numpy.ndarray
        :param groups: the labels of the groups
        :type groups: numpy.ndarray

        :return: the pairwise-transformed vectors with their labels
        :rtype: numpy.ndarray, numpy.ndarray
        """
        LOGGER.debug('Applying pairwise transformation')
        nb_groups = len(set(groups))
        Xp, yp = [], []
        k = 0
        for i in set(groups):
            group_idx = (groups == i)
            X_group = X[group_idx]
            y_group = y[group_idx]

            y_group_true_idx = (y_group == 1)
            X_group_true = X_group[y_group_true_idx][0]
            X_group_false = X_group[~y_group_true_idx]

            for x in X_group_false:
                Xp.append(X_group_true - x)
                yp.append(1)

                # output balanced classes
                if yp[-1] != (-1) ** k:
                    yp[-1] *= -1
                    Xp[-1] *= -1
                k += 1

        Xp, yp = map(np.asanyarray, (Xp, yp))
        return Xp, yp

    def _select_best_C(self, X, y, groups, k=10, cache_size=10000):
        """ Use k-fold cross validation to select the best C parameter for SVM.

        :param X: the matrix describing the vectors
        :type X: numpy.ndarray
        :param y: the labels of the vectors
        :type y: numpy.ndarray
        :param groups: the labels of the groups
        :type groups: numpy.ndarray
        :param k: the number of folds to be used in cross validation
        :type k: int
        :param cache_size: the cache size used by SVM
        :type cache_size: int

        :return: The best C parameter for SVM
        :rtype: float
        """

        def _test_C_value(C, k, X, y, groups):
            gkf = model_selection.GroupKFold(n_splits=k)
            LOGGER.info("Testing C={} (k={})".format(C, k))
            tasks = [
                _do_split(C, train, test, X, y, groups)
                for train, test in gkf.split(X, y, groups)
            ]
            scores = compute(*tasks)
            LOGGER.info("Scores {}".format(scores))
            avg_score = avg(scores)
            LOGGER.info("Done testing C={}: avg score={}".format(C, avg_score))
            return (C, avg_score)

        @delayed
        def _do_split(C, split_train, split_test, X, y, groups):
            # Fit the model
            classifier = svm.SVC(
                kernel='linear',
                C=C,
                cache_size=100
            )
            X_train, X_test = X[split_train], X[split_test]
            y_train, y_test = y[split_train], y[split_test]
            groups_train, groups_test = groups[split_train], groups[split_test]
            Xp, yp = LinearSVMRank.pairwise_transformation(
                X_train,
                y_train,
                groups_train
            )
            Xp_norm = preprocessing.normalize(Xp)
            LOGGER.info("Fitting classifier... (C={})".format(C))
            start = time.clock()
            classifier.fit(Xp_norm, yp)
            end = time.clock()
            LOGGER.info("Done fitting, took {} secs".format(end - start))

            test_score = 0
            nb_groups = len(set(groups_test))
            nb_correct = 0

            # Predict for each group
            for i in set(groups_test):
                group_idx = (groups_test == i)
                X_group = X_test[group_idx]
                y_group = y_test[group_idx]

                coef = classifier.coef_.ravel()
                norm_coef = coef / np.linalg.norm(coef)
                rank_scores = np.dot(X_group, norm_coef.T).ravel().tolist()
                sorted_columns = np.argsort(rank_scores)[::-1].tolist()
                is_correct = y_group[sorted_columns[0]] == 1

                if is_correct:
                    nb_correct += 1

            test_score = float(nb_correct) / float(nb_groups)
            return test_score

        # adjust number of folds (k) based on number of groups (instances)
        total_nb_groups = len(set(groups))
        if total_nb_groups < k:
            LOGGER.warning(
                'k ({0}) is greater than the number of\
                groups ({1}). Using k = {1}'.format(k, total_nb_groups)
            )
            k = total_nb_groups

        LOGGER.info(
            'Selecting best C prameter using k-fold cross\
             validation (k={}, cache_size={})'.format(k, cache_size)
        )

        C_scores = [
            _test_C_value(C, k, X, y, groups)
            for C in 10. ** np.arange(-3, 3)
        ]
        C_scores.sort(key=lambda tup: tup[1], reverse=True)
        best_C = C_scores[0][0]
        best_C_score = C_scores[0][1]

        LOGGER.info('Selecting best C prameter using k-fold cross\
         validation - Results:\n  Scores: {}\n  Best C: {}, Score: {}'.format(
            C_scores, best_C, best_C_score
        ))

        return best_C

    def fit(self, X, y, groups, kfold_C_param=True, C=1, k=10, cache_size=100):
        """Train the SVMRank model.

        :param X: the feature vectors to be used to train the model
        :type X: list of dict
        :param y: the labels of the feature vectors (0 or 1)
        :type y: list of int
        :param groups: the labels of the groups
        :type groups: list of int
        :param kfold_C_param: enable k-fold cross validation to select parameter C (default is False)
        :type kfold_C_param: bool
        :param C: the C parameter to use in SVM, only used if kfold_C_param is False (default is 1)
        :type C: float
        :param k: the number of folds to be used in cross validation, only used if kfold_C_param is True (default is 10)
        :type k: int
        :param cache_size: specify the size of the kernel cache used by SVM (default is 100 MB)
        :type cache_size: int
        """
        X = self._dv.fit_transform(X)
        y, groups = map(np.array, (y, groups))

        LOGGER.info('Fitting data [number of points: {}, number of groups: {}]'.format(X.shape[0], len(set(groups))))

        if self._classifier is None:
            LOGGER.info("Find C is {}".format(kfold_C_param))
            if kfold_C_param:
                C = self._select_best_C(X, y, groups, k=k, cache_size=cache_size)
            self._classifier = svm.SVC(kernel='linear', C=C, cache_size=cache_size)

        # Apply pairwise transform
        Xp, yp = self._pairwise_transformation(X, y, groups)
        Xp_norm = preprocessing.normalize(Xp)

        # Fit linear SVM
        LOGGER.info('Fitting classifier')
        self._classifier.fit(Xp_norm, yp)

    def predict(self, X):
        """Rank a group of feature vectors.

        :param X: the feature vectors to be ranked
        :type X: list of dict

        :return: A tuple containing (list of sorted indexes of X, list of sorted scores)
        :rtype: list of int, list of float
        """
        if not self._classifier:
            LOGGER.error(
                'The classifier is not initialized. Method fit() should be\
                 invoked first'
            )
            return None

        X = self._dv.transform(X)

        LOGGER.info('Applying prediction to matrix {}'.format(X.shape))

        coef = self._classifier.coef_.ravel()
        norm_coef = coef / np.linalg.norm(coef)
        scores = np.dot(X, norm_coef.T).ravel().tolist()
        sorted_columns = np.argsort(scores)[::-1].tolist()
        sorted_scores = map(lambda i: scores[i], sorted_columns)
        return sorted_columns, sorted_scores
