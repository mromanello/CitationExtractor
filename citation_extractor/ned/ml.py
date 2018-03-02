"""Machine learning code related to the NED step."""

# -*- coding: utf-8 -*-
# author: Matteo Filipponi

from __future__ import print_function
import logging
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm, linear_model, cross_validation, preprocessing, model_selection

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
        LOGGER.info('Applying pairwise transformation')
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

    def select_best_C(self, X, y, groups, k=10, cache_size=10000):
        C_scores = []

        for C in 10. ** np.arange(-3, 3):
            gkf = model_selection.GroupKFold(n_splits=k)

            scores = []

            for train, test in gkf.split(X, y, groups):
                X_train, X_test, y_train, y_test, groups_train, groups_test = X[train], X[test], y[train], y[test], groups[train], groups[test]

                # Fit the model
                classifier = svm.SVC(kernel='linear', C=C, cache_size=cache_size)
                Xp, yp = self._pairwise_transformation(X_train, y_train, groups_train)
                Xp_norm = preprocessing.normalize(Xp)
                classifier.fit(Xp_norm, yp)

                score = 0

                # Predict for each group
                for i in set(groups_test):
                    group_idx = (groups == i)
                    X_group = X[group_idx]
                    y_group = y[group_idx]

                    # TODO

                    coef = classifier.coef_.ravel()
                    norm_coef = coef / np.linalg.norm(coef)
                    scores = np.dot(X, norm_coef.T).ravel().tolist()

                    # TODO: update score

                scores.append(score)

            # TODO: compute avg score
            avg_score = 0
            C_scores.append((C, avg_score))

        C_scores.sort(key=lambda tup: tup[1])
        print(C_scores)
        return C_scores[0][0]

    def fit(self, X, y, groups):
        """Train the SVMRank model.

        :param X: the feature vectors to be used to train the model
        :type X: list of dict
        :param y: the labels of the feature vectors (0 or 1)
        :type y: list of int
        :param groups: the labels of the groups
        :type groups: list of int
        """
        X = self._dv.fit_transform(X)
        y, groups = map(np.array, (y, groups))

        LOGGER.info('Fitting data [number of points: {}, number of groups: {}]'.format(X.shape[0], len(set(groups))))

        if self._classifier is None:
            C = self.select_best_C(X, y, groups, k=2)
            C = 100
            cache_size = 10000
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
