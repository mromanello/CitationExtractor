# -*- coding: utf-8 -*-
# author: Matteo Filipponi

"""Machine learning code related to the NED step."""

from __future__ import print_function
import logging
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm, linear_model, cross_validation, preprocessing

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
        # TODO: add feature to use sparse

    def _pairwise_transformation(self, X, y, groups):
        """Apply the pairwise transformation to groups of labeled vectors

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
        for i in range(nb_groups):
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

        # Apply pairwise transform
        Xp, yp = self._pairwise_transformation(X, y, groups)
        Xp_norm = preprocessing.normalize(Xp)

        if not self._classifier:
            # TODO: (optional?) compute best C parameter (k-folded)
            C = 100
            cache_size = 10000
            self._classifier = svm.SVC(
                kernel='linear',
                C=C,
                cache_size=cache_size
            )

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

        X = self._dv.fit_transform(X)

        LOGGER.info('Applying prediction to matrix {}'.format(X.shape))

        coef = self._classifier.coef_.ravel()
        norm_coef = coef / np.linalg.norm(coef)
        scores = np.dot(X, norm_coef.T).ravel().tolist()
        sorted_columns = np.argsort(scores)[::-1].tolist()
        sorted_scores = map(lambda i: scores[i], sorted_columns)
        return sorted_columns, sorted_scores
