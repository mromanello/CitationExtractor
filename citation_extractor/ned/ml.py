"""Machine learning code related to the NED step."""

# -*- coding: utf-8 -*-


from __future__ import print_function
import logging
import random
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm, linear_model, cross_validation, preprocessing

LOGGER = logging.getLogger(__name__)


# TODO: how to deal with parallel computing?
# TODO: how to deal/apply with (optional?) refinement step?


class SVMRank(object):
    def __init__(self):
        LOGGER.info('Initializing SVM Rank')
        self._classifier = None
        self._dv = DictVectorizer(sparse=False)

    def _pairwise_transformation(self, X, y, groups, nb_groups):
        LOGGER.info('Applying pairwise transformation')
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
        """
        :param X: A list of dicts
        :param y: A list of integers (0 or 1)
        :param groups: A list of integers
        :return: None
        """
        X = self._dv.fit_transform(X)
        y, groups = map(np.array, (y, groups))
        nb_groups = len(set(groups))
        LOGGER.info('Fitting data [number of points: {}, number of groups: {}]'.format(len(X), nb_groups))

        # Aapply pairwise transform
        Xp, yp = self._pairwise_transformation(X, y, groups, nb_groups)
        Xp_norm = preprocessing.normalize(Xp)
        print(Xp_norm)
        print(yp)

        # TODO: (optional?) compute best C parameter (k-folded)
        C = 100

        # Fit linear SVM
        LOGGER.info('Fitting classifier')
        self._classifier = svm.SVC(kernel='linear', C=C, cache_size=10000)
        self._classifier.fit(Xp_norm, yp)

    def predict(self, X):
        """

        :param X: A list of dicts
        :return: A tuple. (list of sorted indexes, list of sorted scores)
        """
        if not self._classifier:
            LOGGER.error('The classifier is not initialized. Method fit() should invoked first')
        X = self._dv.fit_transform(X)
        LOGGER.info('Applying prediction to matrix {}'.format(X.shape))
        coef = self._classifier.coef_.ravel()
        norm_coef = coef / np.linalg.norm(coef)
        scores = np.dot(X, norm_coef.T).ravel().tolist()
        sorted_columns = np.argsort(scores)[::-1].tolist()
        sorted_scores = map(lambda i: scores[i], sorted_columns)
        return sorted_columns, sorted_scores


def dict_feat_name_to_index(vect):
    d = {}
    feat_names = vect.get_feature_names()
    for i in range(len(feat_names)):
        d[feat_names[i]] = i
    return d


def main():
    logging.basicConfig(level=logging.DEBUG)

    lowb = 0
    upperb = 1
    shift = 1
    X = [
        dict(x=random.uniform(lowb, upperb), y=random.uniform(lowb, upperb)),
        dict(x=random.uniform(lowb, upperb), y=random.uniform(lowb, upperb)),
        dict(x=random.uniform(lowb, upperb) + shift, y=random.uniform(lowb, upperb) + shift),  # true one
        dict(x=random.uniform(lowb, upperb), y=random.uniform(lowb, upperb)),
        dict(x=random.uniform(lowb, upperb), y=random.uniform(lowb, upperb)),
        dict(x=random.uniform(lowb, upperb) + shift, y=random.uniform(lowb, upperb) + shift)  # true one
    ]
    print(X)
    y = [
        0,
        0,
        1,
        0,
        0,
        1
    ]
    print(y)
    groups = [
        0,
        0,
        0,
        1,
        1,
        1
    ]
    print(groups)

    ranker = SVMRank()
    ranker.fit(X=X, y=y, groups=groups)

    candidates = [
        dict(x=random.uniform(lowb, upperb), y=random.uniform(lowb, upperb)),
        dict(x=random.uniform(lowb, upperb) + shift, y=random.uniform(lowb, upperb) + shift),  # true one
        dict(x=random.uniform(lowb, upperb), y=random.uniform(lowb, upperb))
    ]

    ranked_candidates = ranker.predict(candidates)
    print(ranked_candidates)


if __name__ == '__main__':
    main()
