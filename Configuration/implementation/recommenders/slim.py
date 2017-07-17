'''
Politecnico di Milano.
item_knn.py

Description: This file contains the definition and implementation of SLIM-based
             Recommenders, such as SLIM and Multi-Thread-SLIM.

Created by: Massimo Quadrana.
Modified by Fernando Pérez.

Last modified on 25/03/2017.
'''

import numpy as np
import scipy.sparse as sps
from .base import Recommender, check_matrix
from sklearn.linear_model import ElasticNet

import pdb

# Memory consumption problem solved by:
# https://stackoverflow.com/questions/38140693/python3-multiprocessing-consumes-extensively-much-ram-and-slows-down
def _partial_fit(triplet, X):
    j,l1_ratio,positive_only = triplet
    model = ElasticNet(alpha=1.0,
                       l1_ratio=l1_ratio,
                       positive=positive_only,
                       fit_intercept=False,
                       copy_X=False)

    # WARNING: make a copy of X to avoid race conditions on column j
    # TODO: We can probably come up with something better here.
    X_j = X.copy()
    # get the target column
    y = X_j[:, j].toarray()
    # set the j-th column of X to zero
    X_j.data[X_j.indptr[j]:X_j.indptr[j + 1]] = 0.0
    # fit one ElasticNet model per column
    model.fit(X_j, y)
    # self.model.coef_ contains the coefficient of the ElasticNet model
    # let's keep only the non-zero values
    nnz_idx = model.coef_ > 0.0
    values = model.coef_[nnz_idx]
    rows = np.arange(X.shape[1])[nnz_idx]
    cols = np.ones(nnz_idx.sum()) * j
    return values, rows, cols

class SLIM(Recommender):
    """
    Train a Sparse Linear Methods (SLIM) item similarity model.

    See:
        Efficient Top-N Recommendation by Linear Regression,
        M. Levy and K. Jack, LSRS workshop at RecSys 2013.

        SLIM: Sparse linear methods for top-n recommender systems,
        X. Ning and G. Karypis, ICDM 2011.
        http://glaros.dtc.umn.edu/gkhome/fetch/papers/SLIM2011icdm.pdf
    """

    def __init__(self,
                 l1_penalty=0.1,
                 l2_penalty=0.1,
                 positive_only=True):
        super(SLIM, self).__init__()
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.positive_only = positive_only
        self.l1_ratio = self.l1_penalty / (self.l1_penalty + self.l2_penalty)

    def short_str(self):
        return "SLIM"

    def __str__(self):
        return "SLIM(l1_penalty={},l2_penalty={},positive_only={})".format(
            self.l1_penalty, self.l2_penalty, self.positive_only
        )

    def fit(self, X):
        X = check_matrix(X, 'csr', dtype=np.float32)
        self.dataset = X
        X = check_matrix(X, 'csc', dtype=np.float32)
        n_items = X.shape[1]

        # initialize the ElasticNet model
        self.model = ElasticNet(alpha=1.0,
                                l1_ratio=self.l1_ratio,
                                positive=self.positive_only,
                                fit_intercept=False,
                                copy_X=False)

        # we'll store the W matrix into a sparse csr_matrix
        # let's initialize the vectors used by the sparse.csc_matrix constructor
        values, rows, cols = [], [], []

        # fit each item's factors sequentially (not in parallel)
        for j in range(n_items):
            # get the target column
            y = X[:, j].toarray()
            # set the j-th column of X to zero
            startptr = X.indptr[j]
            endptr = X.indptr[j + 1]
            bak = X.data[startptr: endptr].copy()
            X.data[startptr: endptr] = 0.0
            # fit one ElasticNet model per column
            self.model.fit(X, y)

            # self.model.coef_ contains the coefficient of the ElasticNet model
            # let's keep only the non-zero values
            nnz_idx = self.model.coef_ > 0.0
            values.extend(self.model.coef_[nnz_idx])
            rows.extend(np.arange(n_items)[nnz_idx])
            cols.extend(np.ones(nnz_idx.sum()) * j)

            # finally, replace the original values of the j-th column
            X.data[startptr:endptr] = bak

        # generate the sparse weight matrix
        self.W_sparse = sps.csc_matrix((values, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)

    def user_score(self, user_id):
        user_profile = self._get_user_ratings(user_id)
        scores = user_profile.dot(self.W_sparse).toarray().ravel()
        return scores

    def recommend(self, user_id, n=None, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self._get_user_ratings(user_id)
        scores = user_profile.dot(self.W_sparse).toarray().ravel()
        ranking = scores.argsort()[::-1]
        # rank items
        if exclude_seen:
            ranking = self._filter_seen(user_id, ranking)
        return ranking[:n]

    def predict(self, user_id, rated_indices):
        # compute the scores using the dot product
        user_profile = self._get_user_ratings(user_id)
        scores = user_profile.dot(self.W_sparse).toarray().ravel()
        return scores[rated_indices]

    def recommend_new_user(self, user_profile, n=None, exclude_seen=True):
        assert user_profile.shape[1] == self.W_sparse.shape[0], 'The number of items does not match!'
        # compute the scores using the dot product
        scores = user_profile.dot(self.W_sparse).toarray().ravel()
        ranking = scores.argsort()[::-1]
        # rank items
        if exclude_seen:
            seen = user_profile.indices
            unseen_mask = np.in1d(ranking, seen, assume_unique=True, invert=True)
            ranking = ranking[unseen_mask]
        return ranking[:n]

    def label(self, unlabeled_list, binary_ratings=False, n=None, exclude_seen=True, p_most=1, n_most=3, score_mode='user'):
        # Calculate the scores only one time.
        # users = []
        # items = []
        # for user_idx, item_idx in unlabeled_list:
        #     users.append(user_idx)
        #     items.append(item_idx)
        #
        # users = np.array(users,dtype=np.int32)
        # items = np.array(items,dtype=np.int32)
        unlabeled_list = check_matrix(unlabeled_list, 'lil', dtype=np.float32)
        users,items = unlabeled_list.nonzero()
        n_scores = len(users)
        uniq_users, user_to_idx = np.unique(users,return_inverse=True)
        if (score_mode == 'user'):
            filtered_scores = np.zeros(shape=n_scores,dtype=np.float32)
            curr_user = None
            i = 0
            for user,item in zip(users,items):
                if (curr_user != user):
                    curr_user = user
                    user_profile = self._get_user_ratings(curr_user)
                    scores = user_profile.dot(self.W_sparse).toarray().ravel()

                filtered_scores[i] = scores[item]
                i += 1

        elif (score_mode == 'batch'):
            pass
            # filtered_scores = []
            # uniq_users, user_to_idx = np.unique(users,return_inverse=True)
            # self.calculate_scores_batch(uniq_users)
            # filtered_scores = self.scores[users,items]

        elif (score_mode == 'matrix'):
            # compute the scores using the dot product
            profiles = self._get_user_ratings(uniq_users)
            scores = profiles.dot(self.W_sparse).toarray()
            filtered_scores = scores[user_to_idx,items]

        # At this point, we have all the predicted scores for the users inside
        # U'. Now we will filter the scores by keeping only the scores of the
        # items presented in U'. This will be an array where:
        # filtered_scores[i] = scores[users[i],items[i]]

        # Filtered the scores to have the n-most and p-most.
        # sorted_filtered_scores is sorted incrementally
        sorted_filtered_scores = filtered_scores.argsort()
        p_sorted_scores = sorted_filtered_scores[-p_most:]
        n_sorted_scores = sorted_filtered_scores[:n_most]

        if binary_ratings:
            scores = [(users[i], items[i], 1.0) for i in p_sorted_scores] + [(users[i], items[i], 0.0) for i in n_sorted_scores]
        else:
            scores = [(users[i], items[i], 5.0) for i in p_sorted_scores] + [(users[i], items[i], 1.0) for i in n_sorted_scores]

        # We sort the indices by user, then by item in order to make the
        # assignment to the LIL matrix faster.
        return sorted(scores, key=lambda triplet: (triplet[0],triplet[1]))


from multiprocessing import Pool
from functools import partial


class MultiThreadSLIM(SLIM):
    def __init__(self,
                 l1_penalty=0.1,
                 l2_penalty=0.1,
                 positive_only=True,
                 workers=4):
        super(MultiThreadSLIM, self).__init__(l1_penalty=l1_penalty,
                                              l2_penalty=l2_penalty,
                                              positive_only=positive_only)
        self.workers = workers

    def __str__(self):
        return "SLIM_mt(l1_penalty={},l2_penalty={},positive_only={},workers={})".format(
            self.l1_penalty, self.l2_penalty, self.positive_only, self.workers
        )

    def fit(self, X):
        X = check_matrix(X, 'csr', dtype=np.float32)
        self.dataset = X
        X = check_matrix(X, 'csc', dtype=np.float32)
        n_items = X.shape[1]
        # fit item's factors in parallel
        _pfit = partial(_partial_fit, X=X)
        pool = Pool(processes=self.workers)
        args_triplet = ((j,self.l1_ratio,self.positive_only) for j in np.arange(n_items))
        res = pool.map(_pfit, args_triplet)
        pool.close()
        pool.join()

        # res contains a vector of (values, rows, cols) tuples
        values, rows, cols = [], [], []
        for values_, rows_, cols_ in res:
            values.extend(values_)
            rows.extend(rows_)
            cols.extend(cols_)

        pool = None
        # generate the sparse weight matrix
        self.W_sparse = sps.csr_matrix((values, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)
