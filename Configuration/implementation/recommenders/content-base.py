'''
Politecnico di Milano.
content.py

Description: This file contains the definition of an abstract content-based
             recommender.

Created and modified by Fernando Pérez.

Last modified on 22/05/2017.
'''

import random as random

import numpy as np
import scipy.sparse as sps
from .base import Recommender, check_matrix
from .similarity import Cosine, Pearson, AdjustedCosine

import pdb

class ContentBasedRecommender(Recommender):
    """ Content-Based recommender"""

    def __init__(self, k=50, shrinkage=100, similarity='cosine', normalize=False, sparse_weights=True):
        super(ContentBasedRecommender, self).__init__()
        self.k = k
        self.shrinkage = shrinkage
        self.normalize = normalize
        self.dataset = None
        self.similarity_name = similarity
        self.sparse_weights = sparse_weights
        if similarity == 'cosine':
            self.distance = Cosine(shrinkage=self.shrinkage)
        elif similarity == 'pearson':
            self.distance = Pearson(shrinkage=self.shrinkage)
        elif similarity == 'adj-cosine':
            self.distance = AdjustedCosine(shrinkage=self.shrinkage)
        else:
            raise NotImplementedError('Distance {} not implemented'.format(similarity))

    def __str__(self):
        return "Content(similarity={},k={},shrinkage={},normalize={},sparse_weights={})".format(
            self.similarity_name, self.k, self.shrinkage, self.normalize, self.sparse_weights)

    def fit(self, X):
        '''
            Fits the model given the Item Content Matrix X.
            This computes the similarity based on the distance given in the
            initialization part. After that, it holds only the k-most scored
            items similarities.
        '''
        self.dataset = X
        item_weights = self.distance.compute(X)
        # for each column, keep only the top-k scored items
        idx_sorted = np.argsort(item_weights, axis=0)  # sort by column
        if not self.sparse_weights:
            self.W = item_weights.copy()
            # index of the items that don't belong to the top-k similar items of each column
            not_top_k = idx_sorted[:-self.k, :]
            # use numpy fancy indexing to zero-out the values in sim without using a for loop
            self.W[not_top_k, np.arange(item_weights.shape[1])] = 0.0
        else:
            # iterate over each column and keep only the top-k similar items
            values, rows, cols = [], [], []
            nitems = self.dataset.shape[1]
            for i in range(nitems):
                top_k_idx = idx_sorted[-self.k:, i]
                values.extend(item_weights[top_k_idx, i])
                rows.extend(np.arange(nitems)[top_k_idx])
                cols.extend(np.ones(self.k) * i)
            self.W_sparse = sps.csc_matrix((values, (rows, cols)), shape=(nitems, nitems), dtype=np.float32)

    def recommend(self, user_id, n=None, exclude_seen=True):
        '''
            Makes a recommendation list based on the item similarities and the
            actual rating of the item. 
        '''

        # compute the scores using the dot product
        user_profile = self._get_user_ratings(user_id)

        if self.sparse_weights:
            scores = user_profile.dot(self.W_sparse).toarray().ravel()
        else:
            scores = user_profile.dot(self.W).ravel()

        if self.normalize:
            # normalization will keep the scores in the same range
            # of value of the ratings in dataset
            rated = user_profile.copy()
            rated.data = np.ones_like(rated.data)
            if self.sparse_weights:
                den = rated.dot(self.W_sparse).toarray().ravel()
            else:
                den = rated.dot(self.W).ravel()
            den[np.abs(den) < 1e-6] = 1.0  # to avoid NaNs
            scores /= den
        # rank items
        ranking = scores.argsort()[::-1]
        if exclude_seen:
            ranking = self._filter_seen(user_id, ranking)
        return ranking[:n]

    def recommend_new_user(self, user_profile, n=None, exclude_seen=True):
        # compute the scores using the dot product
        if self.sparse_weights:
            assert user_profile.shape[1] == self.W_sparse.shape[0], 'The number of items does not match!'
            scores = user_profile.dot(self.W_sparse).toarray().ravel()
        else:
            assert user_profile.shape[1] == self.W.shape[0], 'The number of items does not match!'
            scores = user_profile.dot(self.W).ravel()
        if self.normalize:
            # normalization will keep the scores in the same range
            # of value of the ratings in dataset
            rated = user_profile.copy()
            rated.data = np.ones_like(rated.data)
            if self.sparse_weights:
                den = rated.dot(self.W_sparse).toarray().ravel()
            else:
                den = rated.dot(self.W).ravel()
            den[np.abs(den) < 1e-6] = 1.0  # to avoid NaNs
            scores /= den
        # rank items
        ranking = scores.argsort()[::-1]
        if exclude_seen:
            seen = user_profile.indices
            unseen_mask = np.in1d(ranking, seen, assume_unique=True, invert=True)
            ranking = ranking[unseen_mask]
        return ranking[:n]

    def label(self, unlabeled_list, binary_ratings=False, n=None, exclude_seen=True, p_most=1, n_most=3):
        # Shuffle the unlabeled list of tuples (user_idx, item_idx).
        np.random.shuffle(unlabeled_list)

        # TODO: Instead of just labeling p + n items, label p_most and n_most as the
        #       original algorithm says.
        labels = []
        number_labeled = 0
        for user_idx, item_idx in unlabeled_list:
            # compute the scores using the dot product
            user_profile = self._get_user_ratings(user_idx)
            if self.sparse_weights:
                scores = user_profile.dot(self.W_sparse).toarray().ravel()
            else:
                scores = user_profile.dot(self.W).ravel()

            if self.normalize:
                # normalization will keep the scores in the same range
                # of value of the ratings in dataset
                rated = user_profile.copy()
                rated.data = np.ones_like(rated.data)
                if self.sparse_weights:
                    den = rated.dot(self.W_sparse).toarray().ravel()
                else:
                    den = rated.dot(self.W).ravel()
                den[np.abs(den) < 1e-6] = 1.0  # to avoid NaNs
                scores /= den

            # pdb.set_trace()
            if ((not(binary_ratings) and scores[item_idx] >= 1.0 and scores[item_idx] <= 5.0) \
                or \
                (binary_ratings and scores[item_idx] >= 0.0 and scores[item_idx] <= 1.0) ):
                labels.append( (user_idx, item_idx, scores[item_idx]) )
                number_labeled += 1

            if (number_labeled == p_most + n_most):
                break

        return labels
