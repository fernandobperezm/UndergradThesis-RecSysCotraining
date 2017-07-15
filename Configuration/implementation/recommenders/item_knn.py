'''
Politecnico di Milano.
item_knn.py

Description: This file contains the definition and implementation of a
             ItemKNN-based Recommender.

Created by: Massimo Quadrana.
Modified by Fernando Pérez.

Last modified on 25/03/2017.
'''

import random as random

import numpy as np
import scipy.sparse as sps
from .base import Recommender, check_matrix
from .similarity import Cosine, Pearson, AdjustedCosine

import pdb

class ItemKNNRecommender(Recommender):
    """ ItemKNN recommender"""

    def __init__(self, k=50, shrinkage=100, similarity='cosine', normalize=False, sparse_weights=True):
        super(ItemKNNRecommender, self).__init__()
        self.k = k
        self.shrinkage = shrinkage
        self.normalize = normalize
        self.dataset = None
        self.similarity_name = similarity
        self.sparse_weights = sparse_weights
        self.scores = None
        if similarity == 'cosine':
            self.distance = Cosine(shrinkage=self.shrinkage)
        elif similarity == 'pearson':
            self.distance = Pearson(shrinkage=self.shrinkage)
        elif similarity == 'adj-cosine':
            self.distance = AdjustedCosine(shrinkage=self.shrinkage)
        else:
            raise NotImplementedError('Distance {} not implemented'.format(similarity))

    def short_str(self):
        return "ItemKNN"

    def __str__(self):
        return "ItemKNN(similarity={},k={},shrinkage={},normalize={},sparse_weights={})".format(
            self.similarity_name, self.k, self.shrinkage, self.normalize, self.sparse_weights)

    def fit(self, X):
        '''
            X: represents the dataset. It must be of type
        '''
        # convert X to csr matrix for faster row-wise operations
        X = check_matrix(X, 'csr', dtype=np.float32)
        self.dataset = X
        self.scores = None
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
            self.W_sparse = sps.csr_matrix((values, (rows, cols)), shape=(nitems, nitems), dtype=np.float32)

    def calculate_scores_matrix(self):
        if self.sparse_weights:
            self.scores = self.dataset.dot(self.W_sparse).toarray()
        else:
            self.scores = self.dataset.dot(self.W)

        if self.normalize:
            # normalization will keep the scores in the same range
            # of value of the ratings in dataset
            rated = self.dataset.copy()
            rated.data = np.ones_like(rated.data)
            if self.sparse_weights:
                den = rated.dot(self.W_sparse).toarray()
            else:
                den = rated.dot(self.W)
            den[np.abs(den) < 1e-6] = 1.0  # to avoid NaNs
            self.scores /= den

    def calculate_scores_batch(self,users):
        u_begin, u_stop = users.min(), users.max()
        partition_size = 1000
        partitions = np.arange(start=u_begin,stop=u_stop+1,step=partition_size,dtype=np.int32)
        self.scores = np.zeros(shape=self.dataset.shape,dtype=np.float32,order='C')
        for low_user in partitions:
            # As u_stop is an index to take into account and indices ranges slices
            # don't take into account the last, we need to sum one to include
            # this index.
            high_user = min(low_user + partition_size, u_stop+1) # to not exceed csr matrices indices.
            profiles = self._get_user_ratings(range(low_user,high_user))
            if self.sparse_weights:
                self.scores[low_user:high_user] = profiles.dot(self.W_sparse).toarray()
            else:
                self.scores[low_user:high_user] = profiles.dot(self.W)

            if self.normalize:
                # normalization will keep the scores in the same range
                # of value of the ratings in dataset
                rated = profiles.copy()
                rated.data = np.ones_like(rated.data)
                if self.sparse_weights:
                    den = rated.dot(self.W_sparse).toarray()
                else:
                    den = rated.dot(self.W)
                den[np.abs(den) < 1e-6] = 1.0  # to avoid NaNs
                self.scores[low_user:high_user] /= den

    def calculate_scores_user(self,user_id):
        user_profile = self._get_user_ratings(user_id)

        if self.sparse_weights:
            self.scores = user_profile.dot(self.W_sparse).toarray().ravel()
        else:
            self.scores = user_profile.dot(self.W).ravel()

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
            self.scores /= den

    def recommend(self, user_id, n=None, exclude_seen=True,score_mode='user'):
        if (score_mode == 'user'):
            self.calculate_scores_user(user_id)
            ranking = self.scores.argsort()[::-1]
        elif (score_mode == 'matrix' and self.scores is None):
            self.calculate_scores_matrix()
            ranking = self.scores[user_id].argsort()[::-1]

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

    def predict(self, user_id, rated_indices,score_mode='user'):
        # return the scores for the rated items.
        # return the scores for the rated items.
        if (score_mode == 'user'):
            return self.scores[rated_indices]
        elif (score_mode == 'matrix'):
            return self.scores[user_id,rated_indices]

    def label(self, unlabeled_list, binary_ratings=False, n=None, exclude_seen=True, p_most=1, n_most=3, score_mode='batch'):
        '''
            Arguments:
                * unlabeled_list: Its the pool of user/item keys without ratings.
                                  from which we will label. It must be an instance
                                  of a LIL matrix.

        '''

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

        if (score_mode == 'batch'):
            uniq_users, user_to_idx = np.unique(users,return_inverse=True)
            self.calculate_scores_batch(uniq_users)
            filtered_scores = self.scores[user_to_idx,items]

        if (score_mode == 'matrix'):
            if (self.scores is None):
                self.calculate_scores_matrix()
            filtered_scores = self.scores[users,items]

        # Up to this point, we have all the predicted scores for the users inside
        # U'. Now we will filter the scores by keeping only the scores of the
        # items presented in U'. This will be an array where:
        # filtered_scores[i] = scores[users[i],items[i]]

        # positive ratings: explicit ->[4,5], implicit -> [0.75,1]
        # negative ratings: explicit -> [1,2,3], implicit -> [0,0.75)
        # Creating a mask to remove elements out of bounds.
        if (binary_ratings):
            p_mask = (filtered_scores >= 0.75) & (filtered_scores <= 1)
            n_mask = (filtered_scores >= 0.0) & (filtered_scores < 0.75)
        else:
            p_mask = (filtered_scores >= 4.0) & (filtered_scores <= 5.0)
            n_mask = (filtered_scores >= 1.0) & (filtered_scores <= 3.0)

        # In order to have the same array structure as mentioned before. Only
        # keeps positive ratings.
        p_users = users[p_mask]
        p_items = items[p_mask]
        p_filtered_scores = filtered_scores[p_mask]

        # In order to have the same array structure as mentioned before. Only
        # keeps negative ratings.
        n_users = users[n_mask]
        n_items = items[n_mask]
        n_filtered_scores = filtered_scores[n_mask]

        # Filtered the scores to have the n-most and p-most.
        # The p-most are sorted decreasingly.
        # The n-most are sorted incrementally.
        p_sorted_scores = p_filtered_scores.argsort()[::-1]
        n_sorted_scores = n_filtered_scores.argsort()

        # Taking the p_most positive, in decreasing order, if len(p_sorted_scores) < p_most
        # the final array will have length of len(p_sorted_scores)
        p_sorted_scores = p_sorted_scores[:p_most]

        # Similar to p_most but with n_most.
        n_sorted_scores = n_sorted_scores[:n_most]

        scores = [(p_users[i], p_items[i], p_filtered_scores[i]) for i in p_sorted_scores ] + [(n_users[i], n_items[i], n_filtered_scores[i]) for i in n_sorted_scores]

        # We sort the indices by user, then by item in order to make the
        # assignment to the LIL matrix faster.
        return sorted(scores, key=lambda triplet: (triplet[0],triplet[1]))
