'''
Politecnico di Milano.
item_knn.py

Description: This file contains the definition and implementation of a
             UserKNN-based Recommender.

Created by: Massimo Quadrana.
Modified by Fernando Pérez.

Last modified on 25/03/2017.
'''


from .item_knn import ItemKNNRecommender
from .base import check_matrix
import numpy as np
import scipy.sparse as sps
import pdb


class UserKNNRecommender(ItemKNNRecommender):
    def __init__(self, k=50, shrinkage=100, similarity='cosine', normalize=False, sparse_weights=True):
        super().__init__(
            k=k,
            shrinkage=shrinkage,
            similarity=similarity,
            normalize=normalize,
            sparse_weights=sparse_weights
        )

    def short_str(self):
        return "UserKNN"

    def __str__(self):
        return "UserKNN(similarity={},k={},shrinkage={},normalize={},sparse_weights={})".format(
            self.similarity_name, self.k, self.shrinkage, self.normalize, self.sparse_weights)

    def fit(self, X):
        # convert X to csr matrix for faster row-wise operations
        X = check_matrix(X, 'csr', dtype=np.float32)
        M, N = X.shape
        Xt = X.T.tocsr()
        # fit a ItemKNNRecommender on the transposed X matrix
        super().fit(Xt)

        self.dataset = X

        # # precompute the predicted scores for speed
        # if self.sparse_weights:
        #     self.scores = self.W_sparse.dot(X).toarray()
        # else:
        #     self.scores = self.W.dot(X)
        # if self.normalize:
        #     for i in range(M): # <- Bug here, should be N instead of M.
        #         pdb.set_trace()
        #         rated = Xt[i].copy()
        #         rated.data = np.ones_like(rated.data)
        #         if self.sparse_weights:
        #             den = rated.dot(self.W_sparse).toarray().ravel()
        #         else:
        #             den = rated.dot(self.W).ravel()
        #         den[np.abs(den) < 1e-6] = 1.0  # to avoid NaNs
        #         self.scores[:, i] /= den

    def calculate_scores(self):
        if self.sparse_weights:
            self.scores = self.W_sparse.dot(self.dataset).toarray()
        else:
            self.scores = self.W.dot(self.dataset)

        if self.normalize:
            # pdb.set_trace()
            rated = self.dataset.T.copy()
            rated.data = np.ones_like(rated.data)
            if self.sparse_weights:
                den = rated.dot(self.W_sparse).toarray()
            else:
                den = rated.dot(self.W)
            den[np.abs(den) < 1e-6] = 1.0  # to avoid NaNs
            self.scores /= den.T

    def recommend(self, user_id, n=None, exclude_seen=True):
        # pdb.set_trace()
        if (self.scores is None):
            self.calculate_scores()

        ranking = self.scores[user_id].argsort()[::-1]
        if exclude_seen:
            ranking = self._filter_seen(user_id, ranking)
        return ranking[:n]

    def label(self, unlabeled_list, binary_ratings=False, n=None, exclude_seen=True, p_most=1, n_most=3):
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

        # At this point, we have all the predicted scores for the users inside
        # U'. Now we will filter the scores by keeping only the scores of the
        # items presented in U'. This will be an array where:
        # filtered_scores[i] = scores[users[i],items[i]]
        filtered_scores = self.scores[users,items]

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

    def predict(self, user_id, rated_indices):
        # return the scores for the rated items.
        return self.scores[user_id,rated_indices]
