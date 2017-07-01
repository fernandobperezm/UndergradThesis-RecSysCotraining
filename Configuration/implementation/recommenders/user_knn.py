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
import numpy as np
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
        X = X.tocsr()
        M, N = X.shape
        Xt = X.T.tocsr()
        # fit a ItemKNNRecommender on the transposed X matrix
        super().fit(Xt)
        self.dataset = X
        # precompute the predicted scores for speed
        if self.sparse_weights:
            self.scores = self.W_sparse.dot(X).toarray()
        else:
            self.scores = self.W.dot(X)
        if self.normalize:
            for i in range(M):
                rated = Xt[i].copy()
                rated.data = np.ones_like(rated.data)
                if self.sparse_weights:
                    den = rated.dot(self.W_sparse).toarray().ravel()
                else:
                    den = rated.dot(self.W).ravel()
                den[np.abs(den) < 1e-6] = 1.0  # to avoid NaNs
                self.scores[:, i] /= den

    def user_score(self, user_id):
        return self.scores[user_id]

    def recommend(self, user_id, n=None, exclude_seen=True):
        ranking = self.scores[user_id].argsort()[::-1]
        if exclude_seen:
            ranking = self._filter_seen(user_id, ranking)
        return ranking[:n]

    def label(self, unlabeled_list, binary_ratings=False, n=None, exclude_seen=True, p_most=1, n_most=3):
        # Shuffle the unlabeled list of tuples (user_idx, item_idx).
        # Labeling of p-most positive and n-most negative ratings.
        np.random.shuffle(unlabeled_list)

        labels = []
        number_p_most_labeled = 0
        number_n_most_labeled = 0
        for user_idx, item_idx in unlabeled_list:
            # For this recommender, all the sparse_weights and normalization is made
            # on the fit function instead of recommendation.
            scores = self.scores[user_idx]

            # pdb.set_trace()
            # positive ratings: [4,5]
            # negative ratings: [1,2,3]
            if (number_p_most_labeled < p_most):
                if ((not(binary_ratings) and scores[item_idx] >= 4.0 and scores[item_idx] <= 5.0) \
                    or \
                    (binary_ratings and scores[item_idx] == 1.0) ):
                    labels.append( (user_idx, item_idx, scores[item_idx]) )
                    number_p_most_labeled += 1

            if (number_n_most_labeled < n_most):
                if ((not(binary_ratings) and scores[item_idx] >= 1.0 and scores[item_idx] <= 3.0) \
                    or \
                    (binary_ratings and scores[item_idx] == 0.0) ):
                    labels.append( (user_idx, item_idx, scores[item_idx]) )
                    number_n_most_labeled += 1

            if (number_p_most_labeled == p_most and number_n_most_labeled == n_most):
                break

        return labels

    def predict(self, user_id, rated_indices):
        # return the scores for the rated items.
        return self.scores[user_id,rated_indices]
