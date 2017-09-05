'''
Politecnico di Milano.
item_knn.py

Description: This file contains the definition and implementation of a
             ItemKNN-based Recommender.

Created by: Massimo Quadrana.
Modified by: Fernando Pérez.

Last modified on 05/09/2017.
'''

import random as random

import numpy as np
import scipy.sparse as sps
from .base import Recommender, check_matrix
from .similarity import Cosine, Pearson, AdjustedCosine


class ItemKNNRecommender(Recommender):
    """Class that implements an ItemKNN recommender system.

       This type of recommender build a similarity matrix between the items
       and calculates the score for an unknown rating

       Attributes:
            * k: represents the number of neirest neighbors for each item.
            * shrinkage: shrinkage to be applied to reduce the effect of
                         items with not many ratings.
            * normalize: normalize the scores between a range.
            * dataset: dataset which will be used to build the model.
            * similarity_name: name of the similarity function to apply.
            * sparse_weights: consider sparse or dense representation to
                              store the k-most-similar items.
            * scores: the scores are already calculated or not.
            * distance: instance of the similarity function to apply.
            * W: the top-k most similar items for each item in a dense representation.
            * W_sparse: the top-k most similar items for each item in a sparse representation.
    """

    def __init__(self, k=50, shrinkage=100, similarity='cosine', normalize=False, sparse_weights=True):
        """Constructor of the ItemKNNRecommender class.

           Args:
                * k: represents the number of neirest neighbors for each item.
                * shrinkage: shrinkage to be applied to reduce the effect of
                             items with not many ratings.
                * normalize: normalize the scores between a range.
                * similarity: name of similarity function to apply.
                * sparse_weights: consider sparse or dense representation to
                                  store the k-most-similar items.

           Args type:
                * k = k
                * shrinkage: int
                * normalize: bool
                * similarity: str
                * sparse_weights: bool

        """
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
        """ Short string used for dictionaries. """
        return "ItemKNN"

    def __str__(self):
        """ String representation of the class. """
        return "ItemKNN(similarity={},k={},shrinkage={},normalize={},sparse_weights={})".format(
            self.similarity_name, self.k, self.shrinkage, self.normalize, self.sparse_weights)

    def fit(self, X):
        """Trains and builds the model given a dataset.

            The fit function inside the ItemKNN class builds a similarity matrix
            between all the items using the similarity function as defined in
            `self.distance`. Afterwards, for each item it takes only the top-k
            most similar items to it and stores them inside a matrix that can
            be either dense matrix or Scipy.Sparse.

            Args:
                * X: User-Rating Matrix for which we will train the model.

            Args type:
                * X: Scipy.Sparse matrix.
        """
        # convert X to csr matrix for faster row-wise operations
        X = check_matrix(X, 'csr', dtype=np.float32)
        self.dataset = X
        self.scores = None

        # Calculation of the similarity matrix.
        item_weights = self.distance.compute(X)

        # for each column, keep only the top-k most similar items
        idx_sorted = np.argsort(item_weights, axis=0)

        # Decide to use sparse or dense representations.
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
        """Calculates the score for all the items for all the users.

           This function makes the matrix multiplication between the URM and the
           similarities matrix W. In this way, calculate the predicted score for
           each user for all the items.

           All the scores are stored inside `self.scores`, and can be either
           normalized or not.
        """
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
        """Calculates the score for all the items for a batch of users.

           This function makes the matrix multiplication between the profiles
           of the users inside the batch and the item similarities. This matrix
           multiplication returns the scores for each user and all the items.

           The batch process is done by partitioning the list of users into
           different batches, of an specific size, then

           All the scores are stored inside `self.scores`, and can be either
           normalized or not.

           Args:
                * users: list containing the users indices inside the system.

            Args type:
                * users: list of int.
        """
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
        """Calculates the score for all the items for a batch of users.

           This function makes the matrix multiplication between the profile
           of the user and the item similarities. This matrix multiplication
           returns the predicted score for all the items based on the users
           preferences.

           All the scores are stored inside `self.scores`, and can be either
           normalized or not.

           Args:
                * user_id: the user index inside the system.

            Args type:
                * users: int.
        """

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

    def recommend(self, user_id, n=None, exclude_seen=True, score_mode='user'):
        """Makes a top-N recommendation list for a specific user.

            Args:
                * user_id: user index to which we will build the top-N list.
                * n: size of the list.
                * exclude_seen: tells if we should remove already-seen items from
                                the list.
                * score_mode: the score is created only for one user or it is
                              created for all the users.

            Args type:
                * user_id: int
                * n: int
                * exclude_seen: bool
                * score_mode: str

            Returns:
                A personalised ranked list of items represented by their indices.
        """
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
        """Makes a top-N recommendation list for a new user user.

            Args:
                * user_id: user index to which we will build the top-N list.
                * n: size of the list.
                * exclude_seen: tells if we should remove already-seen items from
                                the list.

            Args type:
                * user_id: int
                * n: int
                * exclude_seen: bool

            Returns:
                A personalised ranked list of items represented by their indices.
        """
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
        """Calculates the predicted preference of a user for a list of items.

            Args:
                * user_id: user index to which we will build the top-N list.
                * rated_indices: list that holds the items for which we will
                                 predict the user preference.
                * score_mode: the score is created only for one user or it is
                              created for all the users.

            Args type:
                * user_id: int
                * rated_indices: list of int.

            Returns:
                A list of predicted preferences for each item in the list given.
        """
        # return the scores for the rated items.
        if (score_mode == 'user'):
            return self.scores[rated_indices]
        elif (score_mode == 'matrix'):
            return self.scores[user_id,rated_indices]

    def label(self, unlabeled_list, binary_ratings=False, exclude_seen=True, p_most=1, n_most=3, score_mode='user'):
        """Rates new user-item pairs.

           This function is part of the Co-Training process in which we rate
           all user-item pairs inside an unlabeled pool of samples, afterwards,
           we separate them into positive and negative items based on their score.
           Lastly, we take the p-most positive and n-most negative items from all
           the rated items.

           Inside the function we also measure some statistics that help us to
           analyze the effects of the Co-Training process, such as, number of
           positive, negative and neutral items rated and sets of positive, negative
           and neutral user-item pairs to see the agreement of the recommenders.
           We put all these inside a dictionary.

           Args:
               * unlabeled_list: a matrix that holds the user-item that we must
                                 predict their rating.
               * binary_ratings: tells us if we must predict based on an implicit
                                 (0,1) dataset or an explicit.
               * exclude_seen: tells us if we need to exclude already-seen items.
               * p_most: tells the number of p-most positive items that we
                         should choose.
               * n_most: tells the number of n-most negative items that we
                         should choose.
               * score_mode: the type of score prediction, 'user' represents by
                             sequentially user-by-user, 'batch' represents by
                             taking batches of users, 'matrix' represents to
                             make the preditions by a matrix multiplication.

           Args type:
               * unlabeled_list: Scipy.Sparse matrix.
               * binary_ratings: bool
               * exclude_seen: bool
               * p_most: int
               * n_most: int
               * score_mode: str

           Returns:
               A list containing the user-item-rating triplets and the meta
               dictionary for statistics.
        """

        unlabeled_list = check_matrix(unlabeled_list, 'lil', dtype=np.float32)
        users,items = unlabeled_list.nonzero()
        n_scores = len(users)
        # users is an array where the user is repeated incrementally.
        # items that for the same user is incrementally and for other users
        # it can decrease, example:
        # users = [0,0,0,0,0, 1,1,1, 2], items = [1,6,8,9,19, 0,4,5, 2]

        if (score_mode == 'user'):
            filtered_scores = np.zeros(shape=n_scores,dtype=np.float32)
            curr_user = None
            i = 0
            for user,item in zip(users,items):
                if (curr_user != user):
                    curr_user = user
                    self.calculate_scores_user(curr_user)

                filtered_scores[i] = self.scores[item]
                i += 1

        elif (score_mode == 'batch'):
            filtered_scores = []
            uniq_users, user_to_idx = np.unique(users,return_inverse=True)
            self.calculate_scores_batch(uniq_users)
            filtered_scores = self.scores[users,items]

        elif (score_mode == 'matrix'):
            if (self.scores is None):
                self.calculate_scores_matrix()
            filtered_scores = self.scores[users,items]


        # Up to this point, we have all the predicted scores for the users inside
        # U'. Now we will filter the scores by keeping only the scores of the
        # items presented in U'. This will be an array where:
        # filtered_scores[i] = scores[users[i],items[i]]
        # filtered_scores = self.scores[users,items]

        # positive ratings: explicit ->[3.5,..], implicit -> [1.0]
        # negative ratings: explicit -> [..,2.5], implicit -> [0.0]
        # Creating a mask to remove elements out of bounds.
        if (binary_ratings):
            p_mask = (filtered_scores == 1.0)
            n_mask = (filtered_scores == 0.0)
        else:
            p_mask = (filtered_scores >= 3.5)
            n_mask = (filtered_scores <= 2.5)
            neutral_mask = (filtered_scores > 2.5) & (filtered_scores < 3.5)

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

        # Keeping neutral ratings.
        neutral_users = users[neutral_mask]
        neutral_items = items[neutral_mask]
        neutral_filtered_scores = filtered_scores[neutral_mask]

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

        scores = \
         [(p_users[i], p_items[i], p_filtered_scores[i] if p_filtered_scores[i] < 5.0 else 5.0) for i in p_sorted_scores] +\
         [(n_users[i], n_items[i], n_filtered_scores[i] if n_filtered_scores[i] > 1.0 else 1.0) for i in n_sorted_scores]

        # Creation of statistic sets begin here.
        meta = dict()
        meta['pos_labels'] = len(p_sorted_scores)
        meta['neg_labels'] = len(n_sorted_scores)
        meta['total_labels'] = len(p_sorted_scores) + len(n_sorted_scores)
        meta['pos_set'] = set(zip(p_users, p_items))
        meta['neg_set'] = set(zip(n_users, n_items))
        meta['neutral_set'] = set(zip(neutral_users, neutral_items))

        # We sort the indices by user, then by item in order to make the
        # assignment to the LIL matrix faster.
        return sorted(scores, key=lambda triplet: (triplet[0],triplet[1])), meta
