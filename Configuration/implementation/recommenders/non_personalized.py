'''
Politecnico di Milano.
item_knn.py

Description: This file contains the definition and implementation of some
             NonPersonalized-based Recommender.

Created by: Massimo Quadrana.
Modified by: Fernando Pérez.

Last modified on 05/09/2017.
'''

import numpy as np
import scipy.sparse as sps
from .base import Recommender, check_matrix

class Random(Recommender):
    """Class that represents a random recommender.

        Attributes:
            * seed: a seed to generate the random values in order to make them
                    reproduceable.
            * random_state: a Numpy.Random.RandomState instance that generates
                            the random scores.
            * binary_ratings: tells the recommender if the dataset is of implicit
                              or explicit interactions.

            * n_users: Number of users in the dataset.
            * n_items: Number of items in the dataset.
    """
    def __init__(self,seed=1234,binary_ratings=False):
        """Constructor of the class.

            Args:
                * seed: int
                * random_state: Numpy.random.RandomState instance
                * binary_ratings: bool

            Args type:
                * seed: int
                * random_state: Numpy.random.RandomState instance
                * binary_ratings: bool

        """
        super(Random, self).__init__()
        self.seed = seed
        self.random_state = np.random.RandomState(seed=self.seed)
        self.binary_ratings = binary_ratings

    def __str__(self):
        """ String representation of the class. """
        return "Random(seed={},random_state={},binary_ratings={})".format(
            self.seed,
            self.random_state,
            self.binary_ratings
        )

    def short_str(self):
        """ Short string used for dictionaries. """
        return "Random"

    def fit(self, X):
        """Trains and builds the model given a dataset.

            The fit function inside the Random class stores the dataset, the
            number of users and the number of items.

            Args:
                * X: User-Rating Matrix for which we will train the model.

            Args type:
                * X: Scipy.Sparse matrix.
        """
        X = check_matrix(X, 'csr', dtype=np.float32)
        self.dataset = X
        self.nusers, self.nitems = X.shape

    def recommend(self, user_id, n=None, exclude_seen=True):
        """Makes a top-N recommendation list for a specific user.

            Makes a top-N recommendation by choosing at random item indices.

            Args:
                * user_id: user index to which we will build the top-N list.
                * n: size of the list.
                * exclude_seen: tells if we should remove already-seen items from
                                the list.

            Args type:
                * user_id: int
                * n: int
                * exclude_seen: bool
                * score_mode: str

            Returns:
                A non-personalised ranked list of items represented by their indices.
        """
        ranking = self.random_state.random_integers(low=0, high=self.nitems-1, size=(self.nitems,))
        if exclude_seen:
            ranking = self._filter_seen(user_id, ranking)
        return ranking[:n]

    def predict(self, user_id, rated_indices,score_mode='user'):
        """Calculates the predicted preference of a user for a list of items.

            For the Random class it takes a value between 0 and 1 if using an
            implicit dataset, otherwise, an integer between 1 and 5.

            Args:
                * user_id: user index to which we will build the top-N list.
                * rated_indices: list that holds the items for which we will
                                 predict the user preference.
                * score_mode: calculate the prediction sequentially by user, in
                              batches or performing matrix operations.

            Args type:
                * user_id: int
                * rated_indices: list of int.
                * score_mode: str.

            Returns:
                A list of predicted preferences for each item in the list given.
        """
        # 5) compute the predicted ratings using element-wise sum of the elements.
        # r_ui = mu + bu + bi
        if (score_mode == 'user'):
            shape = rated_indices.shape
            if (self.binary_ratings):
                # For each rated index guess a rating by random choice.
                return self.random_state.random_integers(low=0, high=1, size=shape)
            else:
                # For each rated index guess a rating by random choice.
                return self.random_state.random_integers(low=1, high=5, size=shape)

class TopPop(Recommender):
    """Class that represents a Top Popular recommender.

        Attributes:
            * dataset: the dataset used to train the recommender.
            * pop: a ranked list from the most-rated item to the lowest.
    """

    def __init__(self):
        """ Constructor of the class. """
        super(TopPop, self).__init__()

    def __str__(self):
        """ String representation of the class. """
        return "TopPop"

    def short_str(self):
        """ Short string used for dictionaries. """
        return "TopPop"

    def fit(self, X):
        """Trains and builds the model given a dataset.

            The fit function inside the TopPop class counts the number of ratings
            that each rating had received and stores the indices of the most
            popular items (most rated) on `self.pop`.

            Args:
                * X: User-Rating Matrix for which we will train the model.

            Args type:
                * X: Scipy.Sparse matrix.
        """
        X = check_matrix(X, 'csr', dtype=np.float32)
        self.dataset = X
        # convert to csc matrix for faster column-wise sum
        X = check_matrix(X, 'csc', dtype=np.float32)
        item_pop = (X > 0).sum(axis=0)  # this command returns a numpy.matrix of size (1, nitems)
        item_pop = np.asarray(item_pop).squeeze()  # necessary to convert it into a numpy.array of size (nitems,)
        self.pop = np.argsort(item_pop)[::-1]

    def recommend(self, user_id, n=None, exclude_seen=True):
        """Makes a top-N recommendation list for a specific user.

            Creates the ranking list by taking the most-popular items and removing
            already-seen items if necessary.

            Args:
                * user_id: user index to which we will build the top-N list.
                * n: size of the list.
                * exclude_seen: tells if we should remove already-seen items from
                                the list.

            Args type:
                * user_id: int
                * n: int
                * exclude_seen: bool
                * score_mode: str

            Returns:
                A non-personalised ranked list of items represented by their indices.
        """
        ranking = self.pop
        if exclude_seen:
            ranking = self._filter_seen(user_id, ranking)
        return ranking[:n]

    def predict(self, user_id, rated_indices,score_mode='user'):
        """Calculates the predicted preference of a user for a list of items.

            For the TopPop class this method does nothing.

            Args:
                * user_id: user index to which we will build the top-N list.
                * rated_indices: list that holds the items for which we will
                                 predict the user preference.

            Args type:
                * user_id: int
                * rated_indices: list of int.

            Returns:
                A list of predicted preferences for each item in the list given.
        """
        pass

class GlobalEffects(Recommender):
    """Class that represents a Global Effects recommender.

        Attributes:
            * lambda_user: shrinkage term for the user bias.
            * lambda_item: shrinkage term for the item bias.
            * mu: global average.
            * bi: represent the bias for each item.
            * bu: represents the bias for each user.
            * dataset: the dataset used to train the recommender.
            * item_ranking: a ranked list from the highest-scored item to the lowest.
    """

    def __init__(self, lambda_user=10, lambda_item=25):
        """Constructor of the class.

            Args:
                * lambda_user: shrinkage term for the user bias.
                * lambda_item: shrinkage term for the item bias.

            Args type:
                * seed: int
                * random_state: Numpy.random.RandomState instance
                * binary_ratings: bool

        """
        super(GlobalEffects, self).__init__()
        self.lambda_user = lambda_user
        self.lambda_item = lambda_item

    def __str__(self):
        """ String representation of the class. """
        return 'GlobalEffects'

    def short_str(self):
        """ Short string used for dictionaries. """
        return "GlobalEffects"

    def fit(self, X):
        """Trains and builds the model given a dataset.

            The fit function inside the GlobalEffects class stores the dataset,
            and calculate the global bias $\mu$, the item bias $b_{i}$ and the
            user bias $b_{u}$ for each item, and each user.

            Args:
                * X: User-Rating Matrix for which we will train the model.

            Args type:
                * X: Scipy.Sparse matrix.
        """
        # pdb.set_trace()
        X = check_matrix(X, 'csr', dtype=np.float32)
        self.dataset = X
        # convert to csc matrix for faster column-wise sum
        X = check_matrix(X, 'csc', dtype=np.float32)
        # 1) global average
        self.mu = X.data.sum(dtype=np.float32) / X.data.shape[0]

        # 2) item average bias
        # compute the number of non-zero elements for each column
        col_nnz = np.diff(X.indptr)

        # it is equivalent to:
        # col_nnz = X.indptr[1:] - X.indptr[:-1]
        # and it is **much faster** than
        # col_nnz = (X != 0).sum(axis=0)

        X_unbiased = X.copy()
        X_unbiased.data -= self.mu
        self.bi = X_unbiased.sum(axis=0) / (col_nnz + self.lambda_item)
        self.bi = np.asarray(self.bi).ravel()  # converts 2-d matrix to 1-d array without anycopy

        # 3) user average bias
        # NOTE: the user bias is *useless* for the sake of ranking items. We just show it here for educational purposes.

        # first subtract the item biases from each column
        # then repeat each element of the item bias vector a number of times equal to col_nnz
        # and subtract it from the data vector
        X_unbiased.data -= np.repeat(self.bi, col_nnz)

        # now convert the csc matrix to csr for efficient row-wise computation
        X_csr = X_unbiased.tocsr()
        row_nnz = np.diff(X_csr.indptr)
        # finally, let's compute the bias
        self.bu = X_csr.sum(axis=1).ravel() / (row_nnz + self.lambda_user)
        self.bu = np.squeeze(np.asarray(self.bu)) # To put it in an array

        # 4) precompute the item ranking by using the item bias only
        # the global average and user bias won't change the ranking, so there is no need to use them
        self.item_ranking = np.argsort(self.bi)[::-1]

        # Remove nans in bi and bu.
        self.bi[np.abs(self.bi) < 1e-6] = 0.0
        self.bu[np.abs(self.bu) < 1e-6] = 0.0

    def recommend(self, user_id, n=None, exclude_seen=True):
        """Makes a top-N recommendation list for a specific user.

            Creates the ranking list from the item_ranking, which is an ordered
            list from the highest scored item to the lowest scored.

            Args:
                * user_id: user index to which we will build the top-N list.
                * n: size of the list.
                * exclude_seen: tells if we should remove already-seen items from
                                the list.

            Args type:
                * user_id: int
                * n: int
                * exclude_seen: bool
                * score_mode: str

            Returns:
                A non-personalised ranked list of items represented by their indices.
        """
        ranking = self.item_ranking
        if exclude_seen:
            ranking = self._filter_seen(user_id, ranking)
        return ranking[:n]

    def predict(self, user_id, rated_indices,score_mode='user'):
        """Calculates the predicted preference of a user for a list of items.

            For the GlobalEffects class this method predicts the rating for a
            user u and item i as in the following equation.

                \hat{r}_{u,i} = \mu + b_{i} + b_{u}

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
        # 5) compute the predicted ratings using element-wise sum of the elements.
        # r_ui = mu + bu + bi
        if (score_mode == 'user'):
            mu = self.mu
            bu = self.bu[user_id]
            bi = self.bi[rated_indices]
            return mu + bu + bi
