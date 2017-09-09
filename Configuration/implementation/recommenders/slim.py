'''
Politecnico di Milano.
slim.py

Description: This file contains the definition and implementation of SLIM-based
             Recommenders, such as SLIM and Multi-Thread-SLIM.

Created by: Massimo Quadrana.
Modified by Fernando Pérez.

Last modified on 05/09/2017.
'''

import numpy as np
import scipy.sparse as sps
from .base import Recommender, check_matrix
from sklearn.linear_model import ElasticNet

# Memory consumption problem solved by:
# https://stackoverflow.com/questions/38140693/python3-multiprocessing-consumes-extensively-much-ram-and-slows-down
def _partial_fit(triplet, X):
    """Performs a partial fit of the ElasticNet model.

        This function takes a column and the dataset and performs solves the
        ElasticNet problem for that column, it makes several dataset copies
        to avoid race conditions.

        Args:
            * triplet: contains the column index, the l1_ratio and positive_only
                        in this order.
            * X: The dataset in which we build the model.

        Args type:
            * triplet: (int, float, bool)
            * X: Scipy.Sparse matrix.

        Returns:
            A triplet containing the values, the rows and the columns of the
            model built.

    """
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

    Attibutes:
        * dataset: the dataset which we use to build the model.
        * model: the model instance of the ElasticNet.
        * W_sparse: the similarity matrix.
        * l1_penalty: regularization term for the l1 norm.
        * l2_penalty: regularization term for the l2 norm.
        * positive_only: consider positive samples only.
        * l1_ratio: ratio between l1_penalty and l1_penalty + l2_penalty

    """

    def __init__(self,
                 l1_penalty=0.1,
                 l2_penalty=0.1,
                 positive_only=True):
        """Constructor of the SLIM class.

           Args:
                * l1_penalty: regularization term for the l1 norm.
                * l2_penalty: regularization term for the l2 norm.
                * positive_only: consider positive samples only.

           Args type:
                * l1_penalty: float
                * l2_penalty: float
                * positive_only: bool

        """
        super(SLIM, self).__init__()
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.positive_only = positive_only
        self.l1_ratio = self.l1_penalty / (self.l1_penalty + self.l2_penalty)

    def short_str(self):
        """ Short string used for dictionaries. """
        return "SLIM"

    def __str__(self):
        """ String representation of the class. """
        return "SLIM(l1_penalty={},l2_penalty={},positive_only={})".format(
            self.l1_penalty, self.l2_penalty, self.positive_only
        )

    def fit(self, X):
        """Trains and builds the model given a dataset.

            The fit function inside the SLIM class builds a similarity matrix
            between all the items by learning a model solving an ElasticNet
            optimization. The model is built for each item and it returns the
            similarity between that item and the others. From the model we build
            a similarity matrix stored in sparse format and save it on
            `self.W_sparse`

            Args:
                * X: User-Rating Matrix for which we will train the model.

            Args type:
                * X: Scipy.Sparse matrix.
        """
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

    def recommend(self, user_id, n=None, exclude_seen=True):
        """Makes a top-N recommendation list for a specific user.

            The score is calculated by the dot product between the user preferences
            and each item vector in the similarity matrix. The resulting scores
            are then sorted from highest to lowest.

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
                A personalised ranked list of items represented by their indices.
        """
        # compute the scores using the dot product
        user_profile = self._get_user_ratings(user_id)
        scores = user_profile.dot(self.W_sparse).toarray().ravel()
        ranking = scores.argsort()[::-1]
        # rank items
        if exclude_seen:
            ranking = self._filter_seen(user_id, ranking)
        return ranking[:n]

    def predict(self, user_id, rated_indices):
        """Calculates the predicted preference of a user for a list of items.

            The score is calculated by the dot product between the user preferences
            and each item vector in the similarity matrix.

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
        # compute the scores using the dot product
        user_profile = self._get_user_ratings(user_id)
        scores = user_profile.dot(self.W_sparse).toarray().ravel()
        return scores[rated_indices]

    def recommend_new_user(self, user_profile, n=None, exclude_seen=True):
        """Makes a top-N recommendation list for a specific user.

            The score is calculated by the dot product between the user preferences
            and each item vector in the similarity matrix. The resulting scores
            are then sorted from highest to lowest.

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
                A personalised ranked list of items represented by their indices.
        """
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


        meta = dict()
        meta['pos_labels'] = len(p_sorted_scores)
        meta['neg_labels'] = len(n_sorted_scores)
        meta['total_labels'] = len(p_sorted_scores) + len(n_sorted_scores)
        meta['pos_set'] = set([(users[i], items[i]) for i in p_sorted_scores])
        meta['neg_set'] = set([(users[i], items[i]) for i in n_sorted_scores])
        meta['neutral_set'] = set()

        # We sort the indices by user, then by item in order to make the
        # assignment to the LIL matrix faster.
        return sorted(scores, key=lambda triplet: (triplet[0],triplet[1])), meta


from multiprocessing import Pool
from functools import partial


class MultiThreadSLIM(SLIM):
    """
    Train a Sparse Linear Methods (SLIM) item similarity model by parallelizing
    the ElasticNet model building by columns.

    See:
        Efficient Top-N Recommendation by Linear Regression,
        M. Levy and K. Jack, LSRS workshop at RecSys 2013.

        SLIM: Sparse linear methods for top-n recommender systems,
        X. Ning and G. Karypis, ICDM 2011.
        http://glaros.dtc.umn.edu/gkhome/fetch/papers/SLIM2011icdm.pdf

    Attibutes:
        * dataset: the dataset which we use to build the model.
        * model: the model instance of the ElasticNet.
        * W_sparse: the similarity matrix.
        * l1_penalty: regularization term for the l1 norm.
        * l2_penalty: regularization term for the l2 norm.
        * positive_only: consider positive samples only.
        * l1_ratio: ratio between l1_penalty and l1_penalty + l2_penalty
        * workers: maximum number of processes to use.

    """
    def __init__(self,
                 l1_penalty=0.1,
                 l2_penalty=0.1,
                 positive_only=True,
                 workers=4):
        """Constructor of the MultiThreadSLIM class.

           Args:
                * l1_penalty: regularization term for the l1 norm.
                * l2_penalty: regularization term for the l2 norm.
                * positive_only: consider positive samples only.
                * workers: maximum number of processes to use.

           Args type:
                * l1_penalty: float
                * l2_penalty: float
                * positive_only: bool
                * workers: int

        """
        super(MultiThreadSLIM, self).__init__(l1_penalty=l1_penalty,
                                              l2_penalty=l2_penalty,
                                              positive_only=positive_only)
        self.workers = workers
        self.pool = Pool(processes=self.workers)

    def __str__(self):
        """ String representation of the class. """
        return "SLIM_mt(l1_penalty={},l2_penalty={},positive_only={},workers={})".format(
            self.l1_penalty, self.l2_penalty, self.positive_only, self.workers
        )

    def fit(self, X):
        """Trains and builds the model given a dataset.

            The fit function inside the MultiThreadSLIM class builds a similarity
            matrix between all the items by learning a model solving an ElasticNet
            optimization. The model is built for each item and it returns the
            similarity between that item and the others. From the model we build
            a similarity matrix stored in sparse format and save it on
            `self.W_sparse`.

            As said in the paper, we can think every item column as independent
            vectors, thus, we can parallelize the model building by putting in
            different processes each column and solve the ElasticNet problem on
            them.

            Args:
                * X: User-Rating Matrix for which we will train the model.

            Args type:
                * X: Scipy.Sparse matrix.
        """
        X = check_matrix(X, 'csr', dtype=np.float32)
        self.dataset = X
        X = check_matrix(X, 'csc', dtype=np.float32)
        n_items = X.shape[1]
        # fit item's factors in parallel
        _pfit = partial(_partial_fit, X=X)
        num_tasks = int((n_items / self.workers) + 1)

        args_triplet = ((j,self.l1_ratio,self.positive_only) for j in np.arange(n_items))
        res = self.pool.map(_pfit, args_triplet)
        # self.pool.close()
        # self.pool.join()

        # res contains a vector of (values, rows, cols) tuples
        values, rows, cols = [], [], []
        for values_, rows_, cols_ in res:
            values.extend(values_)
            rows.extend(rows_)
            cols.extend(cols_)

        # generate the sparse weight matrix
        self.W_sparse = sps.csr_matrix((values, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)
