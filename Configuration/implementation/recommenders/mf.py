'''
Politecnico di Milano.
item_knn.py

Description: This file contains the definition and implementation of
             Matrix-Factorization-based Recommenders, such as FunkSVD,
             AsymmetricSVD, Alternating Least Squares, BPR-MF.

Created by: Massimo Quadrana.
Modified by Fernando Pérez.

Last modified on 25/03/2017.
'''

import numpy as np
from .base import Recommender, check_matrix
from .._cython._mf import FunkSVD_sgd, AsySVD_sgd, AsySVD_compute_user_factors, BPRMF_sgd
import logging

import pdb

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")


class FunkSVD(Recommender):
    '''
    FunkSVD model
    Reference: http://sifter.org/~simon/journal/20061211.html

    Factorizes the rating matrix R into the dot product of two matrices U and V of latent factors.
    U represent the user latent factors, V the item latent factors.
    The model is learned by solving the following regularized Least-squares objective function with Stochastic Gradient Descent
    \operatornamewithlimits{argmin} \limits_{U,V}\frac{1}{2}||R - UV^T||^2_2 + \frac{\lambda}{2}(||U||^2_F + ||V||^2_F)
    Latent factors are initialized from a Normal distribution with given mean and std.
    '''

    # TODO: add global effects
    def __init__(self,
                 num_factors=50,
                 lrate=0.01,
                 reg=0.015,
                 iters=10,
                 init_mean=0.0,
                 init_std=0.1,
                 lrate_decay=1.0,
                 rnd_seed=42):
        '''
        Initialize the model
        :param num_factors: number of latent factors
        :param lrate: initial learning rate used in SGD
        :param reg: regularization term
        :param iters: number of iterations in training the model with SGD
        :param init_mean: mean used to initialize the latent factors
        :param init_std: standard deviation used to initialize the latent factors
        :param lrate_decay: learning rate decay
        :param rnd_seed: random seed
        '''
        super(FunkSVD, self).__init__()
        self.num_factors = num_factors
        self.lrate = lrate
        self.reg = reg
        self.iters = iters
        self.init_mean = init_mean
        self.init_std = init_std
        self.lrate_decay = lrate_decay
        self.rnd_seed = rnd_seed

    def __str__(self):
        return "FunkSVD(num_factors={}, lrate={}, reg={}, iters={}, init_mean={}, " \
               "init_std={}, lrate_decay={}, rnd_seed={})".format(
            self.num_factors, self.lrate, self.reg, self.iters, self.init_mean, self.init_std, self.lrate_decay,
            self.rnd_seed
        )

    def fit(self, X):
        self.dataset = X
        X = check_matrix(X, 'csr', dtype=np.float32)
        self.U, self.V = FunkSVD_sgd(X, self.num_factors, self.lrate, self.reg, self.iters, self.init_mean,
                                     self.init_std,
                                     self.lrate_decay, self.rnd_seed)

    def recommend(self, user_id, n=None, exclude_seen=True):
        scores = np.dot(self.U[user_id], self.V.T)
        ranking = scores.argsort()[::-1]
        # rank items
        if exclude_seen:
            ranking = self._filter_seen(user_id, ranking)
        return ranking[:n]

    def predict(self, user_id, rated_indices):
        scores = np.dot(self.U[user_id], self.V.T)
        return scores[rated_indices]

    def label(self, unlabeled_list, binary_ratings=False, n=None, exclude_seen=True, p_most=1, n_most=3):
        # Shuffle the unlabeled list of tuples (user_idx, item_idx).
        np.random.shuffle(unlabeled_list)

        # TODO: Instead of just labeling p + n items, label p_most and n_most as the
        #       original algorithm says.
        labels = []
        number_labeled = 0
        for user_idx, item_idx in unlabeled_list:
            # compute the scores using the dot product
            scores = np.dot(self.U[user_idx], self.V.T)

            pdb.set_trace()
            if ( (not(binary_ratings) and scores[item_idx] >= 1.0 and scores[item_idx] <= 5.0) \
                or \
                 (binary_ratings and scores[item_idx] >= 0.0 and scores[item_idx] <= 1.0) ):
                labels.append( (user_idx, item_idx, scores[item_idx]) )
                number_labeled += 1

            if (number_labeled == p_most + n_most):
                break

        return labels


class AsySVD(Recommender):
    '''
    AsymmetricSVD model
    Reference: Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model (Koren, 2008)

    Factorizes the rating matrix R into two matrices X and Y of latent factors, which both represent item latent features.
    Users are represented by aggregating the latent features in Y of items they have already rated.
    Rating prediction is performed by computing the dot product of this accumulated user profile with the target item's
    latent factor in X.

    The model is learned by solving the following regularized Least-squares objective function with Stochastic Gradient Descent
    \operatornamewithlimits{argmin}\limits_{x*,y*}\frac{1}{2}\sum_{i,j \in R}(r_{ij} - x_j^T \sum_{l \in R(i)} r_{il}y_l)^2 + \frac{\lambda}{2}(\sum_{i}{||x_i||^2} + \sum_{j}{||y_j||^2})
    '''

    # TODO: add global effects
    # TODO: recommendation for new-users. Update the precomputed profiles online
    def __init__(self,
                 num_factors=50,
                 lrate=0.01,
                 reg=0.015,
                 iters=10,
                 init_mean=0.0,
                 init_std=0.1,
                 lrate_decay=1.0,
                 rnd_seed=42):
        '''
        Initialize the model
        :param num_factors: number of latent factors
        :param lrate: initial learning rate used in SGD
        :param reg: regularization term
        :param iters: number of iterations in training the model with SGD
        :param init_mean: mean used to initialize the latent factors
        :param init_std: standard deviation used to initialize the latent factors
        :param lrate_decay: learning rate decay
        :param rnd_seed: random seed
        '''
        super(AsySVD, self).__init__()
        self.num_factors = num_factors
        self.lrate = lrate
        self.reg = reg
        self.iters = iters
        self.init_mean = init_mean
        self.init_std = init_std
        self.lrate_decay = lrate_decay
        self.rnd_seed = rnd_seed

    def __str__(self):
        return "AsySVD(num_factors={}, lrate={}, reg={}, iters={}, init_mean={}, " \
               "init_std={}, lrate_decay={}, rnd_seed={})".format(
            self.num_factors, self.lrate, self.reg, self.iters, self.init_mean, self.init_std, self.lrate_decay,
            self.rnd_seed
        )

    def fit(self, R):
        self.dataset = R
        R = check_matrix(R, 'csr', dtype=np.float32)
        self.X, self.Y = AsySVD_sgd(R, self.num_factors, self.lrate, self.reg, self.iters, self.init_mean,
                                    self.init_std,
                                    self.lrate_decay, self.rnd_seed)
        # precompute the user factors
        M = R.shape[0]
        self.U = np.vstack([AsySVD_compute_user_factors(R[i], self.Y) for i in range(M)])

    def recommend(self, user_id, n=None, exclude_seen=True):
        scores = np.dot(self.X, self.U[user_id].T)
        ranking = scores.argsort()[::-1]
        # rank items
        if exclude_seen:
            ranking = self._filter_seen(user_id, ranking)
        return ranking[:n]

    def predict(self, user_id, rated_indices):
        scores = np.dot(self.X, self.U[user_id].T)
        return scores[rated_indices]

    def label(self, unlabeled_list, binary_ratings=False, n=None, exclude_seen=True, p_most=1, n_most=3):
        # Shuffle the unlabeled list of tuples (user_idx, item_idx).
        np.random.shuffle(unlabeled_list)

        # TODO: Instead of just labeling p + n items, label p_most and n_most as the
        #       original algorithm says.
        labels = []
        number_labeled = 0
        for user_idx, item_idx in unlabeled_list:
            # compute the scores using the dot product
            scores = np.dot(self.X, self.U[user_idx].T)

            if ( (not(binary_ratings) and scores[item_idx] >= 1.0 and scores[item_idx] <= 5.0) \
                or \
                 (binary_ratings and scores[item_idx] >= 0.0 and scores[item_idx] <= 1.0) ):
                labels.append( (user_idx, item_idx, scores[item_idx]) )
                number_labeled += 1

            if (number_labeled == p_most + n_most):
                break

        return labels


class IALS_numpy(Recommender):
    '''
    binary Alternating Least Squares model (or Weighed Regularized Matrix Factorization)
    Reference: Collaborative Filtering for binary Feedback Datasets (Hu et al., 2008)

    Factorization model for binary feedback.
    First, splits the feedback matrix R as the element-wise a Preference matrix P and a Confidence matrix C.
    Then computes the decomposition of them into the dot product of two matrices X and Y of latent factors.
    X represent the user latent factors, Y the item latent factors.

    The model is learned by solving the following regularized Least-squares objective function with Stochastic Gradient Descent
    \operatornamewithlimits{argmin}\limits_{x*,y*}\frac{1}{2}\sum_{i,j}{c_{ij}(p_{ij}-x_i^T y_j) + \lambda(\sum_{i}{||x_i||^2} + \sum_{j}{||y_j||^2})}
    '''

    # TODO: Add support for multiple confidence scaling functions (e.g. linear and log scaling)
    def __init__(self,
                 num_factors=50,
                 reg=0.015,
                 iters=10,
                 scaling='linear',
                 alpha=40,
                 epsilon=1.0,
                 init_mean=0.0,
                 init_std=0.1,
                 rnd_seed=42):
        '''
        Initialize the model
        :param num_factors: number of latent factors
        :param reg: regularization term
        :param iters: number of iterations in training the model with SGD
        :param scaling: supported scaling modes for the observed values: 'linear' or 'log'
        :param alpha: scaling factor to compute confidence scores
        :param epsilon: epsilon used in log scaling only
        :param init_mean: mean used to initialize the latent factors
        :param init_std: standard deviation used to initialize the latent factors
        :param rnd_seed: random seed
        '''

        super(IALS_numpy, self).__init__()
        assert scaling in ['linear', 'log'], 'Unsupported scaling: {}'.format(scaling)

        self.num_factors = num_factors
        self.reg = reg
        self.iters = iters
        self.scaling = scaling
        self.alpha = alpha
        self.epsilon = epsilon
        self.init_mean = init_mean
        self.init_std = init_std
        self.rnd_seed = rnd_seed

    def __str__(self):
        return "WRMF-iALS(num_factors={},  reg={}, iters={}, scaling={}, alpha={}, episilon={}, init_mean={}, " \
               "init_std={}, rnd_seed={})".format(
            self.num_factors, self.reg, self.iters, self.scaling, self.alpha, self.epsilon, self.init_mean,
            self.init_std, self.rnd_seed
        )

    def _linear_scaling(self, R):
        C = R.copy().tocsr()
        C.data *= self.alpha
        C.data += 1.0
        return C

    def _log_scaling(self, R):
        C = R.copy().tocsr()
        C.data = 1.0 + self.alpha * np.log(1.0 + C.data / self.epsilon)
        return C

    def fit(self, R):
        self.dataset = R
        # compute the confidence matrix
        if self.scaling == 'linear':
            C = self._linear_scaling(R)
        else:
            C = self._log_scaling(R)

        Ct = C.T.tocsr()
        M, N = R.shape

        # set the seed
        np.random.seed(self.rnd_seed)

        # initialize the latent factors
        self.X = np.random.normal(self.init_mean, self.init_std, size=(M, self.num_factors))
        self.Y = np.random.normal(self.init_mean, self.init_std, size=(N, self.num_factors))

        for it in range(self.iters):
            self.X = self._lsq_solver_fast(C, self.X, self.Y, self.reg)
            self.Y = self._lsq_solver_fast(Ct, self.Y, self.X, self.reg)
            logger.debug('Finished iter {}'.format(it + 1))

    def recommend(self, user_id, n=None, exclude_seen=True):
        scores = np.dot(self.X[user_id], self.Y.T)
        ranking = scores.argsort()[::-1]
        # rank items
        if exclude_seen:
            ranking = self._filter_seen(user_id, ranking)
        return ranking[:n]

    def predict(self, user_id, rated_indices):
        scores = np.dot(self.X[user_id], self.Y.T)
        return scores[rated_indices]

    def label(self, unlabeled_list, binary_ratings=False, n=None, exclude_seen=True, p_most=1, n_most=3):
        # Shuffle the unlabeled list of tuples (user_idx, item_idx).
        np.random.shuffle(unlabeled_list)

        # TODO: Instead of just labeling p + n items, label p_most and n_most as the
        #       original algorithm says.
        labels = []
        number_labeled = 0
        for user_idx, item_idx in unlabeled_list:
            # compute the scores using the dot product
            scores = np.dot(self.X[user_idx], self.Y.T)

            # As IALS only works with binary ratings, we only bound for binary ratings.
            if (binary_ratings and scores[item_idx] >= 0.0 and scores[item_idx] <= 1.0):
                labels.append( (user_idx, item_idx, scores[item_idx]) )
                number_labeled += 1

            if (number_labeled == p_most + n_most):
                break

        return labels

    def _lsq_solver(self, C, X, Y, reg):
        # precompute YtY
        rows, factors = X.shape
        YtY = np.dot(Y.T, Y)

        for i in range(rows):
            # accumulate YtCiY + reg*I in A
            A = YtY + reg * np.eye(factors)

            # accumulate Yt*Ci*p(i) in b
            b = np.zeros(factors)

            for j, cij in self._nonzeros(C, i):
                vj = Y[j]
                A += (cij - 1.0) * np.outer(vj, vj)
                b += cij * vj

            X[i] = np.linalg.solve(A, b)
        return X

    def _lsq_solver_fast(self, C, X, Y, reg):
        # precompute YtY
        rows, factors = X.shape
        YtY = np.dot(Y.T, Y)

        for i in range(rows):
            # accumulate YtCiY + reg*I in A
            A = YtY + reg * np.eye(factors)

            start, end = C.indptr[i], C.indptr[i + 1]
            j = C.indices[start:end]  # indices of the non-zeros in Ci
            ci = C.data[start:end]  # non-zeros in Ci

            Yj = Y[j]  # only the factors with non-zero confidence
            # compute Yt(Ci-I)Y
            aux = np.dot(Yj.T, np.diag(ci - 1.0))
            A += np.dot(aux, Yj)
            # compute YtCi
            b = np.dot(Yj.T, ci)

            X[i] = np.linalg.solve(A, b)
        return X

    def _nonzeros(self, R, row):
        for i in range(R.indptr[row], R.indptr[row + 1]):
            yield (R.indices[i], R.data[i])


class BPRMF(Recommender):
    '''
    BPRMF model
    '''

    # TODO: add global effects
    def __init__(self,
                 num_factors=50,
                 lrate=0.01,
                 user_reg=0.015,
                 pos_reg=0.015,
                 neg_reg=0.0015,
                 iters=10,
                 sampling_type='user_uniform_item_uniform',
                 sample_with_replacement=True,
                 use_resampling=True,
                 sampling_pop_alpha=1.0,
                 init_mean=0.0,
                 init_std=0.1,
                 lrate_decay=1.0,
                 rnd_seed=42,
                 verbose=True):
        '''
        Initialize the model
        :param num_factors: number of latent factors
        :param lrate: initial learning rate used in SGD
        :param user_reg: regularization for the user factors
        :param pos_reg: regularization for the factors of the positive sampled items
        :param neg_reg: regularization for the factors of the negative sampled items
        :param iters: number of iterations in training the model with SGD
        :param sampling_type: type of sampling. Supported types are 'user_uniform_item_uniform' and 'user_uniform_item_pop'
        :param sample_with_replacement: `True` to sample positive items with replacement (doesn't work with 'user_uniform_item_pop')
        :param use_resampling: `True` to resample at each iteration during training
        :param sampling_pop_alpha: float smoothing factor for popularity based samplers (e.g., 'user_uniform_item_pop')
        :param init_mean: mean used to initialize the latent factors
        :param init_std: standard deviation used to initialize the latent factors
        :param lrate_decay: learning rate decay
        :param rnd_seed: random seed
        :param verbose: controls verbosity in output
        '''
        super(BPRMF, self).__init__()
        self.num_factors = num_factors
        self.lrate = lrate
        self.user_reg = user_reg
        self.pos_reg = pos_reg
        self.neg_reg = neg_reg
        self.iters = iters
        self.sampling_type = sampling_type
        self.sample_with_replacement = sample_with_replacement
        self.use_resampling = use_resampling
        self.sampling_pop_alpha = sampling_pop_alpha
        self.init_mean = init_mean
        self.init_std = init_std
        self.lrate_decay = lrate_decay
        self.rnd_seed = rnd_seed
        self.verbose = verbose

    def __str__(self):
        return "BPRMF(num_factors={}, lrate={}, user_reg={}. pos_reg={}, neg_reg={}, iters={}, " \
               "sampling_type={}, sample_with_replacement={}, use_resampling={}, sampling_pop_alpha={}, init_mean={}, " \
               "init_std={}, lrate_decay={}, rnd_seed={}, verbose={})".format(
            self.num_factors, self.lrate, self.user_reg, self.pos_reg, self.neg_reg, self.iters,
            self.sampling_type, self.sample_with_replacement, self.use_resampling, self.sampling_pop_alpha,
            self.init_mean,
            self.init_std,
            self.lrate_decay,
            self.rnd_seed,
            self.verbose
        )

    def fit(self, R):
        self.dataset = R
        R = check_matrix(R, 'csr', dtype=np.float32)
        self.X, self.Y = BPRMF_sgd(R,
                                   num_factors=self.num_factors,
                                   lrate=self.lrate,
                                   user_reg=self.user_reg,
                                   pos_reg=self.pos_reg,
                                   neg_reg=self.neg_reg,
                                   iters=self.iters,
                                   sampling_type=self.sampling_type,
                                   sample_with_replacement=self.sample_with_replacement,
                                   use_resampling=self.use_resampling,
                                   sampling_pop_alpha=self.sampling_pop_alpha,
                                   init_mean=self.init_mean,
                                   init_std=self.init_std,
                                   lrate_decay=self.lrate_decay,
                                   rnd_seed=self.rnd_seed,
                                   verbose=self.verbose)

    def recommend(self, user_id, n=None, exclude_seen=True):
        scores = np.dot(self.X[user_id], self.Y.T)
        ranking = scores.argsort()[::-1]
        # rank items
        if exclude_seen:
            ranking = self._filter_seen(user_id, ranking)
        return ranking[:n]

    def predict(self, user_id, rated_indices):
        scores = np.dot(self.X[user_id], self.Y.T)
        return scores[rated_indices]

    def label(self, unlabeled_list, binary_ratings=False, n=None, exclude_seen=True, p_most=1, n_most=3):
        # Shuffle the unlabeled list of tuples (user_idx, item_idx).
        np.random.shuffle(unlabeled_list)

        # TODO: Instead of just labeling p + n items, label p_most and n_most as the
        #       original algorithm says.
        labels = []
        number_labeled = 0
        for user_idx, item_idx in unlabeled_list:
            # compute the scores using the dot product
            scores = np.dot(self.X[user_idx], self.Y.T)

            if (scores[item_idx] != 0.0):
                labels.append( (user_idx, item_idx, scores[item_idx]) )
                number_labeled += 1

            if (number_labeled == p_most + n_most):
                break

        return labels
