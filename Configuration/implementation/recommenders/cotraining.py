'''
Politecnico di Milano.
co-training.py

Description: This file contains the definition and implementation of a
             Co-Training environment to train, boost and evaluate different recommenders

Modified by Fernando Pérez.

Last modified on 25/03/2017.
'''

import random as random

import numpy as np
import scipy.sparse as sps
from .base import Recommender, check_matrix
from .similarity import Cosine, Pearson, AdjustedCosine

import pdb


class COTRAINING(object):
    """ CO-TRAINING environment for RecSys"""

    def __init__(self, rec_1, rec_2, eval_obj1, eval_obj2, eval_obj_aggr, n_iters = 30, n_labels = 10, p_most = 1, n_most = 3):
        '''
            Args:
                * rec_1: A Recommender Class object that represents the first
                         recommender.
                * rec_2: A Recommender Class object that represents the second
                         recommender.
                * eval_obj1: An Evaluation Class object that represents the evaluation
                            metrics for the recommender1.
                * eval_obj2: An Evaluation Class object that represents the evaluation
                            metrics for the recommender2.
                * eval_obj_aggr: An Evaluation Class object that represents the evaluation
                            metrics for the aggregate of both recommenders.
                * n_iters: Represents the number of Co-Training iterations.
                * n_labels: The number of elements to label in each Co-Training
                            iteration.
        '''
        super(COTRAINING, self).__init__()
        self.rec_1 = rec_1
        self.rec_2 = rec_2
        self.eval1 = eval_obj1
        self.eval2 = eval_obj2
        self.eval_aggr = eval_obj_aggr
        self.n_iters = n_iters
        self.n_labels = n_labels
        self.p_most = p_most
        self.n_most = n_most

    def __str__(self):
        return "CoTrainingEnv(Rec1={}\n,Rec2={}\n,N Iterations={})".format(
            self.rec1.__str__, self.rec2.__str__, self.n_iters)

    def fit(self, X1, eval_iter = False, number_unlabeled = 75):
        '''
            Depending of the Co-Training approach, you can have two views or
            use different learners, in the case of X2 == None, it's supposed
            that the same dataset will be used,

            Args:
                X_unlabeled: Set of X_unlabeled user/item pairs that doesn't have
                             a rating. It must be a DOK matrix.
                X1: Represents the training dataset to be used for the first
                    recommender. It must be a DOK matrix.
                X2: Represents the training dataset to be used for the second
                    recommender. It must be a DOK matrix.
                eval_iter: Tells if we need to evaluate each recommender and the
                           joint at each iteration.

        '''
        nusers, nitems = X1.shape

        # Create the pool of examples.
        # Using a DoK matrix to have a A[row[k], col[k]] = Data[k] representation
        # without having an efficiency tradeoff.
        u_prime = sp.sparse.dok_matrix((nusers,nitems), dtype=np.int32)

        # Feed U' with unlabeled samples.
        i = 0
        while (i < number_unlabeled):
            rnd_user = np.random.randint(0, high=nusers, dtype='l')
            rnd_item = np.random.randint(0, high=nitems, dtype='l')
            if (X1[rnd_user, rnd_item] == 0.0): # TODO: user better precision (machine epsilon instead of == 0.0)
                u_prime[rnd_user, rnd_item] = 1
                i += 1

        # Training set for Rec2.
        X2 = X1.copy()

        # Co-Training iterations begin here.
        for i_iter in self.n_iters:
            logger.info(("Iteration: {}".format(i_iter)))

            # logger.info('\tRecommender: {}'.format(self.rec_1))
            # tic = dt.now()
            # logger.info('\t\tTraining started for recommender: {}'.format(self.rec_1))
            self.rec_1.fit(X1)
            # logger.info('\t\tTraining completed in {} for recommender: {}'.format(dt.now() - tic, self.rec_1))

            # logger.info('\tRecommender: {}'.format(self.rec_2))
            # tic = dt.now()
            # logger.info('\t\tTraining started for recommender: {}'.format(self.rec_2))
            self.rec_2.fit(X2)
            # logger.info('\t\tTraining completed in {} for recommender: {}'.format(dt.now() - tic, self.rec_2))

            # Label positively and negatively examples from U' for both recommenders.
            unlabeled = u_prime.keys()
            # TODO: Make ALL recommender to have the member function label which must return
            #       a list of Triplets (user_idx, item_idx, predicted label)
            labeled1 = self.rec_1.label(X_unlabeled_list=list(unlabeled), binary_ratings=args.is_binary, exclude_seen=True, p_most=self.p_most, n_most=self.n_most)
            labeled2 = self.rec_2.label(X_unlabeled_list=list(unlabeled), binary_ratings=args.is_binary, exclude_seen=True, p_most=self.p_most, n_most=self.n_most)

            # Add the labeled examples from recommender1 into T2. (and eliminate them from U' as they aren't X_unlabeled anymore).
            for user_idx, item_idx, label in labeled1:
                X2[user_idx, item_idx] = label
                u_prime[user_idx, item_idx] = 0

            # Add the labeled examples from recommender2 into T1. (and eliminate them from U' as they aren't X_unlabeled anymore).
            for user_idx, item_idx, label in labeled2:
                X1[user_idx, item_idx] = label
                u_prime[user_idx, item_idx] = 0

            # Replenish U' with 2*p + 2*n samples from U.
            i = 0
            while (i < (2*self.p_most + 2*self.n_most) ):
                rnd_user = np.random.randint(0, high=nusers, dtype='l')
                rnd_item = np.random.randint(0, high=nitems, dtype='l')
                if (X1[rnd_user, rnd_item] == 0.0 and X2[rnd_user, rnd_item] == 0.0): # TODO: user better precision (machine epsilon instead of == 0.0)
                    u_prime[rnd_user, rnd_item] = 1
                    i += 1

            # Evaluate the recommender in this iteration.
            pass

    def recommend(self, user_id, n=None, exclude_seen=True):
        # compute the scores using the dot product
        user_profile1 = self.rec_1._get_user_ratings(user_id)
        user_profile2 = self.rec_2._get_user_ratings(user_id)

        if self.sparse_weights:
            scores1 = user_profile1.dot(self.rec_1.W_sparse).toarray().ravel()
            scores2 = user_profile2.dot(self.rec_2.W_sparse).toarray().ravel()
        else:
            scores1 = user_profile1.dot(self.rec_1.W).ravel()
            scores2 = user_profile2.dot(self.rec_2.W).ravel()

        if self.rec_1.normalize:
            # normalization will keep the scores in the same range
            # of value of the ratings in dataset
            rated1 = user_profile1.copy()
            rated1.data = np.ones_like(rated1.data)
            if self.rec_1.sparse_weights:
                den1 = rated.dot(self.rec_1.W_sparse).toarray().ravel()
            else:
                den1 = rated.dot(self.rec_1.W).ravel()
            den1[np.abs(den1) < 1e-6] = 1.0  # to avoid NaNs
            scores1 /= den1

        if self.rec_2.normalize:
            # normalization will keep the scores in the same range
            # of value of the ratings in dataset
            rated2 = user_profile2.copy()
            rated2.data = np.ones_like(rated2.data)
            if self.rec_2.sparse_weights:
                den2 = rated.dot(self.rec_2.W_sparse).toarray().ravel()
            else:
                den2 = rated.dot(self.rec_2.W).ravel()
            den2[np.abs(den2) < 1e-6] = 1.0  # to avoid NaNs
            scores2 /= den2

        # Creating the score by averaging the scores.
        scores = (scores1 + scores2) / 2
        ranking = scores.argsort()[::-1]

        if exclude_seen:
            ranking = self.rec_1._filter_seen(user_id, ranking)
        return ranking[:n]


    def random_user_sample_random_item_sample(self, dataset, n_samples = 75):
        '''
            Returns a set of randomly choosen user/item pairs which doesn't have
            rating.
        '''
        nusers, nitems = dataset.shape
        i_sample = 0
        random_pop = set()
        while (i_sample < n_samples):
            rnd_user = np.random.randint(0, high=nusers, dtype='l')
            rnd_item = np.random.randint(0, high=nitems, dtype='l')
            if (dataset[rnd_user, rnd_item] == 0.0): # TODO: user better precision (machine epsilon instead of == 0.0)
                random_pop.add( (rnd_user, rnd_item) )
                i_sample +=1
        return random_pop
