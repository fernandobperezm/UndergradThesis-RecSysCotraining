'''
Politecnico di Milano.
cotraining.py

Description: This file contains the definition and implementation of a
             Co-Training environment to train, boost and evaluate different recommenders

Created by: Fernando Benjamín Pérez Maurera.

Last modified on 05/09/2017.
'''

import random as random
import logging
import traceback
from datetime import datetime as dt

import numpy as np
import scipy.sparse as sp

import sys

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

class CoTraining(object):
    """Class that implements a Co-Training process between two recommenders.

        This implementation of Co-Training considers the Training of 2 different
        recommenders, each recommender has its own training set.

        In the first training round, both share the same Training set.
        Each training round performs sequentially the following steps:
            0. Creation of a random pool of unrated user-item pairs inside
               both training sets.
            1. Each recommender its trained with its respective training set.
            2. The recommenders are evaluated.
            3. The recommenders performs the labeling of new user-item pairs.
            4. The new rated items of Recommender 1 are added into the training
                set of Recommender 2.
            5. The new rated items of Recommender 2 are added into the training
                set of Recommender 1.

        There are several helpers in order to understand the behavior of Co-Training,
        for example, baseline recommenders as GlobalEffects, TopPopular and Random,
        can be trained at each iteration. We also do some statistic operations
        in order to know the number of common elements both rate as positive, and
        so on. Each of these operations are explained in detail.

        See:
            Combining labeled and unlabeled data with co-training,
            Avrim Blum and Tom Mitchell, Proceedings of the eleventh annual
            conference on Computational learning theory, COLT' 98,
            http://dl.acm.org/citation.cfm?id=279962

            PAC Generalization Bounds for Co-training,
            Sanjoy Dasgupta, Michael L. Littman and David McAllester,
            Proceedings of the 14th International Conference on Neural Information
            Processing Systems: Natural and Synthetic, NIPS' 01,
            http://dl.acm.org/citation.cfm?id=2980589


            Analyzing Co-training Style Algorithms
            Wei Wang and Zhi-Hua Zhou, Proceedings of the 18th European
            conference on Machine Learning, ECML '07,
            http://dl.acm.org/citation.cfm?id=1421709


        Attributes:
            * rec_1: first recommender to be trained.
            * rec_2: second recommender to be trained.
            * eval_obj: evaluation metrics.
            * n_iters: Represents the number of Co-Training iterations.
            * n_labels: size of the unlabeled sample pool.
            * p_most: number of p-most positive samples to label.
            * n_most: number of n-most positive samples to label.
            * seed: seed for random number generators.
    """

    def __init__(self, rec_1, rec_2, eval_obj, n_iters = 30, n_labels = 10, p_most = 1, n_most = 3, seed=1024):
        """Constructor of the class.

            Args:
                * rec_1: first recommender to be trained.
                * rec_2: second recommender to be trained.
                * eval_obj: evaluation metrics.
                * n_iters: Represents the number of Co-Training iterations.
                * n_labels: size of the unlabeled sample pool.
                * p_most: number of p-most positive samples to label.
                * n_most: number of n-most positive samples to label.
                * seed: seed for random number generators.

            Args type:
                * rec_1: A Recommender instance
                * rec_2: A Recommender instance
                * eval_obj: An Evaluation instance
                * n_iters: int
                * n_labels: int
                * p_most: int
                * n_most: int
                * seed: int

        """
        super(CoTraining, self).__init__()
        self.rec_1 = rec_1
        self.rec_2 = rec_2
        self.eval = eval_obj
        self.n_iters = n_iters
        self.n_labels = n_labels
        self.p_most = p_most
        self.n_most = n_most
        self.seed = seed

    def short_str(self):
        """ Short string used for dictionaries. """
        return "CoTraining(Rec1={},Rec2={},Iter={})".format(
            self.rec_1.short_str(),self.rec_2.short_str(),self.n_iters)

    def __str__(self):
        """ String representation of the class. """
        return "CoTrainingEnv(Rec1={},Rec2={},Iterations={})".format(
            self.rec_1.__str__(), self.rec_2.__str__(), self.n_iters)

    def fit(self,
            URM_1,
            eval_iter=False,
            binary_ratings=False,
            recommenders=None,
            baselines=False,
            recover_cotraining=False,
            recover_iter=0):
        """Trains two recommenders following the Co-Training process.

            The fitting function supposes that two recommenders are going to
            be trained using two different training sets.

            Args:
                * URM_1: Describes the training set for Recommender 1.
                * eval_iter: evaluate the recommenders at each iteration.
                * binary_ratings: the dataset is implicit or explicit.
                * recommenders: holds all the recommenders, Recommender 1,
                                Recommender 2 and the baselines.
                * baselines: tells if there are baselines.
                * recover_cotraining: resume the Co-Training process or begin
                                      a new one.
                * recover_iter: Resume the Co-Training process at a given iteration.

            Args type:
                * URM_1: Scipy.Sparse matrix
                * eval_iter: bool
                * binary_ratings: bool
                * recommenders: Dictionary<Recommender>
                * baselines: bool
                * recover_cotraining: bool
                * recover_iter: int

        """
        # parameters initialization.
        nusers, nitems = URM_1.shape
        random_state = np.random.RandomState(self.seed)
        error_path = self.eval.results_path + "errors.txt"
        error_file = open(error_path, 'w')

        # Get the baselines instances in order to eval them.
        if (recommenders is not None and baselines == True):
            ge_1 = recommenders['GlobalEffects1']
            ge_2 = recommenders['GlobalEffects2']
            tp_1 = recommenders['TopPop1']
            tp_2 = recommenders['TopPop2']
            random = recommenders['Random']

        # If we must resume Co-Training, then the training sets must be loaded
        if (recover_cotraining):
            begin_iter = recover_iter
            URM_1 = sp.load_npz(file=self.eval.results_path + 'training_set_1_iter{}.npz'.format(recover_iter)).tolil()
            URM_2 = sp.load_npz(file=self.eval.results_path + 'training_set_2_iter{}.npz'.format(recover_iter)).tolil()
        else:
            begin_iter = 0
            URM_2 = URM_1.copy()

        # Co-Training iterations begin here.
        for i_iter in range(begin_iter,self.n_iters+1):
            logger.info("Iteration: {}".format(i_iter))

            u_prime = self.generate_unlabeled_pool(URM_1=URM_1,
                                                   URM_2=URM_2,
                                                   nusers=nusers,
                                                   nitems=nitems,
                                                   random_state=random_state
                                                  )

            if (i_iter % 10 == 0):
                # Backup the dataset at each 10 iters.
                sp.save_npz(file=self.eval.results_path + 'training_set_1_iter{}.npz'.format(i_iter),matrix=URM_1.tocoo(), compressed=True)
                sp.save_npz(file=self.eval.results_path + 'training_set_2_iter{}.npz'.format(i_iter),matrix=URM_2.tocoo(), compressed=True)

            # Try to fit the first recommender.
            try:
                logger.info('\tRecommender: {}'.format(self.rec_1))
                tic = dt.now()
                logger.info('\t\tTraining started for recommender: {}'.format(self.rec_1))
                self.rec_1.fit(URM_1)
                if (self.rec_1.short_str() == "SLIM_BPR_Mono"):
                    print(self.rec_1.evaluateRecommendations(URM_test_new=self.eval.test_set, at=self.eval.at, minRatingsPerUser=1, exclude_seen=True,mode='sequential', filterTopPop = False,fastValidation=True))
                logger.info('\t\tTraining completed in {} for recommender: {}'.format(dt.now() - tic, self.rec_1))
            except:
                logger.info('Could not fit the recommender 1: {}'.format(sys.exc_info()))
                traceback.print_exc(file=error_file)

            # Try to fit the second recommender.
            try:
                logger.info('\tRecommender: {}'.format(self.rec_2))
                tic = dt.now()
                logger.info('\t\tTraining started for recommender: {}'.format(self.rec_2))
                self.rec_2.fit(URM_2)
                if (self.rec_2.short_str() == "SLIM_BPR_Mono"):
                    print(self.rec_2.evaluateRecommendations(URM_test_new=self.eval.test_set, at=self.eval.at, minRatingsPerUser=1, exclude_seen=True,mode='sequential', filterTopPop = False,fastValidation=True))
                logger.info('\t\tTraining completed in {} for recommender: {}'.format(dt.now() - tic, self.rec_2))
            except:
                logger.info('Could not fit the recommender 2: {}'.format(sys.exc_info()))
                traceback.print_exc(file=error_file)

            # If evaluating with baselines, then train them.
            if (baselines):
                try:
                    logger.info('\tRecommender: {}'.format(ge_1))
                    tic = dt.now()
                    logger.info('\t\tTraining started for recommender: {}'.format(ge_1))
                    ge_1.fit(URM_1)
                    logger.info('\t\tTraining completed in {} for recommender: {}'.format(dt.now() - tic, ge_1))
                except:
                    logger.info('Could not fit the recommender global effects: {}'.format(sys.exc_info()))
                    traceback.print_exc(file=error_file)

                try:
                    logger.info('\tRecommender: {}'.format(ge_2))
                    tic = dt.now()
                    logger.info('\t\tTraining started for recommender: {}'.format(ge_2))
                    ge_2.fit(URM_2)
                    logger.info('\t\tTraining completed in {} for recommender: {}'.format(dt.now() - tic, ge_2))
                except:
                    logger.info('Could not fit the recommender global effects: {}'.format(sys.exc_info()))
                    traceback.print_exc(file=error_file)

                try:
                    logger.info('\tRecommender: {}'.format(tp_1))
                    tic = dt.now()
                    logger.info('\t\tTraining started for recommender: {}'.format(tp_1))
                    tp_1.fit(URM_1)
                    logger.info('\t\tTraining completed in {} for recommender: {}'.format(dt.now() - tic, tp_1))
                except:
                    logger.info('Could not fit the recommender top-pop: {}'.format(sys.exc_info()))
                    traceback.print_exc(file=error_file)

                try:
                    logger.info('\tRecommender: {}'.format(tp_2))
                    tic = dt.now()
                    logger.info('\t\tTraining started for recommender: {}'.format(tp_2))
                    tp_2.fit(URM_2)
                    logger.info('\t\tTraining completed in {} for recommender: {}'.format(dt.now() - tic, tp_2))
                except:
                    logger.info('Could not fit the recommender top-pop: {}'.format(sys.exc_info()))
                    traceback.print_exc(file=error_file)

                try:
                    logger.info('\tRecommender: {}'.format(random))
                    tic = dt.now()
                    logger.info('\t\tTraining started for recommender: {}'.format(random))
                    random.fit(URM_1)
                    logger.info('\t\tTraining completed in {} for recommender: {}'.format(dt.now() - tic, random))
                except:
                    logger.info('Could not fit the recommender random: {}'.format(sys.exc_info()))
                    traceback.print_exc(file=error_file)

            # Evaluate the recommenders in this iteration.
            logger.info('\tEvaluating both recommenders.')
            try:
                self.eval.eval(recommenders=recommenders, minRatingsPerUser=1)
                self.eval.log_to_file(
                                      log_type="evaluation",
                                      recommenders=recommenders,
                                      args=
                                        {'index':i_iter
                                        },
                                      )
            except:
                logger.info('Could not evaluate both recomemnders: {}'.format(sys.exc_info()))
                traceback.print_exc(file=error_file)

            # Label positively and negatively examples from U' for both recommenders.
            logger.info('\tLabeling new items.')
            labeled1, labeled2, meta_1, meta_2, meta_both = self.label(unlabeled_set=u_prime, binary_ratings=binary_ratings, exclude_seen=True, p_most=self.p_most, n_most=self.n_most, error_file=error_file)

            try:
                self.eval.log_to_file(
                                      log_type="labeling",
                                      recommenders=
                                        {self.rec_1.short_str():self.rec_1,
                                         self.rec_2.short_str():self.rec_2
                                        },
                                      args=
                                        {'index':i_iter,
                                         'both_pos': meta_both['both_pos'],
                                         'both_neg': meta_both['both_neg'],
                                         'both_neutral': meta_both['both_neutral'],
                                         'pos_only_first': meta_both['pos_only_first'],
                                         'neg_only_first': meta_both['neg_only_first'],
                                         'neutral_only_first': meta_both['neutral_only_first'],
                                         'pos_only_second': meta_both['pos_only_second'],
                                         'neg_only_second': meta_both['neg_only_second'],
                                         'neutral_only_second': meta_both['neutral_only_second'],
                                         self.rec_1.short_str():(meta_1['pos_labels'], meta_1['neg_labels'], meta_1['total_labels']),
                                         self.rec_2.short_str():(meta_2['pos_labels'], meta_2['neg_labels'], meta_2['total_labels']),
                                        }
                                      )
            except:
                logger.info('Could not log the new labeled items: {}'.format(sys.exc_info()))
                traceback.print_exc(file=error_file)

            # Add the labeled examples from recommender1 into T2.
            try:
                for user_idx, item_idx, label in labeled1:
                    URM_2[user_idx,item_idx] = label
            except:
                logger.info('Could not include labeled into URM_2: {}'.format(sys.exc_info()))
                traceback.print_exc(file=error_file)

            # Add the labeled examples from recommender2 into T1.
            try:
                for user_idx, item_idx, label in labeled2:
                    URM_1[user_idx,item_idx] = label
            except:
                logger.info('Could not include labeled into URM_1: {}'.format(sys.exc_info()))
                traceback.print_exc(file=error_file)

        error_file.close()

    def label(self, unlabeled_set, binary_ratings=False, exclude_seen=True, p_most=1000, n_most=100000, error_file=None):
        """Rates new user-item pairs.

           This function is part of the Co-Training process in which we rate
           all user-item pairs inside an unlabeled pool of samples for both
           recommenders.

           The `label` method of each recommender is called.

           After those methods returns, we apply set operations using the
           dictionary of meta statistics they return. Specifically, we determine
           how many items both label as positive, negative and neutral, also,
           how many items only the first recommender label as positive, negative
           and neutral, lastly, how many items only the second recommender label
           as positive, negative and neutral

           Args:
               * unlabeled_set: a matrix that holds the user-item that we must
                                 predict their rating.
               * binary_ratings: tells us if we must predict based on an implicit
                                 (0,1) dataset or an explicit.
               * exclude_seen: tells us if we need to exclude already-seen items.
               * p_most: tells the number of p-most positive items that we
                         should choose.
               * n_most: tells the number of n-most negative items that we
                         should choose.
               * error_file: if the algorithms cannot label, the exception is
                             logged into a file.

           Args type:
               * unlabeled_set: Scipy.Sparse matrix.
               * binary_ratings: bool
               * exclude_seen: bool
               * p_most: int
               * n_most: int
               * error_file: str

           Returns:
               A list containing the user-item-rating triplets of both recommenders,
                the meta dictionary for statistics of both recommenders and
                the meta dictionary statistic for both.
        """

        try:
            labeled1, meta_1 = self.rec_1.label(unlabeled_list=unlabeled_set, binary_ratings=binary_ratings, exclude_seen=True, p_most=self.p_most, n_most=self.n_most)
        except:
            logger.info('Could not label new items for recomemnder 1: {}'.format(sys.exc_info()))
            traceback.print_exc(file=error_file)
        try:
            labeled2, meta_2 = self.rec_2.label(unlabeled_list=unlabeled_set, binary_ratings=binary_ratings, exclude_seen=True, p_most=self.p_most, n_most=self.n_most)
        except:
            logger.info('Could not label new items for recomemnder 2: {}'.format(sys.exc_info()))
            traceback.print_exc(file=error_file)

        meta_both = dict()
        meta_both['both_pos'] = len(meta_1['pos_set'].intersection(meta_2['pos_set']))
        meta_both['both_neg'] = len(meta_1['neg_set'].intersection(meta_2['neg_set']))
        meta_both['both_neutral'] = len(meta_1['neutral_set'].intersection(meta_2['neutral_set']))
        meta_both['pos_only_first'] = len(meta_1['pos_set'].difference(meta_2['pos_set']))
        meta_both['neg_only_first'] = len(meta_1['neg_set'].difference(meta_2['neg_set']))
        meta_both['neutral_only_first'] = len(meta_1['neutral_set'].difference(meta_2['neutral_set']))
        meta_both['pos_only_second'] = len(meta_2['pos_set'].difference(meta_1['pos_set']))
        meta_both['neg_only_second'] = len(meta_2['neg_set'].difference(meta_1['neg_set']))
        meta_both['neutral_only_second'] = len(meta_2['neutral_set'].difference(meta_1['neutral_set']))


        return labeled1, labeled2, meta_1, meta_2, meta_both

    def generate_unlabeled_pool(self, URM_1, URM_2, nusers, nitems, random_state):
        """Generates random user-item pairs that are not labeled in both training sets.

           The function generates a random user and a random item and if the pair
           is not rated neither in URM_1 nor URM_2, then it is considered as
           an unlabeled sample.

           All the users and items are drawn from an Uniform Distribution.

           Args:
               * URM_1: training set of the first recommender.
               * URM_2: training set of the second recommender.
               * nusers: number of users in the dataset.
               * nitems: number of items in the dataset.
               * random_state: random number generator.

           Args type:
               * URM_1: Scipy.Sparse matrix
               * URM_2: Scipy.Sparse matrix
               * nusers: int
               * nitems: int
               * random_state: Numpy.Random.RandomState instance

           Returns:
               A Scipy.Sparse.LilMatrix instance where the nonzero elements
               are the random users and items.
        """
        # Using a LIL matrix to have a A[row[k], col[k]] = Data[k] representation
        # without having an efficiency tradeoff.
        logger.info("Creating a pool of unlabeled samples.")
        u_prime = sp.lil_matrix((nusers,nitems), dtype=np.int32)

        # Feed U' with unlabeled samples.
        users_items = set()
        while (len(users_items) < self.n_labels):
            rnd_user = random_state.randint(0, high=nusers, dtype=np.int32)
            rnd_item = random_state.randint(0, high=nitems, dtype=np.int32)
            if (URM_1[rnd_user,rnd_item] == 0.0 and URM_2[rnd_user,rnd_item] == 0.0 ): # TODO: user better precision (machine epsilon instead of == 0.0)
                users_items.add((rnd_user,rnd_item))

        # As LIL matrices works better if the (user,item) pairs are sorted
        # first by user and then by item.
        for user,item in sorted(users_items, key=lambda u_i: (u_i[0], u_i[1])):
            u_prime[user,item] = 1

        logger.info("Pool created. Its size is: {}.".format(u_prime.nnz))

        return u_prime
