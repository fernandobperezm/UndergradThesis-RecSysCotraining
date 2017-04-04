# -*- coding: utf-8 -*-
'''
Politecnico di Milano.
k-fold-cotraining.py

Description: This file contains the implementation of the co-training algorithm
             as stated in:
             Blum, A., & Mitchell, T. (1998, July). Combining labeled and
             unlabeled data with co-training. In Proceedings of the eleventh
             annual conference on Computational learning theory (pp. 92-100).
             ACM.
             Available here: http://repository.cmu.edu/cgi/viewcontent.cgi?article=1181&context=compsci
             Using k-fold cross-validaton.

Modified by Fernando Pérez.

Last modified on 25/03/2017.
'''

# Import Python utils.
import argparse
import logging
from collections import OrderedDict
from datetime import datetime as dt
import pdb

# Numpy and scipy.
import numpy as np
import scipy as sp

# Import utils such as
from implementation.utils.data_utils import read_dataset, df_to_csr
from implementation.utils.split import k_fold_cv
from implementation.utils.metrics import roc_auc, precision, recall, map, ndcg, rr

# Import recommenders classes.
from implementation.recommenders.item_knn import ItemKNNRecommender
from implementation.recommenders.user_knn import UserKNNRecommender
from implementation.recommenders.slim import SLIM, MultiThreadSLIM
from implementation.recommenders.mf import FunkSVD, IALS_numpy, AsySVD, BPRMF
from implementation.recommenders.non_personalized import TopPop, GlobalEffects

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

available_recommenders = OrderedDict([
    ('top_pop', TopPop),
    ('global_effects', GlobalEffects),
    ('item_knn', ItemKNNRecommender),
    ('user_knn', UserKNNRecommender),
    ('SLIM', SLIM),
    ('SLIM_mt', MultiThreadSLIM),
    ('FunkSVD', FunkSVD),
    ('AsySVD', AsySVD),
    ('IALS_np', IALS_numpy),
    ('BPRMF', BPRMF),
])

# let's use an ArgumentParser to read input arguments
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('--is_binary', action='store_true', default=False)
parser.add_argument('--make_binary', action='store_true', default=False)
parser.add_argument('--binary_th', type=float, default=4.0)
parser.add_argument('--holdout_perc', type=float, default=0.8)
parser.add_argument('--header', type=int, default=None)
parser.add_argument('--columns', type=str, default=None)
parser.add_argument('--sep', type=str, default=',')
parser.add_argument('--user_key', type=str, default='user_id')
parser.add_argument('--item_key', type=str, default='item_id')
parser.add_argument('--rating_key', type=str, default='rating')
parser.add_argument('--rnd_seed', type=int, default=1234)
parser.add_argument('--recommender_1', type=str, default='top_pop')
parser.add_argument('--params_1', type=str, default=None)
parser.add_argument('--recommender_2', type=str, default='top_pop')
parser.add_argument('--params_2', type=str, default=None)
parser.add_argument('--rec_length', type=int, default=10)
parser.add_argument('--k_fold', type=int, default=5)
parser.add_argument('--number_iterations', type=int, default=30)
parser.add_argument('--number_positives', type=int, default=1)
parser.add_argument('--number_negatives', type=int, default=3)
parser.add_argument('--number_unlabeled', type=int, default=75)
args = parser.parse_args()

# get the recommender class
assert args.recommender_1 in available_recommenders, 'Unknown recommender: {}'.format(args.recommender_1)
assert args.recommender_2 in available_recommenders, 'Unknown recommender: {}'.format(args.recommender_2)
RecommenderClass_1 = available_recommenders[args.recommender_1]
RecommenderClass_2 = available_recommenders[args.recommender_2]

# parse recommender parameters
init_args_recomm_1 = OrderedDict()
if args.params_1:
    for p_str in args.params_1.split(','):
        key, value = p_str.split('=')
        try:
            init_args_recomm_1[key] = eval(value)
        except:
            init_args_recomm_1[key] = value

init_args_recomm_2 = OrderedDict()
if args.params_2:
    for p_str in args.params_2.split(','):
        key, value = p_str.split('=')
        try:
            init_args_recomm_2[key] = eval(value)
        except:
            init_args_recomm_2[key] = value

# convert the column argument to list
if args.columns is not None:
    args.columns = args.columns.split(',')

# read the dataset
logger.info('Reading {}'.format(args.dataset))
dataset, item_to_idx, user_to_idx = read_dataset(
    args.dataset,
    header=args.header,
    sep=args.sep,
    columns=args.columns,
    make_binary=args.make_binary,
    binary_th=args.binary_th,
    item_key=args.item_key,
    user_key=args.user_key,
    rating_key=args.rating_key)

nusers, nitems = dataset.user_idx.max() + 1, dataset.item_idx.max() + 1
logger.info('The dataset has {} users and {} items'.format(nusers, nitems))

# compute the k-fold split
logger.info('Computing the {:.0f}% {}-fold cross-validation split'.format(args.holdout_perc * 100, args.k_fold))
roc_auc_, precision_, recall_, map_, mrr_, ndcg_ = np.zeros(args.k_fold), np.zeros(args.k_fold), np.zeros(
    args.k_fold), np.zeros(args.k_fold), np.zeros(args.k_fold), np.zeros(args.k_fold)

roc_auc_2, precision_2, recall_2, map_2, mrr_2, ndcg_2 = np.zeros(args.k_fold), np.zeros(args.k_fold), np.zeros(
    args.k_fold), np.zeros(args.k_fold), np.zeros(args.k_fold), np.zeros(args.k_fold)
i_fold = 0
# TODO: Partition the dataset in Train, Validation and Testing for k-fold cross
#       validation as stated by Professor Restelli Marcello.
for train_df, test_df in k_fold_cv(dataset,
                              user_key=args.user_key,
                              item_key=args.item_key,
                              k=args.k_fold,
                              seed=args.rnd_seed,
                              clean_test=True):
    logger.info(("Fold: {}".format(i_fold)))

    # TODO: Modularize Co-training.
    '''
        Co-Training begins here.
        As the Co-Training algorith states:
        L: a set of labeled samples.
        U: a set of unlabeled samples.
        L union U === train: a set of labeled and unlabeled samples.
        U' === u_prime: pool of examples.
        u === args.number_unlabeled number of samples to take from U when creating U'.
        k === args.number_iterations number of Co-Training iterations
        h1 === h1: first classifier/regressor.
        h2 === h2: second classifier/regressor.
        p === number_positives: number of positives examples to label from U'
        n === number_negatives: number of negative examples to label from U'
    '''
    # Create our label and unlabeled samples set.
    train = df_to_csr(train_df,
                      is_binary=args.is_binary,
                      nrows=nusers,
                      ncols=nitems,
                      item_key='item_idx',
                      user_key='user_idx',
                      rating_key=args.rating_key)

    # Create our test set.
    test = df_to_csr(test_df,
                     is_binary=args.is_binary,
                     nrows=nusers,
                     ncols=nitems,
                     item_key='item_idx',
                     user_key='user_idx',
                     rating_key=args.rating_key)

    # Create the pool of examples.
    # Using a DoK matrix to have a A[row[k], col[k]] = Data[k] representation
    # without having an efficiency tradeoff.
    u_prime = sp.sparse.dok_matrix((nusers,nitems), dtype=np.int32)

    # Feed U' with unlabeled samples.
    i = 0
    while (i < args.number_unlabeled):
        rnd_user = np.random.randint(0, high=nusers, dtype='l')
        rnd_item = np.random.randint(0, high=nitems, dtype='l')
        if (train[rnd_user, rnd_item] == 0.0): # TODO: user better precision (machine epsilon instead of == 0.0)
            u_prime[rnd_user, rnd_item] = 1
            i += 1

    # Co-Training iterations begin here.
    for i_iter in range(args.number_iterations):
        logger.info(("\tIteration: {}".format(i_iter)))
        # train the recommenders
        h1 = RecommenderClass_1(**init_args_recomm_1)
        h2 = RecommenderClass_2(**init_args_recomm_2)

        logger.info('\t\tRecommender: {}'.format(h1))
        tic = dt.now()
        logger.info('\t\t\tTraining started for recommender: {}'.format(h1))
        h1.fit(train)
        logger.info('\t\t\tTraining completed in {} for recommender: {}'.format(dt.now() - tic, h1))

        logger.info('\t\tRecommender: {}'.format(h2))
        tic = dt.now()
        logger.info('\t\t\tTraining started for recommender: {}'.format(h2))
        h2.fit(train)
        logger.info('\t\t\tTraining completed in {} for recommender: {}'.format(dt.now() - tic, h2))

        # Label positively and negatively examples from U' for both recommenders.
        unlabeled = u_prime.keys()
        # TODO: Make ALL recommender to have the member function label which must return
        #       a list of Triplets (user_idx, item_idx, predicted label)
        labeled1 = h1.label(unlabeled_list=list(unlabeled), exclude_seen=True, p_most=args.number_positives, n_most=args.number_negatives)
        labeled2 = h2.label(unlabeled_list=list(unlabeled), exclude_seen=True, p_most=args.number_positives, n_most=args.number_negatives)

        # Add the labeled examples into L (and eliminate them from U' as they aren't unlabeled anymore).
        for user_idx, item_idx, label in (labeled1 + labeled2):
            train[user_idx, item_idx] = label
            u_prime[user_idx, item_idx] = 0

        # Replenish U' with 2*p + 2*n samples from U.
        i = 0
        while (i < (2*args.number_positives + 2*args.number_negatives) ):
            rnd_user = np.random.randint(0, high=nusers, dtype='l')
            rnd_item = np.random.randint(0, high=nitems, dtype='l')
            if (train[rnd_user, rnd_item] == 0.0): # TODO: user better precision (machine epsilon instead of == 0.0)
                u_prime[rnd_user, rnd_item] = 1.0
                i += 1

    # evaluate the ranking quality
    at = args.rec_length
    n_eval = 0
    for test_user in range(nusers):
        user_profile = train[test_user]
        relevant_items = test[test_user].indices
        if len(relevant_items) > 0:
            #TODO: evaluate both recommenders not just the first.
            n_eval += 1

            # H1 recommendation.
            # this will rank **all** items
            recommended_items = h1.recommend(user_id=test_user, exclude_seen=True)
            # evaluate the recommendation list with ranking metrics ONLY
            roc_auc_[i_fold] += roc_auc(recommended_items, relevant_items)
            precision_[i_fold] += precision(recommended_items, relevant_items, at=at)
            recall_[i_fold] += recall(recommended_items, relevant_items, at=at)
            map_[i_fold] += map(recommended_items, relevant_items, at=at)
            mrr_[i_fold] += rr(recommended_items, relevant_items, at=at)
            ndcg_[i_fold] += ndcg(recommended_items, relevant_items, relevance=test[test_user].data, at=at)

            # H2 recommendation.
            # this will rank **all** items
            recommended_items_2 = h2.recommend(user_id=test_user, exclude_seen=True)
            # evaluate the recommendation list with ranking metrics ONLY
            roc_auc_2[i_fold] += roc_auc(recommended_items_2, relevant_items)
            precision_2[i_fold] += precision(recommended_items_2, relevant_items, at=at)
            recall_2[i_fold] += recall(recommended_items_2, relevant_items, at=at)
            map_2[i_fold] += map(recommended_items_2, relevant_items, at=at)
            mrr_2[i_fold] += rr(recommended_items_2, relevant_items, at=at)
            ndcg_2[i_fold] += ndcg(recommended_items_2, relevant_items, relevance=test[test_user].data, at=at)

    # H1 evaluation
    roc_auc_[i_fold] /= n_eval
    precision_[i_fold] /= n_eval
    recall_[i_fold] /= n_eval
    map_[i_fold] /= n_eval
    mrr_[i_fold] /= n_eval
    ndcg_[i_fold] /= n_eval

    # H2 evaluation.
    roc_auc_2[i_fold] /= n_eval
    precision_2[i_fold] /= n_eval
    recall_2[i_fold] /= n_eval
    map_2[i_fold] /= n_eval
    mrr_2[i_fold] /= n_eval
    ndcg_2[i_fold] /= n_eval

    i_fold += 1

logger.info('Ranking quality for H1')
logger.info('ROC-AUC: {:.4f}'.format(roc_auc_.mean()))
logger.info('Precision@{}: {:.4f}'.format(at, precision_.mean()))
logger.info('Recall@{}: {:.4f}'.format(at, recall_.mean()))
logger.info('MAP@{}: {:.4f}'.format(at, map_.mean()))
logger.info('MRR@{}: {:.4f}'.format(at, mrr_.mean()))
logger.info('NDCG@{}: {:.4f}'.format(at, ndcg_.mean()))

logger.info('Ranking quality for H2')
logger.info('ROC-AUC: {:.4f}'.format(roc_auc_2.mean()))
logger.info('Precision@{}: {:.4f}'.format(at, precision_2.mean()))
logger.info('Recall@{}: {:.4f}'.format(at, recall_2.mean()))
logger.info('MAP@{}: {:.4f}'.format(at, map_2.mean()))
logger.info('MRR@{}: {:.4f}'.format(at, mrr_2.mean()))
logger.info('NDCG@{}: {:.4f}'.format(at, ndcg_2.mean()))
