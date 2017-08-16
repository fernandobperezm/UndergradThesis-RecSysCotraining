# -*- coding: utf-8 -*-
'''
Politecnico di Milano.
tuning.py

Description: This file contains the fitting and evaluation of two
            It makes an evaluation using Co-Training and without it, the
            evaluation metrics are RMSE, roc_auc, precision, recall, map, ndcg, rr.

Modified by Fernando Pérez.

Last modified on 25/03/2017.
'''

# Import Python utils.
import argparse
import logging
from collections import OrderedDict
from datetime import datetime as dt
import pdb
import sys

# Numpy and scipy.
import numpy as np
import scipy as sp

# Import utils such as
from implementation.utils.data_utils import read_dataset, df_to_csr, df_to_dok, df_to_lil, results_to_file
from implementation.utils.split import holdout
from implementation.utils.metrics import roc_auc, precision, recall, map, ndcg, rr
from implementation.utils.evaluation import Evaluation

# Import recommenders classes.
from implementation.recommenders.item_knn import ItemKNNRecommender
from implementation.recommenders.user_knn import UserKNNRecommender
from implementation.recommenders.slim import SLIM, MultiThreadSLIM
from implementation.recommenders.mf import FunkSVD, IALS_numpy, AsySVD, BPRMF
from implementation.recommenders.non_personalized import Random, TopPop, GlobalEffects
from implementation.recommenders.content import ContentBasedRecommender
from implementation.recommenders.cotraining import CoTraining
from implementation.recommenders.bpr import BPRMF_THEANO

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

available_recommenders = OrderedDict([
    ('top_pop', TopPop),
    ('global_effects', GlobalEffects),
    ('content', ContentBasedRecommender),
    ('item_knn', ItemKNNRecommender),
    ('user_knn', UserKNNRecommender),
    ('SLIM', SLIM),
    ('SLIM_mt', MultiThreadSLIM),
    ('FunkSVD', FunkSVD),
    ('AsySVD', AsySVD),
    ('IALS_np', IALS_numpy),
    ('BPRMF', BPRMF),
    ('BPRMF_THEANO', BPRMF_THEANO)
])

# let's use an ArgumentParser to read input arguments
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('--results_path', type=str, default='')
parser.add_argument('--results_file', type=str, default='')
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
parser.add_argument('--rec_length', type=int, default=10)
args = parser.parse_args()

# get the recommender class
assert args.recommender_1 in available_recommenders, 'Unknown recommender: {}'.format(args.recommender_1)
RecommenderClass_1 = available_recommenders[args.recommender_1]

# parse recommender parameters
init_args_recomm_1 = OrderedDict()
if args.params_1:
    for p_str in args.params_1.split(','):
        key, value = p_str.split('=')
        try:
            init_args_recomm_1[key] = eval(value)
        except:
            init_args_recomm_1[key] = value

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
logger.info('Computing the holdout split at: {:.0f}%'.format(args.holdout_perc * 100))

train_df, test_df = holdout(dataset,
                            user_key=args.user_key,
                            item_key=args.item_key,
                            perc=args.holdout_perc,
                            seed=1234,
                            clean_test=True)

# Create our label and unlabeled samples set.
# As the train set will be modifed in the co-training approach, it's more
# efficient to modify a dok_matrix than a csr_matrix.
train = df_to_lil(train_df,
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

# Evaluations cotrained.
eval_ctr = Evaluation(results_path=args.results_path, results_file=args.results_file, test_set=test, val_set = None, at = 10,co_training=True)

# Recommender evaluation.
results_to_file(args.results_path + args.results_file, header=True) # Write the header of the file.

neighbors = range(1,501,50)
shrinkages = range(0,300,50)

i = 0
for neighbor in neighbors:
    for shrinkage in shrinkages:
        # pdb.set_trace()
        logger.info("Case #Neighbors: {} - Shrinkage: {}".format(neighbor,shrinkage))
        # Recommender to be tuned.
        logger.info('Instantiating recommender.')
        h1_ctr = ItemKNNRecommender(k=neighbor, shrinkage=shrinkage, similarity='adj-cosine', normalize=True, sparse_weights=True)

        logger.info('Fitting recommender.')
        tic = dt.now()
        h1_ctr.fit(train)
        logger.info('Finished the fitting process in time: {}'.format(dt.now() - tic))

        logger.info('Evaluating recommender.')
        tic = dt.now()
        eval_ctr.eval(recommenders={h1_ctr.short_str(): h1_ctr}, minRatingsPerUser=1)
        eval_ctr.log_by_index(i, h1_ctr, h1_ctr)
        logger.info('Finished the Evaluation process in time: {}'.format(dt.now() - tic))

        i += 1


# Plotting.
try:
    eval_ctr.plot_all_recommenders(recommenders={h1_ctr.short_str(): h1_ctr}, n_iters=args.number_iterations)
except:
    logger.info('Could not save the figures: {}'.format(sys.exc_info()))
