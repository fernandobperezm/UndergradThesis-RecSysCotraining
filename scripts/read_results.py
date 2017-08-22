# -*- coding: utf-8 -*-
'''
Politecnico di Milano.
read_results.py

Description: This file reads the results given in the path: results_path + results_file.
             It puts all the info inside an Evaluation instance and proceeds to
             plot all the results.

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
import pandas as pd

# Import utils such as
from implementation.utils.data_utils import read_dataset, df_to_csr, df_to_dok, df_to_lil, results_to_file, results_to_df
from implementation.utils.split import holdout
from implementation.utils.evaluation import Evaluation

# Import recommenders classes.
from implementation.recommenders.item_knn import ItemKNNRecommender
from implementation.recommenders.user_knn import UserKNNRecommender
from implementation.recommenders.slim import SLIM, MultiThreadSLIM
from implementation.recommenders.mf import FunkSVD, IALS_numpy, AsySVD, BPRMF
from implementation.recommenders.non_personalized import Random, TopPop, GlobalEffects
from implementation.recommenders.content import ContentBasedRecommender
from implementation.recommenders.cotraining import CoTraining

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
parser.add_argument('--recommender_2', type=str, default='top_pop')
parser.add_argument('--params_2', type=str, default=None)
parser.add_argument('--rec_length', type=int, default=10)
parser.add_argument('--number_iterations', type=int, default=30)
parser.add_argument('--number_positives', type=int, default=1)
parser.add_argument('--number_negatives', type=int, default=3)
parser.add_argument('--number_unlabeled', type=int, default=75)
parser.add_argument('--to_read',type=str,default=None)
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
# logger.info('Co-Training env. #Positives: {}, #Negatives: {}, #Unlabeled: {}'.format(args.number_positives, args.number_negatives, args.number_unlabeled))
# logger.info('Reading {}'.format(args.dataset))
# dataset, item_to_idx, user_to_idx = read_dataset(
#     args.dataset,
#     header=args.header,
#     sep=args.sep,
#     columns=args.columns,
#     make_binary=args.make_binary,
#     binary_th=args.binary_th,
#     item_key=args.item_key,
#     user_key=args.user_key,
#     rating_key=args.rating_key)

# nusers, nitems = dataset.user_idx.max() + 1, dataset.item_idx.max() + 1
# logger.info('The dataset has {} users and {} items'.format(nusers, nitems))
#
# # compute the k-fold split
# logger.info('Computing the holdout split at: {:.0f}%'.format(args.holdout_perc * 100))

# train_df, test_df = holdout(dataset,
#                             user_key=args.user_key,
#                             item_key=args.item_key,
#                             perc=args.holdout_perc,
#                             seed=1234,
#                             clean_test=True)
#
# # Create our label and unlabeled samples set.
# # As the train set will be modifed in the co-training approach, it's more
# # efficient to modify a dok_matrix than a csr_matrix.
# train = df_to_lil(train_df,
#                   is_binary=args.is_binary,
#                   nrows=nusers,
#                   ncols=nitems,
#                   item_key='item_idx',
#                   user_key='user_idx',
#                   rating_key=args.rating_key)
#
# # Create our test set.
# test = df_to_csr(test_df,
#                  is_binary=args.is_binary,
#                  nrows=nusers,
#                  ncols=nitems,
#                  item_key='item_idx',
#                  user_key='user_idx',
#                  rating_key=args.rating_key)
# # Baseline recommenders.
# global_effects = GlobalEffects()
# top_pop = TopPop()
# random = Random(seed=1234,binary_ratings=args.is_binary)

# read the results
filepath = args.results_path + args.results_file
results = results_to_df(filepath)

# Create the recommenders.
h1_ctr = RecommenderClass_1(**init_args_recomm_1)
h2_ctr = RecommenderClass_2(**init_args_recomm_2)

# Creating the evaluation instance.
evaluation = Evaluation(results_path=args.results_path, results_file=args.results_file, test_set=None, val_set = None, at = args.rec_length, co_training=True)
# evaluation.df_to_eval(results, h1_ctr, h2_ctr)

# Baseline fitting.
# global_effects.fit(train)
# top_pop.fit(train)
# random.fit(train)

# Evaluate the baselines.
# evaluation.eval_baselines(random=random,global_effects=global_effects,top_pop=top_pop)
#
# # Plotting all the results.
# evaluation.plot_all_recommenders(rec_1=h1_ctr, rec_2=h2_ctr) # First 7 figures.
# evaluation.plot_all(rec_index=0,rec=h1_ctr) # First 7 figures. Rec_index
# evaluation.plot_all(rec_index=1,rec=h2_ctr) # Third 7 figures.

if args.to_read is not None:
    list_to_read = args.to_read.split(",")
    for option_to_read in list_to_read:
        filename = args.results_path + option_to_read + ".csv"
        results = results_to_df(filename,type_res=option_to_read)
        evaluation.df_to_eval(df=results,
                              rec_1=h1_ctr,
                              rec_2=h2_ctr,
                              recommenders = {h1_ctr.short_str(): (h1_ctr,1),
                                              h2_ctr.short_str(): (h2_ctr,2)
                                             },
                              read_iter=None,
                              type_res=option_to_read
                             )
        evaluation.plot_statistics(recommenders={h1_ctr.short_str(): (h1_ctr,1),
                                                 h2_ctr.short_str(): (h2_ctr,2),
                                                 'both': (None,3),
                                                },
                                   n_iters=args.number_iterations,
                                   file_prefix="Together_",
                                   statistic_type=option_to_read
                                  )
        evaluation.plot_statistics(recommenders={h1_ctr.short_str(): (h1_ctr,1)
                                                },
                                   n_iters=args.number_iterations,
                                   file_prefix=h1_ctr.short_str() + "_",
                                   statistic_type=option_to_read
                                  )
        evaluation.plot_statistics(recommenders={h2_ctr.short_str(): (h2_ctr,2)
                                                },
                                   n_iters=args.number_iterations,
                                   file_prefix=h2_ctr.short_str() + "_",
                                   statistic_type=option_to_read
                                  )

        evaluation.plot_statistics(recommenders={'both': (None,3),
                                                },
                                   n_iters=args.number_iterations,
                                   file_prefix='Both' + "_",
                                   statistic_type=option_to_read
                                  )
        evaluation.plot_statistics(recommenders={'both': (None,3),
                                                },
                                   n_iters=args.number_iterations,
                                   file_prefix='Both' + "_",
                                   statistic_type=option_to_read
                                  )
