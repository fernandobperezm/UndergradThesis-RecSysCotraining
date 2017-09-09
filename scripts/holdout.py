# -*- coding: utf-8 -*-
'''
Politecnico di Milano.
holdout.py

Description: This is the main file to run a Holdout evaluation using Co-Training
             between two recommenders.

Modified by Fernando Pérez.

Last modified on 05/09/2017.
'''

# Import Python utils.
import argparse
import csv
import logging
import sys
import traceback
from collections import OrderedDict
from datetime import datetime as dt

# Numpy and scipy.
import numpy as np
import scipy as sp

# Import utils such as
from implementation.utils.data_utils import read_dataset, df_to_csr, df_to_dok, df_to_lil, results_to_file, results_to_df
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
from implementation.recommenders.SLIM_BPR_Mono import SLIM_BPR_Mono

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
    ('BPRMF_THEANO', BPRMF_THEANO),
    ('SLIM_BPR', SLIM_BPR_Mono)
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
parser.add_argument('--recover_cotraining', action='store_true', default=False)
parser.add_argument('--recover_iter', type=int, default=None)
parser.add_argument('--make_pop_bins', action="store_true", default=False)
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
logger.info('Co-Training env. #Positives: {}, #Negatives: {}, #Unlabeled: {}'.format(
    args.number_positives, args.number_negatives, args.number_unlabeled)
)
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

# compute the holdout split.
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

# Baseline recommenders.
global_effects_1 = GlobalEffects()
global_effects_2 = GlobalEffects()
top_pop_1 = TopPop()
top_pop_2 = TopPop()
random = Random(seed=1234,binary_ratings=args.is_binary)

# Co-Trained recommenders.
h1_ctr = RecommenderClass_1(**init_args_recomm_1)
h2_ctr = RecommenderClass_2(**init_args_recomm_2)

# Recommenders dictionary.
recommenders = dict()
recommenders[h1_ctr.short_str()] = h1_ctr
recommenders[h2_ctr.short_str()] = h2_ctr
recommenders["TopPop1"] = top_pop_1
recommenders["TopPop2"] = top_pop_2
recommenders["GlobalEffects1"] = global_effects_1
recommenders["GlobalEffects2"] = global_effects_2
recommenders[random.short_str()] = random

# Evaluations cotrained.
eval_ctr = Evaluation(results_path=args.results_path,
                      results_file=args.results_file,
                      test_set=test,
                      val_set = None,
                      at = 10,
                      co_training=True,
                      eval_bins = args.make_pop_bins
                     )

# If making popularity bins, then create them.
if (args.make_pop_bins):
    logger.info("Creating the user and item popularity bins.")
    eval_ctr.make_pop_bins(URM=train, type_res="item_pop_bin")
    eval_ctr.make_pop_bins(URM=train, type_res="user_pop_bin")

# Read the previous results if recovering.
if (args.recover_cotraining):
    filepath = args.results_path + args.results_file
    results = results_to_df(filepath)
    eval_ctr.df_to_eval(results,
                        h1_ctr,
                        h2_ctr,
                        recommenders=recommenders,
                        read_iter=args.recover_iter
                       )

cotraining = CoTraining(rec_1=h1_ctr,
                        rec_2=h2_ctr,
                        eval_obj=eval_ctr,
                        n_iters = args.number_iterations,
                        n_labels = args.number_unlabeled,
                        p_most = args.number_positives,
                        n_most = args.number_negatives
                       )

# Write the header of the evaluation results file.
try:
    results = open(args.results_path + args.results_file, mode='r')
    results.close()
except:
    filepath = args.results_path + args.results_file
    logger.info("Creating header for file: {}".format(filepath))
    available_metrics = ['rmse','roc_auc','precision', 'recall', 'map', 'mrr', 'ndcg']
    columns = ['cotraining','iteration', '@k', 'recommender'] + available_metrics
    with open(filepath, 'w', newline='') as resultsfile:
        csvwriter = csv.writer(resultsfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(columns)

# Cotraining fitting and evaluation.
logger.info('Beggining the Co-Training process.')
tic = dt.now()
cotraining.fit(train,
               eval_iter=True,
               binary_ratings=args.is_binary,
               recommenders=recommenders,
               baselines=True,
               recover_cotraining=args.recover_cotraining,
               recover_iter=args.recover_iter
              )
logger.info('Finished the Co-Training process in time: {}'.format(dt.now() - tic))

# Plotting.
try:
    only_h1 = recommenders.copy()
    only_h2 = recommenders.copy()
    del(only_h1[h2_ctr.short_str()])
    del(only_h2[h1_ctr.short_str()])
    if (args.recover_cotraining):
        # All the recommenders in the same plot.
        eval_ctr.plot_all_recommenders(recommenders={h1_ctr.short_str(): h1_ctr,
                                                     h2_ctr.short_str(): h2_ctr},
                                       n_iters=args.number_iterations,
                                       file_prefix="Together_"
                                      )
        # Only the first recommender.
        eval_ctr.plot_all_recommenders(recommenders={h1_ctr.short_str(): h1_ctr},
                                       n_iters=args.number_iterations,
                                       file_prefix=h1_ctr.short_str()+"_"
                                      )
        # Only the second recommender.
        eval_ctr.plot_all_recommenders(recommenders={h2_ctr.short_str(): h2_ctr},
                                       n_iters=args.number_iterations,
                                       file_prefix=h2_ctr.short_str()+"_"
                                      )
    else:
        # All the recommenders in the same plot, including baselines.
        eval_ctr.plot_all_recommenders(recommenders=recommenders,
                                       n_iters=args.number_iterations,
                                       file_prefix="Together_"
                                      )
        # All the recommenders without the second recommender.
        eval_ctr.plot_all_recommenders(recommenders=only_h1,
                                       n_iters=args.number_iterations,
                                       file_prefix=h1_ctr.short_str()+"_"
                                      )
        # All the recommenders without the first recommender.
        eval_ctr.plot_all_recommenders(recommenders=only_h2,
                                       n_iters=args.number_iterations,
                                       file_prefix=h2_ctr.short_str()+"_"
                                      )

    for n_iter in range(0,args.number_iterations+1,10):
        eval_ctr.plot_popularity_bins(recommenders={h1_ctr.short_str():(h1_ctr,1),
                                                      h2_ctr.short_str():(h2_ctr,2),
                                                     },
                                        niter = n_iter,
                                        file_prefix="Together_",
                                        bin_type="item_pop_bin"
                                       )

        eval_ctr.plot_popularity_bins(recommenders={h1_ctr.short_str():(h1_ctr,1),
                                                     },
                                        niter = n_iter,
                                        file_prefix=h1_ctr.short_str() + "_",
                                        bin_type="item_pop_bin"
                                       )

        eval_ctr.plot_popularity_bins(recommenders={h2_ctr.short_str():(h2_ctr,2),
                                                     },
                                        niter = n_iter,
                                        file_prefix=h2_ctr.short_str() + "_",
                                        bin_type="item_pop_bin"
                                       )

    for statistic in ['label_comparison','numberlabeled']:
        eval_ctr.plot_statistics(recommenders={h1_ctr.short_str(): (h1_ctr,1),
                                               h2_ctr.short_str(): (h2_ctr,2),
                                               'both': (None,3),
                                                },
                                   n_iters=args.number_iterations,
                                   file_prefix="Together_",
                                   statistic_type=statistic
                                  )
        eval_ctr.plot_statistics(recommenders={h1_ctr.short_str(): (h1_ctr,1)
                                                },
                                   n_iters=args.number_iterations,
                                   file_prefix=h1_ctr.short_str() + "_",
                                   statistic_type=statistic
                                  )
        eval_ctr.plot_statistics(recommenders={h2_ctr.short_str(): (h2_ctr,2)
                                                },
                                   n_iters=args.number_iterations,
                                   file_prefix=h2_ctr.short_str() + "_",
                                   statistic_type=statistic
                                  )

        eval_ctr.plot_statistics(recommenders={'both': (None,3),
                                                },
                                   n_iters=args.number_iterations,
                                   file_prefix='Both' + "_",
                                   statistic_type=statistic
                                  )
        eval_ctr.plot_statistics(recommenders={'both': (None,3),
                                                },
                                   n_iters=args.number_iterations,
                                   file_prefix='Both' + "_",
                                   statistic_type=statistic
                                  )
except:
    error_path = args.results_path + "errors.txt"
    error_file = open(error_path, 'a')
    logger.info('Could not save the figures: {}'.format(sys.exc_info()))
    traceback.print_exc(file=error_file)
    error_file.close()
