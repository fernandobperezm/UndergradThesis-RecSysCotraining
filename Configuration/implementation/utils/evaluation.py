'''
Politecnico di Milano.
evaluation.py

Description: This file contains the definition and implementation of a evaluation
             metrics for RecSys under a module.

Modified by Fernando Pérez.

Last modified on 25/03/2017.
'''

import random as random

import logging
import numpy as np
import scipy.sparse as sps
import implementation.utils.metrics as metrics
import implementation.utils.data_utils as data_utils
from implementation.recommenders.base import check_matrix

import pdb
import csv

import matplotlib
matplotlib.use('Agg') # Directive to save the images in PNG without X windows environment.
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")


class Evaluation(object):
    """ EVALUATION class for RecSys"""

    def __init__(self, results_path, results_file, test_set, val_set = None, at = 10, co_training=False, eval_bins = False):
        '''
            Args:
                * recommender: A Recommender Class object that represents the first
                         recommender.
                * nusers: The number of users to evaluate. It represents user indices.
        '''
        super(Evaluation, self).__init__()
        self.results_path = results_path
        self.results_file = results_file
        self.test_set = test_set
        self.val_set = val_set
        self.at = at
        self.cotraining = co_training
        self.bins = dict()
        self.eval_bins = eval_bins
        self.rec_evals = dict()
        self.rmse = (list(), list())
        self.roc_auc = (list(), list())
        self.precision = (list(), list())
        self.recall = (list(), list())
        self.map = (list(), list())
        self.mrr = (list(), list())
        self.ndcg = (list(), list())

    def __str__(self):
        return "Evaluation(Rec={}\n)".format(
            self.recommender.__str__())

    def check_ranked_in_bins(self,ranked_list,user_idx,rec_key,n=10):
        # pdb.set_trace()
        if (not 'item_pop_bin' in self.rec_evals[rec_key].keys()):
            self.rec_evals[rec_key]['item_pop_bin'] = np.zeros(10)

        # if (not rec_key in self.bins['item_pop_bin']):
        #     self.rec_evals[rec_key]['item_pop_bin'] = np.zeros(10)

        for item_idx in ranked_list:
            # in self.bins['item_pop_bin'][item_idx] we will have to which bin
            # the item belongs.
            bin_idx = self.bins['item_pop_bin'][item_idx]
            self.rec_evals[rec_key]['item_pop_bin'][bin_idx] += 1

    def matrix_to_eval(self,URM, type_res):
        nbins = 10
        if (type_res == "item_pop_bin"):
            # Supposing a CSC matrix, dataset.
            item_pop = np.asarray(np.sum(URM > 0, axis=0)).squeeze().astype(np.int32)
            nitems, = item_pop.shape

            # ascending order popularity, item[0] is the least item idx , item[size-1] is the most popular
            item_idx_pop = item_pop.argsort()

            partition_size = int(nitems / nbins)
            pop_bin = 0 # Least popular bin.
            bins_list = []
            for low in range(0, nitems, partition_size):
                high = low + partition_size
                bins_list += list(zip(item_idx_pop[low:high], [pop_bin]*partition_size))
                if (pop_bin) < nbins - 1:
                    pop_bin += 1

            if (self.bins is None):
                self.bins = dict()
            self.bins['item_pop_bin'] = dict(bins_list)

        if (type_res == "user_pop_bin"):
            # Supposing a CSR matrix, dataset.
            user_pop = np.asarray(np.sum(URM > 0, axis=1)).squeeze().astype(np.int32)
            nusers, = user_pop.shape

            # ascending order popularity, user[0] is the least user idx , user[size-1] is the most popular
            user_idx_pop = user_pop.argsort()

            partition_size = int(nusers / nbins)
            pop_bin = 0 # Least popular bin.
            bins_list = []
            for low in range(0, nusers, partition_size):
                high = low + partition_size
                bins_list += list(zip(user_idx_pop[low:high], [pop_bin]*partition_size))
                if (pop_bin) < nbins - 1:
                    pop_bin += 1

            if (self.bins is None):
                self.bins = dict()
            self.bins['user_pop_bin'] = dict(bins_list)

    def df_to_eval(self, df, rec_1, rec_2, recommenders = None, read_iter=None, type_res=None ):
        # Getting rec1 and rows.
        # pdb.set_trace()
        for rec_key in recommenders.keys():
            recommender,pos = recommenders[rec_key]

            if (not rec_key in self.rec_evals.keys()):
                self.rec_evals[rec_key] = dict()

            if (type_res is None):
                if (rec_key in {"TopPop1", "TopPop2", "GlobalEffects1", "GlobalEffects2", "Random"}):
                    # Returns all the rows that have in the column 'recommender'
                    # the recommender name.
                    rows_rec = df.loc[df.recommender == str(rec_key)]
                else:
                    rows_rec = df.loc[df.recommender == str(recommender)]

                self.rec_evals[rec_key] = dict()

                # Put the information.
                self.rec_evals[rec_key]['RMSE'] = list(rows_rec.rmse.values[:read_iter])
                self.rec_evals[rec_key]['ROC_AUC'] = list(rows_rec.roc_auc.values[:read_iter])
                self.rec_evals[rec_key]['Precision'] = list(rows_rec.precision.values[:read_iter])
                self.rec_evals[rec_key]['Recall'] = list(rows_rec.recall.values[:read_iter])
                self.rec_evals[rec_key]['MAP'] = list(rows_rec.map.values[:read_iter])
                self.rec_evals[rec_key]['MRR'] = list(rows_rec.mrr.values[:read_iter])
                self.rec_evals[rec_key]['NDCG'] = list(rows_rec.ndcg.values[:read_iter])

            elif (type_res == "numberlabeled"):
                if (rec_key in {"TopPop1", "TopPop2", "GlobalEffects1", "GlobalEffects2", "Random"}):
                    rows_rec = df.loc[df.recommender == str(rec_key)]
                else:
                    rows_rec = df.loc[df.recommender == str(recommender)]

                self.rec_evals[rec_key]['pos_labeled'] = list(rows_rec.pos_labeled.values[:read_iter])
                self.rec_evals[rec_key]['neg_labeled'] = list(rows_rec.neg_labeled.values[:read_iter])
                self.rec_evals[rec_key]['total_labeled'] = list(rows_rec.total_labeled.values[:read_iter])

            elif (type_res == "label_comparison"):
                columns = ['iteration',
                           'both_positive', 'both_negative', 'both_neutral',
                           'pos_only_first', 'neg_only_first', 'neutral_only_first',
                           'pos_only_second', 'neg_only_second', 'neutral_only_second']

                if (not 'both' in self.rec_evals.keys()):
                    self.rec_evals['both'] = dict()
                    self.rec_evals['both']['positive'] = list(df.both_positive.values[:read_iter])
                    self.rec_evals['both']['negative'] = list(df.both_negative.values[:read_iter])
                    self.rec_evals['both']['neutral'] = list(df.both_neutral.values[:read_iter])

                if (pos == 1):
                    pos_col = list(df.pos_only_first.values[:read_iter])
                    neg_col = list(df.neg_only_first.values[:read_iter])
                    neutral_col = list(df.neutral_only_first.values[:read_iter])

                elif (pos == 2):
                    pos_col = list(df.pos_only_second.values[:read_iter])
                    neg_col = list(df.neg_only_second.values[:read_iter])
                    neutral_col = list(df.neutral_only_second.values[:read_iter])

                self.rec_evals[rec_key]['positive'] = pos_col
                self.rec_evals[rec_key]['negative'] = neg_col
                self.rec_evals[rec_key]['neutral'] = neutral_col

            elif (type_res == "dataset_item_pop"):
                pdb.set_trace()
                # This creates an array where item_pop[item_idx] = popularity
                items_pop = df.item_idx.value_counts()
                nitems, = items_pop.shape
                partition_size = nitems / 10
                # for item_idx in range(nitems):
                #
                #
                # print(df)
                # pass

            elif (type_res == "dataset_user_pop"):
                pdb.set_trace()
                users = df.user_idx.unique()
                # print(df)
                # pass

    def eval(self, recommenders=None, minRatingsPerUser=1,  ):
        '''
            recommenders: dict that contains as key the recommender name
                          and as value the reference of the recommender.
        '''
        self.test_set = check_matrix(self.test_set, 'csr', dtype=np.float32)

        nusers, nitems = self.test_set.shape
        at = self.at
        n_eval = 0

        rows = self.test_set.indptr
        numRatings = np.ediff1d(rows)
        mask = numRatings >= minRatingsPerUser
        usersToEvaluate = np.arange(nusers)[mask]
        usersToEvaluate = list(usersToEvaluate)

        recommenders_to_evaluate = list(recommenders.keys())
        n_recs = len(recommenders_to_evaluate)
        rmse_, roc_auc_, precision_, recall_, map_, mrr_, ndcg_ = np.zeros(shape=(n_recs,)), np.zeros(shape=(n_recs,)), np.zeros(shape=(n_recs,)), np.zeros(shape=(n_recs,)), np.zeros(shape=(n_recs,)), np.zeros(shape=(n_recs,)), np.zeros(shape=(n_recs,))
        for rec_key in recommenders_to_evaluate:
            if (not rec_key in self.rec_evals):
                self.rec_evals[rec_key] = dict()
                self.rec_evals[rec_key]['RMSE'] = list()
                self.rec_evals[rec_key]['ROC_AUC'] = list()
                self.rec_evals[rec_key]['Precision'] = list()
                self.rec_evals[rec_key]['Recall'] = list()
                self.rec_evals[rec_key]['MAP'] = list()
                self.rec_evals[rec_key]['MRR'] = list()
                self.rec_evals[rec_key]['NDCG'] = list()

        # row_indices, _ = self.test_set.nonzero() # users with ratings in the test set. nonzero returns a tuple, the first element are the rows.
        # relevant_users = np.unique(row_indices) # In this way we only consider users with ratings in the test set and not ALL the users.
        # for test_user in relevant_users:
        for test_user in usersToEvaluate:
            if (test_user % 10000 == 0):
                logger.info("Evaluating user {}".format(test_user))

            # Getting user_profile by it's rated items (relevant_items) in the test.
            relevant_items = self.test_set[test_user].indices
            relevant_predictions = self.test_set[test_user,relevant_items].toarray()
            relevant_data = self.test_set[test_user].data
            i = 0
            for rec_key in recommenders_to_evaluate:
                rec_to_eval = recommenders[rec_key]

                ranked_items = rec_to_eval.recommend(user_id=test_user, n=at, exclude_seen=True)
                predicted_relevant_items = rec_to_eval.predict(user_id=test_user, rated_indices=relevant_items)

                # evaluate the recommendation list with RMSE and ranking metrics.
                is_relevant = np.in1d(ranked_items, relevant_items, assume_unique=True)
                # TopPop only works for ranking metrics.
                if (rec_key == "TopPop1" or rec_key == "TopPop2"):
                    rmse_[i] += 0.0
                else:
                    rmse_[i] += metrics.rmse(predicted_relevant_items, relevant_predictions)
                roc_auc_[i] += metrics.roc_auc(is_relevant)
                precision_[i] += metrics.precision(is_relevant)
                recall_[i] += metrics.recall(is_relevant, relevant_items)
                map_[i] += metrics.map(is_relevant, relevant_items)
                mrr_[i] += metrics.rr(is_relevant)
                ndcg_[i] += metrics.ndcg(ranked_items, relevant_items, relevance=relevant_data, at=at)

                if (self.eval_bins and
                    rec_key != "TopPop1" and rec_key != "TopPop2" and
                    rec_key != "GlobalEffects1" and rec_key != "GlobalEffects2" and
                    rec_key != "Random"):

                    self.check_ranked_in_bins(ranked_list=ranked_items,user_idx=test_user,rec_key=rec_key,n=at)

                i += 1

            # Increase the number of evaluations performed.
            n_eval += 1

        # Recommender evaluation.
        i = 0
        for rec_key in recommenders_to_evaluate:
            self.rec_evals[rec_key]['RMSE'].append(rmse_[i] / n_eval)
            self.rec_evals[rec_key]['ROC_AUC'].append(roc_auc_[i] / n_eval)
            self.rec_evals[rec_key]['Precision'].append(precision_[i] / n_eval)
            self.rec_evals[rec_key]['Recall'].append(recall_[i] / n_eval)
            self.rec_evals[rec_key]['MAP'].append(map_[i] / n_eval)
            self.rec_evals[rec_key]['MRR'].append(mrr_[i] / n_eval)
            self.rec_evals[rec_key]['NDCG'].append(ndcg_[i] / n_eval)

            i += 1

    def log_all(self):
        for index in range(len(self.rmse)):
            self.log_by_index(index)

    def log_to_file(self,log_type,recommenders,args):
        '''
            recommenders: dictionary of recommenders to eval.
            type:
                * 'evaluation'
                * 'labeling'
                * 'tuning'
            args: dictionary of arguments.
                * 'index'
                * 'rec_key': dictionary of recommenders containing:
                    * pos_lab_rec, neg_lab_rec, total_lab_rec as a triplet
                * 'pos_1'
                * 'pos_2'
                * 'neg_1'
                * 'neg_2'
                * 'both_pos'
                * 'both_neg'
                * 'both_neutral'
        '''
        filepath = self.results_path
        columns = []
        index = args['index']

        if (log_type == 'evaluation'):
            available_metrics = ['rmse','roc_auc','precision', 'recall', 'map', 'mrr', 'ndcg']
            columns = ['cotraining','iteration', '@k', 'recommender'] + available_metrics
            filepath += self.results_file

            try:
                csvfile = open(filepath, mode='r')
                csvfile.close()
            except:
                logger.info("Creating header for file: {}".format(filepath))
                with open(filepath, 'w', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    csvwriter.writerow(columns)


            with open(filepath, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)

                for rec_key in recommenders.keys():
                    recommender = recommenders[rec_key]
                    if (rec_key in {"TopPop1", "TopPop2", "GlobalEffects1", "GlobalEffects2", "Random"}):
                        try:
                            rec_evaluation = [self.rec_evals[rec_key]['RMSE'][index],
                                              self.rec_evals[rec_key]['ROC_AUC'][index],
                                              self.rec_evals[rec_key]['Precision'][index],
                                              self.rec_evals[rec_key]['Recall'][index],
                                              self.rec_evals[rec_key]['MAP'][index],
                                              self.rec_evals[rec_key]['MRR'][index],
                                              self.rec_evals[rec_key]['NDCG'][index]
                                            ]
                            row = [self.cotraining, index, self.at, rec_key] + rec_evaluation
                            csvwriter.writerow(row)
                        except:
                            pass
                    else:
                        rec_evaluation = [self.rec_evals[rec_key]['RMSE'][index],
                                          self.rec_evals[rec_key]['ROC_AUC'][index],
                                          self.rec_evals[rec_key]['Precision'][index],
                                          self.rec_evals[rec_key]['Recall'][index],
                                          self.rec_evals[rec_key]['MAP'][index],
                                          self.rec_evals[rec_key]['MRR'][index],
                                          self.rec_evals[rec_key]['NDCG'][index]
                                          ]
                        row = [self.cotraining, index, self.at, str(recommender)] + rec_evaluation
                        csvwriter.writerow(row)

        elif (log_type == 'labeling'):
            columns = ['iteration','recommender', 'pos_labeled', 'neg_labeled', 'total_labeled']
            filepath1 = filepath + "numberlabeled.csv"

            try:
                csvfile = open(filepath1, mode='r')
                csvfile.close()
            except:
                logger.info("Creating header for file: {}".format(filepath1))
                with open(filepath1, 'w', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    csvwriter.writerow(columns)

            with open(filepath1, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)

                for rec_key in recommenders.keys():
                    recommender = recommenders[rec_key]
                    pos_rec, neg_rec, total_rec = args[rec_key]
                    row = [index, str(recommender), pos_rec, neg_rec, total_rec]
                    csvwriter.writerow(row)

            columns = ['iteration',
                       'both_positive', 'both_negative', 'both_neutral',
                       'pos_only_first', 'neg_only_first', 'neutral_only_first',
                       'pos_only_second', 'neg_only_second', 'neutral_only_second']

            filepath2 = filepath + "label_comparison.csv"

            try:
                csvfile = open(filepath2, mode='r')
                csvfile.close()
            except:
                logger.info("Creating header for file: {}".format(filepath2))
                with open(filepath2, 'w', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    csvwriter.writerow(columns)

            with open(filepath2, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)

                row = [index,
                       args['both_pos'], args['both_neg'], args['both_neutral'],
                       args['pos_only_first'], args['neg_only_first'], args['neutral_only_first'],
                       args['pos_only_second'], args['neg_only_second'], args['neutral_only_second']
                       ]
                csvwriter.writerow(row)


        elif (log_type == 'tuning'):
            available_metrics = ['rmse','roc_auc','precision', 'recall', 'map', 'mrr', 'ndcg']
            columns = ['recommender'] + available_metrics
            filepath += "tuning.csv"

            try:
                csvfile = open(filepath, mode='r')
                csvfile.close()
            except:
                logger.info("Creating header for file: {}".format(filepath))
                with open(filepath, 'w', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    csvwriter.writerow(columns)

            with open(filepath, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)

                for rec_key in recommenders.keys():
                    recommender = recommenders[rec_key]
                    if (rec_key in {"TopPop1", "TopPop2", "GlobalEffects1", "GlobalEffects2", "Random"}):
                        try:
                            rec_evaluation = [self.rec_evals[rec_key]['RMSE'][index],
                                              self.rec_evals[rec_key]['ROC_AUC'][index],
                                              self.rec_evals[rec_key]['Precision'][index],
                                              self.rec_evals[rec_key]['Recall'][index],
                                              self.rec_evals[rec_key]['MAP'][index],
                                              self.rec_evals[rec_key]['MRR'][index],
                                              self.rec_evals[rec_key]['NDCG'][index]
                                            ]
                            row = [rec_key] + rec_evaluation
                            csvwriter.writerow(row)
                        except:
                            pass
                    else:
                        rec_evaluation = [self.rec_evals[rec_key]['RMSE'][index],
                                          self.rec_evals[rec_key]['ROC_AUC'][index],
                                          self.rec_evals[rec_key]['Precision'][index],
                                          self.rec_evals[rec_key]['Recall'][index],
                                          self.rec_evals[rec_key]['MAP'][index],
                                          self.rec_evals[rec_key]['MRR'][index],
                                          self.rec_evals[rec_key]['NDCG'][index]
                                          ]
                        row = [str(recommender)] + rec_evaluation
                        csvwriter.writerow(row)

    def log_by_index(self,index,rec_1, rec_2):
        filepath = self.results_path + self.results_file
        data_utils.results_to_file(filepath=filepath,
                        cotraining=self.cotraining,
                        iterations=index,
                        recommender1=rec_1,
                        evaluation1=[self.rec_evals[rec_1.short_str()]['RMSE'][index],
                                     self.rec_evals[rec_1.short_str()]['ROC_AUC'][index],
                                     self.rec_evals[rec_1.short_str()]['Precision'][index],
                                     self.rec_evals[rec_1.short_str()]['Recall'][index],
                                     self.rec_evals[rec_1.short_str()]['MAP'][index],
                                     self.rec_evals[rec_1.short_str()]['MRR'][index],
                                     self.rec_evals[rec_1.short_str()]['NDCG'][index]
                                    ],
                        at=self.at
                        )

        data_utils.results_to_file(filepath=filepath,
                        cotraining=self.cotraining,
                        iterations=index,
                        recommender1=rec_2,
                        evaluation1=[self.rec_evals[rec_2.short_str()]['RMSE'][index],
                                     self.rec_evals[rec_2.short_str()]['ROC_AUC'][index],
                                     self.rec_evals[rec_2.short_str()]['Precision'][index],
                                     self.rec_evals[rec_2.short_str()]['Recall'][index],
                                     self.rec_evals[rec_2.short_str()]['MAP'][index],
                                     self.rec_evals[rec_2.short_str()]['MRR'][index],
                                     self.rec_evals[rec_2.short_str()]['NDCG'][index]
                                    ],
                        at=self.at
                        )

    def log_number_labeled(self,index, rec_1, rec_2, nlabeled1, nlabeled2):
        filepath = self.results_path + "numberlabeled.csv"
        data_utils.results_to_file(filepath=filepath,
                        cotraining=self.cotraining,
                        iterations=index,
                        recommender1=rec_1,
                        evaluation1=[nlabeled1],
                        at=self.at
                        )

        data_utils.results_to_file(filepath=filepath,
                        cotraining=self.cotraining,
                        iterations=index,
                        recommender1=rec_2,
                        evaluation1=[nlabeled2],
                        at=self.at
                        )

    def plot_popularity_bins(self, recommenders, niter, file_prefix, bin_type):
        # pdb.set_trace()
        # X data.
        nbins = 10
        xdata_rangebins = np.arange(nbins)

        # Properties.
        width = 0.25
        colors = ['#ff0000','#00ff00','#0000ff']

        fig, ax = plt.subplots()
        # add some text for labels, title and axes ticks
        ax.set_ylabel('Number of recommended items')
        ax.set_xlabel('Popularity bins from least to most popular')
        ax.set_title('Number of recommended items with respect to their popularity bin')

        savepath = self.results_path + file_prefix + "Popularity_Bin_ITER{}".format(niter)
        i = 0
        handles = []
        for rec_key in recommenders.keys():
            recommender, pos = recommenders[rec_key]

            ydata_n_elements_by_bin = self.rec_evals[rec_key][bin_type]
            rects = plt.bar(xdata_rangebins + i * width,
                             ydata_n_elements_by_bin,
                             width,
                             color=colors[i],
                             label=rec_key+"-TrainingSet{}".format(pos)
                            )
            handles.append(rects)
            i +=1

        plt.legend(handles=handles,
                   bbox_to_anchor=(0,-0.35,1,0.2),
                   loc="upper left",
                   mode="expand",
                   borderaxespad=0,
                   ncol=2
                  )
        plt.savefig(savepath, bbox_inches="tight")
        plt.clf()

    def plot_statistics(self, recommenders=None, n_iters=30, file_prefix="", statistic_type=None):
        if (statistic_type is None):
            self.plot_all_recommenders(recommenders=recommenders, n_iters=n_iters, file_prefix=file_prefix)

        else:
            for rec_key in recommenders.keys():
                recommender, pos = recommenders[rec_key]

                if (statistic_type == 'numberlabeled'):
                    # self.rec_evals[rec_key]['number_pos'] = list(rows_rec.pos_labeled.values[:read_iter])
                    # self.rec_evals[rec_key]['number_neg'] = list(rows_rec.neg_labeled.values[:read_iter])
                    # self.rec_evals[rec_key]['number_total'] = list(rows_rec.total_labeled.values[:read_iter])
                    # if ('both' in recommenders.keys()):
                    #     del recommenders['both']

                    recommenders_to_evaluate = list(recommenders.keys())
                    n_recs = len(recommenders_to_evaluate)
                    iterations = np.arange(n_iters+1)

                    # colors = ['b-*','g-s','k-8','r-^','y-X','c-d','m-*',]
                    markers = ['*','s','^']
                    linestyle = '-'
                    colors = ['#ff0000','#00ff00','#0000ff']

                    titles = ['Number and type of items labeled by each recommender.',
                             ]
                    savepaths = [self.results_path + file_prefix + "number_labeled_ITER{}".format(n_iters),
                                ]
                    ylabels = ['Number of rated items']
                    label_types = ['pos_labeled','neg_labeled','total_labeled']

                    # Iterating for each metric
                    for i in range(len(ylabels)):
                        fig = plt.figure(i)
                        plt.title(titles[i])
                        plt.ylabel(ylabels[i])
                        plt.xlabel('Iterations')
                        plt.grid(True)
                        savepath = savepaths[i]
                        handles = []

                        j = 0
                        # Plotting in the same figure the different recommenders.
                        for rec_key in recommenders_to_evaluate:
                            rec = recommenders[rec_key] # load the recommender reference.
                            rec_eval = self.rec_evals[rec_key] # Load the recommender evaluation.

                            if (rec_key !='both'):
                                k = 0
                                for label in label_types:
                                    rec_plot, = plt.plot(iterations,
                                                         rec_eval[label],
                                                         marker = markers[k],
                                                         linestyle = linestyle,
                                                         color=colors[j],
                                                         markerfacecolor=colors[j],
                                                         markeredgecolor=colors[j],
                                                         label=rec_key+label)
                                    handles.append(rec_plot)
                                    k +=1
                                j += 1

                        # plt.legend(handles=handles,
                        #            bbox_to_anchor=(0,0),
                        #            loc="upper left",
                        #            bbox_transform=fig.transFigure,
                        #            ncol=3
                        #           )
                        plt.legend(handles=handles,
                                   bbox_to_anchor=(0,-0.35,1,0.2),
                                   loc="upper left",
                                   mode="expand",
                                   borderaxespad=0,
                                   ncol=2
                                  )
                        plt.savefig(savepath, bbox_inches="tight")
                        plt.clf()

                elif (statistic_type == 'label_comparison'):
                    recommenders_to_evaluate = list(recommenders.keys())
                    n_recs = len(recommenders_to_evaluate)
                    iterations = np.arange(n_iters+1)

                    # colors = ['b-*','g-s','k-8','r-^','y-X','c-d','m-*',]
                    markers = ['*','s','^']
                    linestyle = '-'
                    colors = ['#ff0000','#00ff00','#0000ff']
                    titles = ['Comparison between labeled items for the recommenders.',
                             ]
                    savepaths = [self.results_path + file_prefix + "label_comparison_ITER{}".format(n_iters),
                                ]
                    ylabels = ['Number of rated items']
                    label_types = ['positive','negative','neutral']

                    # Iterating for each metric
                    for i in range(len(ylabels)):
                        fig = plt.figure(i)
                        plt.title(titles[i])
                        plt.ylabel(ylabels[i])
                        plt.xlabel('Iterations')
                        plt.grid(True)
                        savepath = savepaths[i]
                        handles = []

                        j = 0
                        # Plotting in the same figure the different recommenders.
                        for rec_key in recommenders_to_evaluate:
                            # rec = recommenders[rec_key] # load the recommender reference.
                            rec_eval = self.rec_evals[rec_key] # Load the recommender evaluation.

                            k = 0
                            for label in label_types:
                                # rec_plot, = plt.plot(iterations, rec_eval[label], colors[j], label=rec_key+label)
                                rec_plot, = plt.plot(iterations,
                                                     rec_eval[label],
                                                     marker = markers[k],
                                                     linestyle = linestyle,
                                                     color=colors[j],
                                                     markerfacecolor=colors[j],
                                                     markeredgecolor=colors[j],
                                                     label=rec_key+label)
                                handles.append(rec_plot)
                                k +=1
                            j += 1

                        # plt.legend(handles=handles,
                        #            bbox_to_anchor=(0,0),
                        #            loc="upper left",
                        #            bbox_transform=fig.transFigure,
                        #            ncol=3
                        #           )
                        plt.legend(handles=handles,
                                  bbox_to_anchor=(0,-0.35,1,0.2),
                                   loc="upper left",
                                   mode="expand",
                                   borderaxespad=0,
                                   ncol=3
                                  )
                        plt.savefig(savepath, bbox_inches="tight")
                        plt.clf()

    def plot_all_recommenders(self, recommenders=None, n_iters=30, file_prefix=""):
        # pdb.set_trace()
        recommenders_to_evaluate = list(recommenders.keys())
        n_recs = len(recommenders_to_evaluate)
        iterations = np.arange(n_iters+1)

        colors = ['b-*','g-s','k-8','r-^','y-X','c-d','m-*']
        titles = ['RMSE between the recommenders.',
                  'ROC-AUC@{} between the recommenders.'.format(self.at),
                  'Precision@{} between the recommenders.'.format(self.at),
                  'Recall@{} between the recommenders.'.format(self.at),
                  'MAP@{} between the recommenders.'.format(self.at),
                  'MRR@{} between the recommenders.'.format(self.at),
                  'NDCG@{} between the recommenders.'.format(self.at)
                 ]
        savepaths = [self.results_path + file_prefix + "RMSE_{}iter.png".format(n_iters),
                     self.results_path + file_prefix + "ROC-AUC_{}iter.png".format(n_iters),
                     self.results_path + file_prefix + "Precision_{}iter.png".format(n_iters),
                     self.results_path + file_prefix + "Recall_{}iter.png".format(n_iters),
                     self.results_path + file_prefix + "MAP_{}iter.png".format(n_iters),
                     self.results_path + file_prefix + "MRR_{}iter.png".format(n_iters),
                     self.results_path + file_prefix + "NDCG_{}iter.png".format(n_iters)
                    ]
        ylabels = ['RMSE', 'ROC_AUC', 'Precision', 'Recall', 'MAP', 'MRR', 'NDCG']

        # Iterating for each metric
        for i in range(len(ylabels)):
            fig = plt.figure(i+1)
            plt.title(titles[i])
            plt.ylabel(ylabels[i])
            plt.xlabel('Iterations')
            plt.grid(True)
            savepath = savepaths[i]
            handles = []

            j = 0
            # Plotting in the same figure the different recommenders.
            for rec_key in recommenders_to_evaluate:
                rec = recommenders[rec_key] # load the recommender reference.
                rec_eval = self.rec_evals[rec_key] # Load the recommender evaluation.

                if (ylabels[i] != "RMSE" or (rec_key != "TopPop1" and rec_key != "TopPop2")):
                    rec_plot, = plt.plot(iterations, rec_eval[ylabels[i]], colors[j], label=rec_key)
                    handles.append(rec_plot)

                j += 1

            # plt.legend(handles=handles,
            #            bbox_to_anchor=(0,0),
            #            loc="upper left",
            #            bbox_transform=fig.transFigure,
            #            ncol=3
            #           )
            plt.legend(handles=handles,
                      bbox_to_anchor=(0,-0.35,1,0.2),
                       loc="upper left",
                       mode="expand",
                       borderaxespad=0,
                       ncol=3
                      )
            plt.savefig(savepath, bbox_inches="tight")
            plt.clf()

        # # Plot each metric in a different file.
        # # RMSE.
        # fig = plt.figure(1)
        # plt.title('RMSE between the recommenders.')
        # # self_plot, = plt.plot(iterations, self.rmse,  'r-', label=self.recommender.short_str())
        # eval1_plot, = plt.plot(iterations, self.rmse[0], 'b-', label=rec_1.short_str())
        # eval2_plot, = plt.plot(iterations, self.rmse[1], 'g-', label=rec_2.short_str())
        # random_plot, = plt.plot(iterations, [random_eval[0]]*n_iters, 'k-', label='Random')
        # ge_plot, = plt.plot(iterations, [ge_eval[0]]*n_iters, 'r-', label='Global Effects')
        # # tp_plot, = plt.plot(iterations, [tp_eval[0]]*n_iters, 'y-', label='Top Popular')
        # plt.ylabel('RMSE')
        # plt.xlabel('Iterations')
        # plt.legend(handles=[eval1_plot,eval2_plot,random_plot,ge_plot])
        # plt.grid(True)
        # savepath = self.results_path + "Together_RMSE_{}iter.png".format(n_iters)
        # plt.savefig(savepath)
        # plt.clf()
        #
        # # ROC-AUC.
        # fig = plt.figure(2)
        # plt.title('ROC-AUC@{} between the recommenders.'.format(self.at))
        # # self_plot, = plt.plot(iterations, self.roc_auc,  'r-', label=self.recommender.short_str())
        # eval1_plot, = plt.plot(iterations, self.roc_auc[0], 'b-', label=rec_1.short_str())
        # eval2_plot, = plt.plot(iterations, self.roc_auc[1], 'g-', label=rec_2.short_str())
        # random_plot, = plt.plot(iterations, [random_eval[1]]*n_iters, 'k-', label='Random')
        # ge_plot, = plt.plot(iterations, [ge_eval[1]]*n_iters, 'r-', label='Global Effects')
        # tp_plot, = plt.plot(iterations, [tp_eval[1]]*n_iters, 'y-', label='Top Popular')
        # plt.ylabel('ROC-AUC')
        # plt.xlabel('Iterations')
        # plt.legend(handles=[eval1_plot,eval2_plot,random_plot,ge_plot,tp_plot])
        # plt.grid(True)
        # savepath = self.results_path + "Together_ROC-AUC_{}iter.png".format(n_iters)
        # plt.savefig(savepath)
        # plt.clf()
        #
        # # Precision
        # fig = plt.figure(3)
        # plt.title('Precision@{} between the recommenders.'.format(self.at))
        # # self_plot, = plt.plot(iterations, self.precision,  'r-', label=self.recommender.short_str())
        # eval1_plot, = plt.plot(iterations, self.precision[0], 'b-', label=rec_1.short_str())
        # eval2_plot, = plt.plot(iterations, self.precision[1], 'g-', label=rec_2.short_str())
        # random_plot, = plt.plot(iterations, [random_eval[2]]*n_iters, 'k-', label='Random')
        # ge_plot, = plt.plot(iterations, [ge_eval[2]]*n_iters, 'r-', label='Global Effects')
        # tp_plot, = plt.plot(iterations, [tp_eval[2]]*n_iters, 'y-', label='Top Popular')
        # plt.ylabel('Precision')
        # plt.xlabel('Iterations')
        # plt.legend(handles=[eval1_plot,eval2_plot,random_plot,ge_plot,tp_plot])
        # plt.grid(True)
        # savepath = self.results_path + "Together_Precision_{}iter.png".format(n_iters)
        # plt.savefig(savepath)
        # plt.clf()
        #
        # # Recall
        # fig = plt.figure(4)
        # plt.title('Recall@{} between the recommenders.'.format(self.at))
        # # self_plot, = plt.plot(iterations, self.recall,  'r-', label=self.recommender.short_str())
        # eval1_plot, = plt.plot(iterations, self.recall[0], 'b-', label=rec_1.short_str())
        # eval2_plot, = plt.plot(iterations, self.recall[1], 'g-', label=rec_2.short_str())
        # random_plot, = plt.plot(iterations, [random_eval[3]]*n_iters, 'k-', label='Random')
        # ge_plot, = plt.plot(iterations, [ge_eval[3]]*n_iters, 'r-', label='Global Effects')
        # tp_plot, = plt.plot(iterations, [tp_eval[3]]*n_iters, 'y-', label='Top Popular')
        # plt.ylabel('Recall')
        # plt.xlabel('Iterations')
        # plt.legend(handles=[eval1_plot,eval2_plot,random_plot,ge_plot,tp_plot])
        # plt.grid(True)
        # savepath = self.results_path + "Together_Recall_{}iter.png".format(n_iters)
        # plt.savefig(savepath)
        # plt.clf()
        #
        # # MAP
        # fig = plt.figure(5)
        # plt.title('MAP@{} between the recommenders.'.format(self.at))
        # # self_plot, = plt.plot(iterations, self.map,  'r-', label=self.recommender.short_str())
        # eval1_plot, = plt.plot(iterations, self.map[0], 'b-', label=rec_1.short_str())
        # eval2_plot, = plt.plot(iterations, self.map[1], 'g-', label=rec_2.short_str())
        # random_plot, = plt.plot(iterations, [random_eval[4]]*n_iters, 'k-', label='Random')
        # ge_plot, = plt.plot(iterations, [ge_eval[4]]*n_iters, 'r-', label='Global Effects')
        # tp_plot, = plt.plot(iterations, [tp_eval[4]]*n_iters, 'y-', label='Top Popular')
        # plt.ylabel('MAP')
        # plt.xlabel('Iterations')
        # plt.grid(True)
        # plt.legend(handles=[eval1_plot,eval2_plot,random_plot,ge_plot,tp_plot])
        # savepath = self.results_path + "Together_MAP_{}iter.png".format(n_iters)
        # plt.savefig(savepath)
        # plt.clf()
        #
        # # MRR
        # fig = plt.figure(6)
        # plt.title('MRR@{} between the recommenders.'.format(self.at))
        # # self_plot, = plt.plot(iterations, self.mrr, 'r-', label=self.recommender.short_str())
        # eval1_plot, = plt.plot(iterations, self.mrr[0], 'b-', label=rec_1.short_str())
        # eval2_plot, = plt.plot(iterations, self.mrr[1], 'g-', label=rec_2.short_str())
        # random_plot, = plt.plot(iterations, [random_eval[5]]*n_iters, 'k-', label='Random')
        # ge_plot, = plt.plot(iterations, [ge_eval[5]]*n_iters, 'r-', label='Global Effects')
        # tp_plot, = plt.plot(iterations, [tp_eval[5]]*n_iters, 'y-', label='Top Popular')
        # plt.ylabel('MRR')
        # plt.xlabel('Iterations')
        # plt.legend(handles=[eval1_plot,eval2_plot,random_plot,ge_plot,tp_plot])
        # plt.grid(True)
        # savepath = self.results_path + "Together_MRR_{}iter.png".format(n_iters)
        # plt.savefig(savepath)
        # plt.clf()
        #
        # # NDCG
        # fig = plt.figure(7)
        # plt.title('NDCG@{} between the recommenders.'.format(self.at))
        # # self_plot, = plt.plot(iterations, self.ndcg, 'r-', label=self.recommender.short_str())
        # eval1_plot, = plt.plot(iterations, self.ndcg[0], 'b-', label=rec_1.short_str())
        # eval2_plot, = plt.plot(iterations, self.ndcg[1], 'g-', label=rec_2.short_str())
        # random_plot, = plt.plot(iterations, [random_eval[6]]*n_iters, 'k-', label='Random')
        # ge_plot, = plt.plot(iterations, [ge_eval[6]]*n_iters, 'r-', label='Global Effects')
        # tp_plot, = plt.plot(iterations, [tp_eval[6]]*n_iters, 'y-', label='Top Popular')
        # plt.ylabel('NDCG')
        # plt.xlabel('Iterations')
        # plt.legend(handles=[eval1_plot,eval2_plot,random_plot,ge_plot,tp_plot])
        # plt.grid(True)
        # savepath = self.results_path + "Together_NDCG_{}iter.png".format(n_iters)
        # plt.savefig(savepath)
        # plt.clf()
