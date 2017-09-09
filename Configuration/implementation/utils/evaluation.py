'''
Politecnico di Milano.
evaluation.py

Description: This file contains the definition and implementation of a evaluation
             metrics for RecSys under a module.

Modified by Fernando Pérez.

Last modified on 06/09/2017.
'''

import random as random

import logging
import csv
import numpy as np
import scipy.sparse as sps
import implementation.utils.metrics as metrics
import implementation.utils.data_utils as data_utils
from implementation.recommenders.base import check_matrix

import matplotlib
# Directive to save the images in PNG without X windows environment.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

class Evaluation(object):
    """The class represents an evaluation framework for Recommender Systems.

        The class provides different methods to evaluate recommenders and plot
        those results.

        It can evaluate the recommenders for top-N recommendation and rating
        prediction using the metrics defined in metrics.py.

        It can calculate the popularity bins for a dataset and count which items
        fall in which bin.

        It provides methods to transform a Pandas.Dataframe into an evaluation
        object.

        It also provides methods to log and plot the results.

        Attributes:
            * results_path: where the results folder is located.
            * results_file: the name of the results file.
            * test_set: dataset to be used for testing purposes.
            * val_set: dataset to be used for validation purposes.
            * at: size of the top-N recommendation list.
            * cotraining: used cotraining or not.
            * bins: to which popularity bin falls each item.
            * eval_bins: calculate the popularity bins.
            * rec_evals: holds the evaluation for each metric for each recommender.
            * nbins: the number of popularity bins.

    """

    def __init__(self, results_path, results_file, test_set, val_set = None, at = 10, co_training=False, eval_bins = False):
        """Constructor of the class.

            Args:
                * results_path: where the results folder is located.
                * results_file: the name of the results file.
                * test_set: dataset to be used for testing purposes.
                * val_set: dataset to be used for validation purposes.
                * at: size of the top-N recommendation list.
                * cotraining: used cotraining or not.
                * bins: to which popularity bin falls each item.
                * eval_bins: calculate the popularity bins.
                * rec_evals: holds the evaluation for each metric for each
                             recommender.

            Args type:
                * results_path: str
                * results_file: str
                * test_set: Scipy.Sparse matrix instance.
                * val_set: Scipy.Sparse matrix instance.
                * at: int
                * cotraining: bool
                * bins: Dictionary<int:int>
                * eval_bins: bool
                * rec_evals: Dictionary<str:Dictionary<str:[float]>>
        """
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

    def add_statistics(self,recommenders,both):
        """Add statistics info into the Evaluation instance.

            The statistics that are added inside the instance are the agreement
            between the recommenders and the number of labeled items at each
            iteration.

            Args:
                * recommenders: The recommenders we will log alongisde their
                                 individual statistics.
                * both: represents the statistics of the agreement between
                        the recommenders.

            Args type:
                * recommenders: Dictionary<str:Dictionary<str:AnyType>>
                * both: Dictionary<str:int>

            recommenders arguments:
                * 'recommender': recommender instance.
                * 'positive': positive elements only by the recommender
                * 'negative': negative elements only by the recommender
                * 'neutral': neutral elements only by the recommender
                * 'pos_labeled': number of p-most positive elements labeled.
                * 'neg_labeled': number of n-most negative elements labeled.
                * 'total_labeled': number of p-most + n-most elements labeled.

            both arguments:
                * 'both_pos': agreement in positive items,
                * 'both_neg': agreement in negative items
                * 'both_neutral': agreement in neutral items

        """
        # The agreement first.
        # Checking if the agreement for both recommenders is already there.
        if (not 'both' in self.rec_evals.keys()):
            self.rec_evals['both'] = dict()
            self.rec_evals['both']['positive'] = list()
            self.rec_evals['both']['negative'] = list()
            self.rec_evals['both']['neutral'] = list()

        # Adding the agreement between recommenders.
        self.rec_evals['both']['positive'].append(both['both_pos'])
        self.rec_evals['both']['negative'].append(both['both_neg'])
        self.rec_evals['both']['neutral'].append(both['both_neutral'])

        # Adding the disagreement between recommenders.
        for rec_key in recommenders.keys():
            if (not 'positive' in self.rec_evals[rec_key]):
                self.rec_evals[rec_key]['positive'] = list()

            if (not 'negative' in self.rec_evals[rec_key]):
                self.rec_evals[rec_key]['negative'] = list()

            if (not 'neutral' in self.rec_evals[rec_key]):
                self.rec_evals[rec_key]['neutral'] = list()

            self.rec_evals[rec_key]['positive'].append(recommenders[rec_key]['positive'])
            self.rec_evals[rec_key]['negative'].append(recommenders[rec_key]['negative'])
            self.rec_evals[rec_key]['neutral'].append(recommenders[rec_key]['neutral'])

        # Now the number of p-most and n-most labeled.
        for rec_key in recommenders.keys():
            if (not 'pos_labeled' in self.rec_evals[rec_key]):
                self.rec_evals[rec_key]['pos_labeled'] = list()

            if (not 'neg_labeled' in self.rec_evals[rec_key]):
                self.rec_evals[rec_key]['neg_labeled'] = list()

            if (not 'total_labeled' in self.rec_evals[rec_key]):
                self.rec_evals[rec_key]['total_labeled'] = list()

            self.rec_evals[rec_key]['pos_labeled'].append(recommenders[rec_key]['pos_labeled'])
            self.rec_evals[rec_key]['neg_labeled'].append(recommenders[rec_key]['neg_labeled'])
            self.rec_evals[rec_key]['total_labeled'].append(recommenders[rec_key]['total_labeled'])

    def check_ranked_in_bins(self, ranked_list, rec_key):
        """Checks in which bin each recommended item falls.

            This methods takes a top-N recommendtion list and for each item in
            it, it checks in which popularity bin it falls, afterwards, it
            increments the count of number of recommended items for that popularity
            bin.

            Args:
                * ranked_list: a top-N recommendation list.
                * rec_key: which recommender is giving this ranked list.

            Args type:
                * ranked_list: list of int.
                * rec_key: str

            Returns
                A list containing how many items fell in each bin.
        """
        # if (not 'item_pop_bin' in self.rec_evals[rec_key].keys()):
        #     self.rec_evals[rec_key]['item_pop_bin'] = np.zeros(10, dtype=np.int32)
        #
        # for item_idx in ranked_list:
        #     # in self.bins['item_pop_bin'][item_idx] we will have to which bin
        #     # the item belongs.
        #     bin_idx = self.bins['item_pop_bin'][item_idx]
        #     self.rec_evals[rec_key]['item_pop_bin'][bin_idx] += 1
        pop_bins = np.zeros(10, dtype=np.int32)
        for item_idx in ranked_list:
            # in self.bins['item_pop_bin'][item_idx] we will have to which bin
            # the item belongs.
            bin_idx = self.bins['item_pop_bin'][item_idx]
            pop_bins[bin_idx] += 1
        return pop_bins

    def make_pop_bins(self, URM, type_res):
        """Takes a URM and calculates the popularity bins.

            This methods takes a sparse matrix and creates a user or popularity
            bin. The method by default makes 10 popularity bins.

            Args:
                * URM: the dataset for which we will create the popularity bins.
                * type_res: type of popularity bins to be created, can be
                            `item_pop_bin` and `user_pop_bin`.

            Args type:
                * URM: Scipy.Sparse matrix.
                * type_res: str
        """
        self.nbins = 10
        if (type_res == "item_pop_bin"):
            # Supposing a CSC matrix, dataset.
            URM = check_matrix(URM, 'csc', dtype=np.float32)
            item_pop = np.asarray(np.sum(URM > 0, axis=0)).squeeze().astype(np.int32)
            nitems, = item_pop.shape

            # ascending order popularity, item[0] is the least item idx , item[size-1] is the most popular
            item_idx_pop = item_pop.argsort()

            partition_size = int(nitems / self.nbins)
            pop_bin = 0 # Least popular bin.
            bins_list = []
            for low in range(0, nitems, partition_size):
                high = low + partition_size
                bins_list += list(zip(item_idx_pop[low:high], [pop_bin]*partition_size))
                if (pop_bin) < self.nbins - 1:
                    pop_bin += 1

            if (self.bins is None):
                self.bins = dict()
            self.bins['item_pop_bin'] = dict(bins_list)

        if (type_res == "user_pop_bin"):
            # Supposing a CSR matrix, dataset.
            URM = check_matrix(URM, 'csr', dtype=np.float32)
            user_pop = np.asarray(np.sum(URM > 0, axis=1)).squeeze().astype(np.int32)
            nusers, = user_pop.shape

            # ascending order popularity, user[0] is the least user idx , user[size-1] is the most popular
            user_idx_pop = user_pop.argsort()

            partition_size = int(nusers / self.nbins)
            pop_bin = 0 # Least popular bin.
            bins_list = []
            for low in range(0, nusers, partition_size):
                high = low + partition_size
                bins_list += list(zip(user_idx_pop[low:high], [pop_bin]*partition_size))
                if (pop_bin) < self.nbins - 1:
                    pop_bin += 1

            if (self.bins is None):
                self.bins = dict()
            self.bins['user_pop_bin'] = dict(bins_list)

    def df_to_eval(self, df, recommenders = None, read_iter=None, type_res="evaluation"):
        """Takes the dataframe information into the evaluation instance.

            Args:
                * df: the information we are going to read
                * recommenders: the recommenders we are going to store the
                                information
                * read_iter: the final iteration to read.
                * type_res: type of results in df, it can be None, `numberlabeled`
                            or `label_comparison`. The first reads the results
                            as the evaluation of the recommenders, the second
                            reads the results as the number of labeled items
                            at each iteration and the last reads the results
                            as the comparison between the type of rating.

            Args type:
                * df: Pandas.Dataframe
                * recommenders: Dictionary<str:Recommender>
                * read_iter: int
                * type_res: str

        """
        # Getting rec1 and rows.
        for rec_key in recommenders.keys():
            recommender,pos = recommenders[rec_key]

            if (not rec_key in self.rec_evals.keys()):
                self.rec_evals[rec_key] = dict()

            if (type_res == "evaluation"):
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

            elif (type_res == "item_pop_bin"):
                if (rec_key in {"TopPop1", "TopPop2", "GlobalEffects1", "GlobalEffects2", "Random"}):
                    rows_rec = df.loc[df.recommender == str(rec_key)]
                else:
                    rows_rec = df.loc[df.recommender == str(recommender)]

                self.rec_evals[rec_key]['item_pop_bin'] = list(
                    rows_rec[['bin_0', 'bin_1', 'bin_2', 'bin_3',
                              'bin_4', 'bin_5', 'bin_6', 'bin_7',
                              'bin_8', 'bin_9']].values[:read_iter].tolist())

    def eval(self, recommenders=None, minRatingsPerUser=1):
        """Performs the evaluation of the recommenders.

            This method evaluates all the recommenders inside `recommenders`
            by two methods, rating prediction and top-N recommendation list.
            The metric used for the first is `RMSE`, for the second, the metrics
            are: `MAP`, `ROC-AUC`, `NDCG`, `Precision`, `Recall`, and `MRR`.

            It also checks for a given top-N recommendation list, the popularity
            of each item.

            This method performs the evaluation only to the users in the test
            set that has more or equal ratings than `minRatingsPerUser`.

            After the evaluation is finished for all those users, the average
            of each metric is taken.

            Args:
                * recommenders: the recommenders we are going to evaluate.
                * minRatingsPerUser: number of minimum ratings that each user
                                     needs to have in the test set in order
                                     to be evaluated.

            Args type:
                * recommenders: Dictionary<str:Recommenders>
                * minRatingsPerUser: int

        """
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

        rmse_ = np.zeros(shape=(n_recs,))
        roc_auc_ = np.zeros(shape=(n_recs,))
        precision_ = np.zeros(shape=(n_recs,))
        recall_ = np.zeros(shape=(n_recs,))
        map_ = np.zeros(shape=(n_recs,))
        mrr_ = np.zeros(shape=(n_recs,))
        ndcg_ = np.zeros(shape=(n_recs,))

        pop_bins_by_recommender = dict()

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
                self.rec_evals[rec_key]['item_pop_bin'] = list()

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

                ranked_items = rec_to_eval.recommend(user_id=test_user,
                                                     n=at,
                                                     exclude_seen=True
                                                    )
                predicted_relevant_items = rec_to_eval.predict(user_id=test_user,
                                                               rated_indices=relevant_items
                                                              )

                # evaluate the recommendation list with RMSE and ranking metrics.
                is_relevant = np.in1d(ranked_items,
                                      relevant_items,
                                      assume_unique=True
                                     )
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

                if (self.eval_bins):
                    if (not rec_key in pop_bins_by_recommender.keys()):
                        pop_bins_by_recommender[rec_key] = np.zeros(self.nbins, dtype=np.int32)

                    pop_bins_by_recommender[rec_key] += self.check_ranked_in_bins(ranked_list=ranked_items,rec_key=rec_key)

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

            if (self.eval_bins):
                self.rec_evals[rec_key]['item_pop_bin'].append(pop_bins_by_recommender[rec_key])

            i += 1

    def log_to_file(self,log_type,recommenders,args):
        """Writes a log into a file of the results.

            The type of log is determined by `log_type`. If it is `evaluation`
            then the evaluation for the current iteration is logged. If it is
            `labeling` then the number of positive, negative and total labeled
            items is saved in one file, in the other the comparison between the
            agreement between the recommender is saved. If it is `tuning` then
            the hyper-parameters evaluation  for the recommender are logged.

            Args:
                * log_type: the type of logging we are performing. Possible values
                            are `evaluation`, `labeling` and `tuning`.
                * recommenders: The recommenders we will log.
                * args: represents keyword arguments for the function.

            Args type:
                * log_type: str
                * recommenders: Dictionary<str:Recommender>
                * args: Dictionary<str:AnyType>


            Keywords arguments:
                * 'index': current iteration.
                * 'rec_key': dictionary of recommenders containing:
                    ** pos_lab_rec: number of items the recommender labeled as
                                    positive.
                    ** neg_lab_rec: number of items the recommender labeled as
                                    negative.
                    ** total_lab_rec: total number of items the recommender
                                      labeled.
                * 'pos_only_first': How many items only the recommender 1
                                    labeled as positive.
                * 'pos_only_second': How many items only the recommender 2
                                     labeled as positive.
                * 'neg_only_first': How many items only the recommender 1
                                    labeled as negative.
                * 'neg_only_second': How many items only the recommender 2
                                      labeled as negative.
                * 'neutral_only_first': How many items only the recommender 1
                                        labeled as neutral.
                * 'neutral_only_second': How many items only the recommender 2
                                         labeled as neutral.
                * 'both_pos': How many items both recommenders labeled as
                              positive.
                * 'both_neg': How many items both recommenders labeled as
                              negative.
                * 'both_neutral': How many items both recommenders labeled as
                                  neutral.
        """
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
            # File: numberlabeled.csv
            columns = ['iteration','recommender',
                       'pos_labeled', 'neg_labeled', 'total_labeled'
                      ]
            filepath1 = filepath + "numberlabeled.csv"

            try:
                csvfile = open(filepath1, mode='r')
                csvfile.close()
            except:
                logger.info("Creating header for file: {}".format(filepath1))
                with open(filepath1, 'w', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile,
                                           delimiter=' ',
                                           quotechar='|',
                                           quoting=csv.QUOTE_MINIMAL)
                    csvwriter.writerow(columns)

            with open(filepath1, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile,
                                       delimiter=' ',
                                       quotechar='|',
                                       quoting=csv.QUOTE_MINIMAL)

                for rec_key in recommenders.keys():
                    recommender = recommenders[rec_key]
                    pos_rec, neg_rec, total_rec = args[rec_key]
                    row = [index, str(recommender), pos_rec, neg_rec, total_rec]
                    csvwriter.writerow(row)

            # File: label_comparison.csv
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
                    csvwriter = csv.writer(csvfile,
                                           delimiter=' ',
                                           quotechar='|',
                                           quoting=csv.QUOTE_MINIMAL
                                          )
                    csvwriter.writerow(columns)

            with open(filepath2, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile,
                                       delimiter=' ',
                                       quotechar='|',
                                       quoting=csv.QUOTE_MINIMAL
                                      )

                row = [index,
                       args['both_pos'], args['both_neg'], args['both_neutral'],
                       args['pos_only_first'], args['neg_only_first'], args['neutral_only_first'],
                       args['pos_only_second'], args['neg_only_second'], args['neutral_only_second']
                       ]
                csvwriter.writerow(row)

            # File: item_pop_bin.csv
            if (not self.eval_bins):
                return

            columns = ['iteration','pop_bin_type','recommender'] +\
                      ["bin_{}".format(i) for i in range(self.nbins)]

            filepath3 = filepath + "item_pop_bin.csv"

            try:
                csvfile = open(filepath3, mode='r')
                csvfile.close()
            except:
                logger.info("Creating header for file: {}".format(filepath3))
                with open(filepath3, 'w', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile,
                                           delimiter=' ',
                                           quotechar='|',
                                           quoting=csv.QUOTE_MINIMAL
                                          )
                    csvwriter.writerow(columns)

            with open(filepath3, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile,
                                       delimiter=' ',
                                       quotechar='|',
                                       quoting=csv.QUOTE_MINIMAL
                                      )

                for rec_key in recommenders.keys():
                    recommender = recommenders[rec_key]
                    row = [index, 'item_pop_bin', str(recommender)] +\
                          list(self.rec_evals[rec_key]['item_pop_bin'][index])
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

    def plot_popularity_bins(self, recommenders, niter, file_prefix, bin_type):
        """Plots the number of items recommended for each popularity bin.

            This method creates a bar plot in which each bar color is a different
            recommender. For each recommender the number of items recommended by
            its popularity is plotted. The plot is saved into a file.

            Args:
                * recommenders: The recommenders we will plot.
                * niter: number of Co-Training iterations performed.
                * file_prefix: prefix of the file logged.
                * bin_type: plot popularity bins of users or items.

            Args type:
                * recommenders: Dictionary<str:Recommender>
                * niter: int
                * file_prefix: str
                * bin_type: str
        """
        xdata_rangebins = np.arange(self.nbins)

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

            ydata_n_elements_by_bin = self.rec_evals[rec_key][bin_type][niter]
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
        plt.close('all')

    def plot_statistics(self, recommenders=None, n_iters=30, file_prefix="", statistic_type=None):
        """Plots the different evaluated statistics for all the recommenders.

            This method creates a line plot in which line color represents
            a recommender and each marker represents an statistic.

            If the statistic_type is None, then the evaluation of each metric
            is plotted. If the statistic_type is `numberlabeled` then
            it is plotted how many positive, negative and total items both
            recommender could label while doing Co-Training. If the statistic_type
            is `label_comparison` then agreement between the recommenders is
            plotted.

            Args:
                * recommenders: The recommenders we will plot.
                * n_iters: number of Co-Training iterations performed.
                * file_prefix: prefix of the file logged.
                * statistic_type: plot evaluation, `numberlabeled` or `label_comparison`.

            Args type:
                * recommenders: Dictionary<str:Recommender>
                * niter: int
                * file_prefix: str
                * statistic_type: str
        """
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
                        plt.close('all')

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
                        plt.close('all')

    def plot_all_recommenders(self, recommenders=None, n_iters=30, file_prefix=""):
        """Plots the evaluation for rating prediction and top-N recommendation.

            This method creates a plot where all the recommenders inside
            `recommenders` are plotted in the same box. Each recommender has
            its own color. There is one plot for each metric.

            Args:
                * recommenders: The recommenders we will plot.
                * n_iters: number of Co-Training iterations performed.
                * file_prefix: prefix of the file logged.

            Args type:
                * recommenders: Dictionary<str:Recommender>
                * niter: int
                * file_prefix: str
        """
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
            plt.close('all')
