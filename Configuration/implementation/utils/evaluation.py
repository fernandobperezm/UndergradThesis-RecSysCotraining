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

import pdb

import matplotlib
matplotlib.use('Agg') # Directive to save the images in PNG without X windows environment.
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

class Evaluation(object):
    """ EVALUATION class for RecSys"""

    def __init__(self, results_path, results_file, test_set, val_set = None, at = 10, co_training=False):
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

    def df_to_eval(self, df, rec_1, rec_2):
        # Getting rec1 and rows.
        rows_rec1 = df.loc[df.recommender == str(rec_1)]
        rows_rec2 = df.loc[df.recommender == str(rec_2)]

        self.rmse = ( rows_rec1.rmse.values, rows_rec2.rmse.values )
        self.roc_auc = ( rows_rec1.roc_auc.values, rows_rec2.roc_auc.values )
        self.precision = ( rows_rec1.precision.values, rows_rec2.precision.values )
        self.recall = ( rows_rec1.recall.values, rows_rec2.recall.values )
        self.map = ( rows_rec1.map.values, rows_rec2.map.values )
        self.mrr = ( rows_rec1.mrr.values, rows_rec2.mrr.values )
        self.ndcg = ( rows_rec1.ndcg.values, rows_rec2.ndcg.values )

    def eval_baselines(self,random,global_effects,top_pop):
        nusers, nitems = self.test_set.shape
        at = self.at
        n_eval = 0
        rmse_random, roc_auc_random, precision_random, recall_random, map_random, mrr_random, ndcg_random = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        rmse_ge, roc_auc_ge, precision_ge, recall_ge, map_ge, mrr_ge, ndcg_ge = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        rmse_tp, roc_auc_tp, precision_tp, recall_tp, map_tp, mrr_tp, ndcg_tp = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        row_indices, _ = self.test_set.nonzero() # users with ratings in the test set. nonzero returns a tuple, the first element are the rows.
        relevant_users = np.unique(row_indices) # In this way we only consider users with ratings in the test set and not ALL the users.
        for test_user in relevant_users:
        # for test_user in np.arange(start=0,stop=nusers,dtype=np.int32):
            if (test_user % 10000 == 0):
                logger.info("Evaluating user {}".format(test_user))

            # Getting user_profile by it's rated items (relevant_items) in the test.
            # logger.info("Getting relevant items.")
            relevant_items = self.test_set[test_user].indices
            relevant_predictions = self.test_set[test_user,relevant_items].toarray()
            relevant_data = self.test_set[test_user].data

            # Getting user profile given the train set.
            # user_profile = train_set[test_user]

            # Random.
            # recommender recommendation.
            # this will rank self.at (n) items and will predict the score for the relevant items.
            ranked_items = random.recommend(user_id=test_user, n=self.at, exclude_seen=True)
            is_relevant = np.in1d(ranked_items, relevant_items, assume_unique=True)
            predicted_relevant_items = random.predict(user_id=test_user, rated_indices=relevant_items)

            # evaluate the recommendation list with RMSE and ranking metrics.
            rmse_random += metrics.rmse(predicted_relevant_items, relevant_predictions)
            roc_auc_random += metrics.roc_auc(is_relevant)
            precision_random += metrics.precision(is_relevant)
            recall_random += metrics.recall(is_relevant, relevant_items)
            map_random += metrics.map(is_relevant, relevant_items)
            mrr_random += metrics.rr(is_relevant)
            ndcg_random += metrics.ndcg(ranked_items, relevant_items, relevance=relevant_data, at=at)


            # Global Effects
            # recommender recommendation.
            # this will rank self.at (n) items and will predict the score for the relevant items.
            ranked_items_2 = global_effects.recommend(user_id=test_user, n=self.at, exclude_seen=True)
            is_relevant = np.in1d(ranked_items_2, relevant_items, assume_unique=True)
            predicted_relevant_items_2 = global_effects.predict(user_id=test_user, rated_indices=relevant_items)

            # evaluate the recommendation list with RMSE and ranking metrics.
            rmse_ge += metrics.rmse(predicted_relevant_items_2, relevant_predictions)
            roc_auc_ge += metrics.roc_auc(is_relevant)
            precision_ge += metrics.precision(is_relevant)
            recall_ge += metrics.recall(is_relevant, relevant_items)
            map_ge += metrics.map(is_relevant, relevant_items)
            mrr_ge += metrics.rr(is_relevant)
            ndcg_ge += metrics.ndcg(ranked_items_2, relevant_items, relevance=relevant_data, at=at)

            # Top-Pop
            # recommender recommendation.
            # this will rank self.at (n) items and will predict the score for the relevant items.
            ranked_items_3 = top_pop.recommend(user_id=test_user, n=self.at, exclude_seen=True)
            is_relevant = np.in1d(ranked_items_3, relevant_items, assume_unique=True)
            # predicted_relevant_items_3 = top_pop.predict(user_id=test_user, rated_indices=relevant_items)

            # evaluate the recommendation list with RMSE and ranking metrics.
            # rmse_tp += metrics.rmse(predicted_relevant_items_3, relevant_predictions)
            roc_auc_tp += metrics.roc_auc(is_relevant)
            precision_tp += metrics.precision(is_relevant)
            recall_tp += metrics.recall(is_relevant, relevant_items)
            map_tp += metrics.map(is_relevant, relevant_items)
            mrr_tp += metrics.rr(is_relevant)
            ndcg_tp += metrics.ndcg(ranked_items_3, relevant_items, relevance=relevant_data, at=at)

            # Increase the number of evaluations performed.
            n_eval += 1

        # Recommender evaluation.
        # names = ['rmse','roc_auc','precision', 'recall', 'map', 'mrr', 'ndcg']
        # formats = [np.float32,np.float32,np.float32,np.float32,np.float32,np.float32,np.float32]
        self.baselines_eval = dict()
        self.baselines_eval['random'] = [rmse_random / n_eval, roc_auc_random / n_eval, precision_random / n_eval, recall_random / n_eval, map_random / n_eval, mrr_random / n_eval, ndcg_random / n_eval]
        self.baselines_eval['global_effects'] = [rmse_ge / n_eval, roc_auc_ge / n_eval, precision_ge / n_eval, recall_ge / n_eval, map_ge / n_eval, mrr_ge / n_eval, ndcg_ge / n_eval ]
        self.baselines_eval['top_pop'] = [0.0, roc_auc_tp / n_eval, precision_tp / n_eval, recall_tp / n_eval, map_tp / n_eval, mrr_tp / n_eval, ndcg_tp / n_eval]

    def eval(self, rec_1, rec_2):
        nusers, nitems = self.test_set.shape
        at = self.at
        n_eval = 0
        rmse_, roc_auc_, precision_, recall_, map_, mrr_, ndcg_ = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        rmse2_, roc_auc2_, precision2_, recall2_, map2_, mrr2_, ndcg2_ = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        row_indices, _ = self.test_set.nonzero() # users with ratings in the test set. nonzero returns a tuple, the first element are the rows.
        relevant_users = np.unique(row_indices) # In this way we only consider users with ratings in the test set and not ALL the users.
        for test_user in relevant_users:
        # for test_user in np.arange(start=0,stop=nusers,dtype=np.int32):
            if (test_user % 10000 == 0):
                logger.info("Evaluating user {}".format(test_user))

            # Getting user_profile by it's rated items (relevant_items) in the test.
            # logger.info("Getting relevant items.")
            relevant_items = self.test_set[test_user].indices

            # Getting user profile given the train set.
            # user_profile = train_set[test_user]

            # Recommender 1.
            # recommender recommendation.
            # this will rank self.at (n) items and will predict the score for the relevant items.
            # logger.info("Ranking Rec1.")
            ranked_items = rec_1.recommend(user_id=test_user, n=self.at, exclude_seen=True)
            is_relevant = np.in1d(ranked_items, relevant_items, assume_unique=True)
            # logger.info("Predicting Rec1.")
            predicted_relevant_items = rec_1.predict(user_id=test_user, rated_indices=relevant_items)

            # evaluate the recommendation list with RMSE and ranking metrics.
            # logger.info("Evaluating rmse rec1.")
            rmse_ += metrics.rmse(predicted_relevant_items, self.test_set[test_user,relevant_items].toarray())
            # logger.info("Evaluating ranking metrics rec1")
            roc_auc_ += metrics.roc_auc(is_relevant)
            precision_ += metrics.precision(is_relevant)
            recall_ += metrics.recall(is_relevant, relevant_items)
            map_ += metrics.map(is_relevant, relevant_items)
            mrr_ += metrics.rr(is_relevant)
            ndcg_ += metrics.ndcg(ranked_items, relevant_items, relevance=self.test_set[test_user].data, at=at)


            # Recommender 2.
            # recommender recommendation.
            # this will rank self.at (n) items and will predict the score for the relevant items.
            # logger.info("Ranking rec2")
            ranked_items_2 = rec_2.recommend(user_id=test_user, n=self.at, exclude_seen=True)
            is_relevant = np.in1d(ranked_items_2, relevant_items, assume_unique=True)
            # logger.info("Predicting rec2")
            predicted_relevant_items_2 = rec_2.predict(user_id=test_user, rated_indices=relevant_items)

            # evaluate the recommendation list with RMSE and ranking metrics.
            # logger.info("Evaluating rmse rec2")
            rmse2_ += metrics.rmse(predicted_relevant_items_2, self.test_set[test_user,relevant_items].toarray())
            # logger.info("Evaluating ranking metrics rec2")
            roc_auc2_ += metrics.roc_auc(is_relevant)
            precision2_ += metrics.precision(is_relevant)
            recall2_ += metrics.recall(is_relevant, relevant_items)
            map2_ += metrics.map(is_relevant, relevant_items)
            mrr2_ += metrics.rr(is_relevant)
            ndcg2_ += metrics.ndcg(ranked_items_2, relevant_items, relevance=self.test_set[test_user].data, at=at)

            # Increase the number of evaluations performed.
            n_eval += 1

        # Recommender evaluation.
        self.rmse[0].append(rmse_ / n_eval)
        self.roc_auc[0].append(roc_auc_ / n_eval)
        self.precision[0].append(precision_ / n_eval)
        self.recall[0].append(recall_ / n_eval)
        self.map[0].append(map_ / n_eval)
        self.mrr[0].append(mrr_ / n_eval)
        self.ndcg[0].append(ndcg_ / n_eval)

        self.rmse[1].append(rmse2_ / n_eval)
        self.roc_auc[1].append(roc_auc2_ / n_eval)
        self.precision[1].append(precision2_ / n_eval)
        self.recall[1].append(recall2_ / n_eval)
        self.map[1].append(map2_ / n_eval)
        self.mrr[1].append(mrr2_ / n_eval)
        self.ndcg[1].append(ndcg2_ / n_eval)

    def log_all(self):
        for index in range(len(self.rmse)):
            self.log_by_index(index)

    def log_by_index(self,index,rec_1, rec_2):
        filepath = self.results_path + self.results_file
        data_utils.results_to_file(filepath=filepath,
                        cotraining=self.cotraining,
                        iterations=index,
                        recommender1=rec_1,
                        evaluation1=[self.rmse[0][index], self.roc_auc[0][index], self.precision[0][index], self.recall[0][index], self.map[0][index], self.mrr[0][index], self.ndcg[0][index]],
                        at=self.at
                        )

        data_utils.results_to_file(filepath=filepath,
                        cotraining=self.cotraining,
                        iterations=index,
                        recommender1=rec_2,
                        evaluation1=[self.rmse[1][index], self.roc_auc[1][index], self.precision[1][index], self.recall[1][index], self.map[1][index], self.mrr[1][index], self.ndcg[1][index]],
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

    def plot_all(self,rec_index,rec):
        # plot with various axes scales
        random_eval = self.baselines_eval['random']
        ge_eval = self.baselines_eval['global_effects']
        tp_eval = self.baselines_eval['top_pop']

        n_iter = len(self.rmse[rec_index])
        iterations = np.arange(n_iter)

        # rmse
        plt.figure(1)
        rec_plot, = plt.plot(iterations,self.rmse[rec_index], 'b-', label=rec.short_str())
        random_plot, = plt.plot(iterations, [random_eval[0]]*n_iter, 'k-', label='Random')
        ge_plot, = plt.plot(iterations, [ge_eval[0]]*n_iter, 'r-', label='Global Effects')
        # tp_plot, = plt.plot(iterations, [tp_eval[0]]*n_iter, 'y-', label='Top Popular')
        plt.title('RMSE for {} recommender'.format(rec.short_str()))
        plt.ylabel('RMSE')
        plt.xlabel('Iterations')
        plt.legend(handles=[rec_plot,random_plot,ge_plot])
        plt.grid(True)
        savepath = self.results_path + "RMSE_{}iter_{}.png".format(n_iter,rec.short_str())
        plt.savefig(savepath)
        plt.clf()

        # roc_auc
        plt.figure(2)
        rec_plot, = plt.plot(iterations,self.roc_auc[rec_index], 'b-', label=rec.short_str())
        random_plot, = plt.plot(iterations, [random_eval[1]]*n_iter, 'k-', label='Random')
        ge_plot, = plt.plot(iterations, [ge_eval[1]]*n_iter, 'r-', label='Global Effects')
        tp_plot, = plt.plot(iterations, [tp_eval[1]]*n_iter, 'y-', label='Top Popular')
        plt.title('ROC-AUC@{} for {} recommender'.format(self.at, rec.short_str()))
        plt.ylabel('ROC-AUC')
        plt.xlabel('Iterations')
        plt.legend(handles=[rec_plot,random_plot,ge_plot,tp_plot])
        plt.grid(True)
        savepath = self.results_path + "ROC-AUC_{}iter_{}.png".format(n_iter,rec.short_str())
        plt.savefig(savepath)
        plt.clf()

        # precision
        plt.figure(3)
        rec_plot, = plt.plot(iterations,self.precision[rec_index], 'b-', label=rec.short_str())
        random_plot, = plt.plot(iterations, [random_eval[2]]*n_iter, 'k-', label='Random')
        ge_plot, = plt.plot(iterations, [ge_eval[2]]*n_iter, 'r-', label='Global Effects')
        tp_plot, = plt.plot(iterations, [tp_eval[2]]*n_iter, 'y-', label='Top Popular')
        plt.title('Precision@{} for {} recommender'.format(self.at, rec.short_str()))
        plt.ylabel('Precision')
        plt.xlabel('Iterations')
        plt.legend(handles=[rec_plot,random_plot,ge_plot,tp_plot])
        plt.grid(True)
        savepath = self.results_path + "Precision_{}iter_{}.png".format(n_iter,rec.short_str())
        plt.savefig(savepath)
        plt.clf()

        # recall
        plt.figure(4)
        rec_plot, =  plt.plot(iterations,self.recall[rec_index], 'b-', label=rec.short_str())
        random_plot, = plt.plot(iterations, [random_eval[3]]*n_iter, 'k-', label='Random')
        ge_plot, = plt.plot(iterations, [ge_eval[3]]*n_iter, 'r-', label='Global Effects')
        tp_plot, = plt.plot(iterations, [tp_eval[3]]*n_iter, 'y-', label='Top Popular')
        plt.title('Recall@{} for {} recommender'.format(self.at, rec.short_str()))
        plt.ylabel('Recall')
        plt.xlabel('Iterations')
        plt.legend(handles=[rec_plot,random_plot,ge_plot,tp_plot])
        plt.grid(True)
        savepath = self.results_path + "Recall_{}iter_{}.png".format(n_iter,rec.short_str())
        plt.savefig(savepath)
        plt.clf()

        # map
        plt.figure(5)
        rec_plot, = plt.plot(iterations,self.map[rec_index], 'b-', label=rec.short_str())
        random_plot, = plt.plot(iterations, [random_eval[4]]*n_iter, 'k-', label='Random')
        ge_plot, = plt.plot(iterations, [ge_eval[4]]*n_iter, 'r-', label='Global Effects')
        tp_plot, = plt.plot(iterations, [tp_eval[4]]*n_iter, 'y-', label='Top Popular')
        plt.title('MAP@{} for {} recommender'.format(self.at, rec.short_str()))
        plt.ylabel('MAP')
        plt.xlabel('Iterations')
        plt.legend(handles=[rec_plot,random_plot,ge_plot,tp_plot])
        plt.grid(True)
        savepath = self.results_path + "MAP_{}iter_{}.png".format(n_iter,rec.short_str())
        plt.savefig(savepath)
        plt.clf()

        # mrr
        plt.figure(6)
        rec_plot, = plt.plot(iterations,self.mrr[rec_index], 'b-', label=rec.short_str())
        random_plot, = plt.plot(iterations, [random_eval[5]]*n_iter, 'k-', label='Random')
        ge_plot, = plt.plot(iterations, [ge_eval[5]]*n_iter, 'r-', label='Global Effects')
        tp_plot, = plt.plot(iterations, [tp_eval[5]]*n_iter, 'y-', label='Top Popular')
        plt.title('MRR@{} for {} recommender'.format(self.at, rec.short_str()))
        plt.ylabel('MRR')
        plt.xlabel('Iterations')
        plt.legend(handles=[rec_plot,random_plot,ge_plot,tp_plot])
        plt.grid(True)
        savepath = self.results_path + "MRR_{}iter_{}.png".format(n_iter,rec.short_str())
        plt.savefig(savepath)
        plt.clf()

        # ndcg
        plt.figure(7)
        rec_plot, = plt.plot(iterations,self.ndcg[rec_index], 'b-', label=rec.short_str())
        random_plot, = plt.plot(iterations, [random_eval[6]]*n_iter, 'k-', label='Random')
        ge_plot, = plt.plot(iterations, [ge_eval[6]]*n_iter, 'r-', label='Global Effects')
        tp_plot, = plt.plot(iterations, [tp_eval[6]]*n_iter, 'y-', label='Top Popular')
        plt.title('NDCG@{} for {} recommender'.format(self.at, rec.short_str()))
        plt.ylabel('NDCG')
        plt.xlabel('Iterations')
        plt.legend(handles=[rec_plot,random_plot,ge_plot,tp_plot])
        plt.grid(True)
        savepath = self.results_path + "NDCG_{}iter_{}.png".format(n_iter,rec.short_str())
        plt.savefig(savepath)
        plt.clf()

    def plot_all_recommenders(self, rec_1, rec_2):
        # pdb.set_trace()
        random_eval = self.baselines_eval['random']
        ge_eval = self.baselines_eval['global_effects']
        tp_eval = self.baselines_eval['top_pop']

        iterations = np.arange(len(self.rmse[0]))
        n_iters = len(iterations)
        # Plot each metric in a different file.
        # RMSE.
        plt.figure(1)
        plt.title('RMSE between the recommenders.')
        # self_plot, = plt.plot(iterations, self.rmse,  'r-', label=self.recommender.short_str())
        eval1_plot, = plt.plot(iterations, self.rmse[0], 'b-', label=rec_1.short_str())
        eval2_plot, = plt.plot(iterations, self.rmse[1], 'g-', label=rec_2.short_str())
        random_plot, = plt.plot(iterations, [random_eval[0]]*n_iters, 'k-', label='Random')
        ge_plot, = plt.plot(iterations, [ge_eval[0]]*n_iters, 'r-', label='Global Effects')
        # tp_plot, = plt.plot(iterations, [tp_eval[0]]*n_iters, 'y-', label='Top Popular')
        plt.ylabel('RMSE')
        plt.xlabel('Iterations')
        plt.legend(handles=[eval1_plot,eval2_plot,random_plot,ge_plot])
        plt.grid(True)
        savepath = self.results_path + "Together_RMSE_{}iter.png".format(n_iters)
        plt.savefig(savepath)
        plt.clf()

        # ROC-AUC.
        plt.figure(2)
        plt.title('ROC-AUC@{} between the recommenders.'.format(self.at))
        # self_plot, = plt.plot(iterations, self.roc_auc,  'r-', label=self.recommender.short_str())
        eval1_plot, = plt.plot(iterations, self.roc_auc[0], 'b-', label=rec_1.short_str())
        eval2_plot, = plt.plot(iterations, self.roc_auc[1], 'g-', label=rec_2.short_str())
        random_plot, = plt.plot(iterations, [random_eval[1]]*n_iters, 'k-', label='Random')
        ge_plot, = plt.plot(iterations, [ge_eval[1]]*n_iters, 'r-', label='Global Effects')
        tp_plot, = plt.plot(iterations, [tp_eval[1]]*n_iters, 'y-', label='Top Popular')
        plt.ylabel('ROC-AUC')
        plt.xlabel('Iterations')
        plt.legend(handles=[eval1_plot,eval2_plot,random_plot,ge_plot,tp_plot])
        plt.grid(True)
        savepath = self.results_path + "Together_ROC-AUC_{}iter.png".format(n_iters)
        plt.savefig(savepath)
        plt.clf()

        # Precision
        plt.figure(3)
        plt.title('Precision@{} between the recommenders.'.format(self.at))
        # self_plot, = plt.plot(iterations, self.precision,  'r-', label=self.recommender.short_str())
        eval1_plot, = plt.plot(iterations, self.precision[0], 'b-', label=rec_1.short_str())
        eval2_plot, = plt.plot(iterations, self.precision[1], 'g-', label=rec_2.short_str())
        random_plot, = plt.plot(iterations, [random_eval[2]]*n_iters, 'k-', label='Random')
        ge_plot, = plt.plot(iterations, [ge_eval[2]]*n_iters, 'r-', label='Global Effects')
        tp_plot, = plt.plot(iterations, [tp_eval[2]]*n_iters, 'y-', label='Top Popular')
        plt.ylabel('Precision')
        plt.xlabel('Iterations')
        plt.legend(handles=[eval1_plot,eval2_plot,random_plot,ge_plot,tp_plot])
        plt.grid(True)
        savepath = self.results_path + "Together_Precision_{}iter.png".format(n_iters)
        plt.savefig(savepath)
        plt.clf()

        # Recall
        plt.figure(4)
        plt.title('Recall@{} between the recommenders.'.format(self.at))
        # self_plot, = plt.plot(iterations, self.recall,  'r-', label=self.recommender.short_str())
        eval1_plot, = plt.plot(iterations, self.recall[0], 'b-', label=rec_1.short_str())
        eval2_plot, = plt.plot(iterations, self.recall[1], 'g-', label=rec_2.short_str())
        random_plot, = plt.plot(iterations, [random_eval[3]]*n_iters, 'k-', label='Random')
        ge_plot, = plt.plot(iterations, [ge_eval[3]]*n_iters, 'r-', label='Global Effects')
        tp_plot, = plt.plot(iterations, [tp_eval[3]]*n_iters, 'y-', label='Top Popular')
        plt.ylabel('Recall')
        plt.xlabel('Iterations')
        plt.legend(handles=[eval1_plot,eval2_plot,random_plot,ge_plot,tp_plot])
        plt.grid(True)
        savepath = self.results_path + "Together_Recall_{}iter.png".format(n_iters)
        plt.savefig(savepath)
        plt.clf()

        # MAP
        plt.figure(5)
        plt.title('MAP@{} between the recommenders.'.format(self.at))
        # self_plot, = plt.plot(iterations, self.map,  'r-', label=self.recommender.short_str())
        eval1_plot, = plt.plot(iterations, self.map[0], 'b-', label=rec_1.short_str())
        eval2_plot, = plt.plot(iterations, self.map[1], 'g-', label=rec_2.short_str())
        random_plot, = plt.plot(iterations, [random_eval[4]]*n_iters, 'k-', label='Random')
        ge_plot, = plt.plot(iterations, [ge_eval[4]]*n_iters, 'r-', label='Global Effects')
        tp_plot, = plt.plot(iterations, [tp_eval[4]]*n_iters, 'y-', label='Top Popular')
        plt.ylabel('MAP')
        plt.xlabel('Iterations')
        plt.grid(True)
        plt.legend(handles=[eval1_plot,eval2_plot,random_plot,ge_plot,tp_plot])
        savepath = self.results_path + "Together_MAP_{}iter.png".format(n_iters)
        plt.savefig(savepath)
        plt.clf()

        # MRR
        plt.figure(6)
        plt.title('MRR@{} between the recommenders.'.format(self.at))
        # self_plot, = plt.plot(iterations, self.mrr, 'r-', label=self.recommender.short_str())
        eval1_plot, = plt.plot(iterations, self.mrr[0], 'b-', label=rec_1.short_str())
        eval2_plot, = plt.plot(iterations, self.mrr[1], 'g-', label=rec_2.short_str())
        random_plot, = plt.plot(iterations, [random_eval[5]]*n_iters, 'k-', label='Random')
        ge_plot, = plt.plot(iterations, [ge_eval[5]]*n_iters, 'r-', label='Global Effects')
        tp_plot, = plt.plot(iterations, [tp_eval[5]]*n_iters, 'y-', label='Top Popular')
        plt.ylabel('MRR')
        plt.xlabel('Iterations')
        plt.legend(handles=[eval1_plot,eval2_plot,random_plot,ge_plot,tp_plot])
        plt.grid(True)
        savepath = self.results_path + "Together_MRR_{}iter.png".format(n_iters)
        plt.savefig(savepath)
        plt.clf()

        # NDCG
        plt.figure(7)
        plt.title('NDCG@{} between the recommenders.'.format(self.at))
        # self_plot, = plt.plot(iterations, self.ndcg, 'r-', label=self.recommender.short_str())
        eval1_plot, = plt.plot(iterations, self.ndcg[0], 'b-', label=rec_1.short_str())
        eval2_plot, = plt.plot(iterations, self.ndcg[1], 'g-', label=rec_2.short_str())
        random_plot, = plt.plot(iterations, [random_eval[6]]*n_iters, 'k-', label='Random')
        ge_plot, = plt.plot(iterations, [ge_eval[6]]*n_iters, 'r-', label='Global Effects')
        tp_plot, = plt.plot(iterations, [tp_eval[6]]*n_iters, 'y-', label='Top Popular')
        plt.ylabel('NDCG')
        plt.xlabel('Iterations')
        plt.legend(handles=[eval1_plot,eval2_plot,random_plot,ge_plot,tp_plot])
        plt.grid(True)
        savepath = self.results_path + "Together_NDCG_{}iter.png".format(n_iters)
        plt.savefig(savepath)
        plt.clf()
