train_set_1'''
Politecnico di Milano.
evaluation.py

Description: This file contains the definition and implementation of a evaluation
             metrics for RecSys under a module.

Modified by Fernando Pérez.

Last modified on 25/03/2017.
'''

import random as random

import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import implementation.utils.metrics as metrics
import implementation.utils.data_utils as data_utils

import pdb

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
        self.rmse = None
        self.roc_auc = None
        self.precision = None
        self.recall = None
        self.map = None
        self.mrr = None
        self.ndcg = None

    def __str__(self):
        return "Evaluation(Rec={}\n)".format(
            self.recommender.__str__())


    def eval(self, rec_1, rec_2):
        # if (not isinstance(train_set_1,sps.csr_matrix)):
        #     train_set_1 = train_set_1.tocsr().astype(np.float32)
        #
        # if (not isinstance(train_set_2,sps.csr_matrix)):
        #     train_set_2 = train_set_2.tocsr().astype(np.float32)

        at = self.at
        n_eval = 0
        self.rmse, self.roc_auc, self.precision, self.recall, self.map, self.mrr, self.ndcg = (list(), list()), (list(), list()), (list(), list()), (list(), list()), (list(), list()), (list(), list()), (list(), list())
        rmse_, roc_auc_, precision_, recall_, map_, mrr_, ndcg_ = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        rmse2_, roc_auc2_, precision2_, recall2_, map2_, mrr2_, ndcg2_ = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        row_indices, _ = self.test_set.nonzero() # users with ratings in the test set. nonzero returns a tuple, the first element are the rows.
        relevant_users = np.unique(row_indices) # In this way we only consider users with ratings in the test set and not ALL the users.
        for test_user in relevant_users:
            # Getting user_profile by it's rated items (relevant_items) in the test.
            relevant_items = self.test_set[test_user].indices

            # Getting user profile given the train set.
            # user_profile = train_set[test_user]

            # Recommender 1.
            # recommender recommendation.
            # this will rank self.at (n) items and will predict the score for the relevant items.
            ranked_items = rec_1.recommend(user_id=test_user, n=self.at, exclude_seen=True)
            predicted_relevant_items = rec_1.predict(user_id=test_user, rated_indices=relevant_items)

            # evaluate the recommendation list with RMSE and ranking metrics.
            rmse_ += metrics.rmse(predicted_relevant_items, self.test_set[test_user,relevant_items].toarray())
            roc_auc_ += metrics.roc_auc(ranked_items, relevant_items)
            precision_ += metrics.precision(ranked_items, relevant_items, at=at)
            recall_ += metrics.recall(ranked_items, relevant_items, at=at)
            map_ += metrics.map(ranked_items, relevant_items, at=at)
            mrr_ += metrics.rr(ranked_items, relevant_items, at=at)
            ndcg_ += metrics.ndcg(ranked_items, relevant_items, relevance=self.test_set[test_user].data, at=at)


            # Recommender 2.
            # recommender recommendation.
            # this will rank self.at (n) items and will predict the score for the relevant items.
            ranked_items_2 = rec_2.recommend(user_id=test_user, n=self.at, exclude_seen=True)
            predicted_relevant_items_2 = rec_2.predict(user_id=test_user, rated_indices=relevant_items)

            # evaluate the recommendation list with RMSE and ranking metrics.
            rmse2_ += metrics.rmse(predicted_relevant_items_2, self.test_set[test_user,relevant_items].toarray())
            roc_auc2_ += metrics.roc_auc(ranked_items_2, relevant_items)
            precision2_ += metrics.precision(ranked_items_2, relevant_items, at=at)
            recall2_ += metrics.recall(ranked_items_2, relevant_items, at=at)
            map2_ += metrics.map(ranked_items_2, relevant_items, at=at)
            mrr2_ += metrics.rr(ranked_items_2, relevant_items, at=at)
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

    def plot_all(self,index,rec):
        # plot with various axes scales
        iterations = len(self.rmse[index])
        # rmse
        plt.figure(1)
        plt.plot(self.rmse[index])
        plt.title('RMSE for {} recommender'.format(rec.short_str()))
        plt.ylabel('RMSE')
        plt.xlabel('Iterations')
        plt.grid(True)
        savepath = self.results_path + "RMSE_{}iter_{}.png".format(iterations,rec.short_str())
        plt.savefig(savepath)
        plt.clf()

        # roc_auc
        plt.figure(2)
        plt.plot(self.roc_auc[index])
        plt.title('ROC-AUC@{} for {} recommender'.format(self.at, rec.short_str()))
        plt.ylabel('ROC-AUC')
        plt.xlabel('Iterations')
        plt.grid(True)
        savepath = self.results_path + "ROC-AUC_{}iter_{}.png".format(iterations,rec.short_str())
        plt.savefig(savepath)
        plt.clf()

        # precision
        plt.figure(3)
        plt.plot(self.precision[index])
        plt.title('Precision@{} for {} recommender'.format(self.at, rec.short_str()))
        plt.ylabel('Precision')
        plt.xlabel('Iterations')
        plt.grid(True)
        savepath = self.results_path + "Precision_{}iter_{}.png".format(iterations,rec.short_str())
        plt.savefig(savepath)
        plt.clf()

        # recall
        plt.figure(4)
        plt.plot(self.recall[index])
        plt.title('Recall@{} for {} recommender'.format(self.at, rec.short_str()))
        plt.ylabel('Recall')
        plt.xlabel('Iterations')
        plt.grid(True)
        savepath = self.results_path + "Recall_{}iter_{}.png".format(iterations,rec.short_str())
        plt.savefig(savepath)
        plt.clf()

        # map
        plt.figure(5)
        plt.plot(self.map[index])
        plt.title('MAP@{} for {} recommender'.format(self.at, rec.short_str()))
        plt.ylabel('MAP')
        plt.xlabel('Iterations')
        plt.grid(True)
        savepath = self.results_path + "MAP_{}iter_{}.png".format(iterations,rec.short_str())
        plt.savefig(savepath)
        plt.clf()

        # mrr
        plt.figure(6)
        plt.plot(self.mrr[index])
        plt.title('MRR@{} for {} recommender'.format(self.at, rec.short_str()))
        plt.ylabel('MRR')
        plt.xlabel('Iterations')
        plt.grid(True)
        savepath = self.results_path + "MRR_{}iter_{}.png".format(iterations,rec.short_str())
        plt.savefig(savepath)
        plt.clf()

        # ndcg
        plt.figure(7)
        plt.plot(self.ndcg[index])
        plt.title('NDCG@{} for {} recommender'.format(self.at, rec.short_str()))
        plt.ylabel('NDCG')
        plt.xlabel('Iterations')
        plt.grid(True)
        savepath = self.results_path + "NDCG_{}iter_{}.png".format(iterations,rec.short_str())
        plt.savefig(savepath)
        plt.clf()

    def plot_all_recommenders(self, eval1, eval2):
        iterations = np.arange(len(self.rmse))
        n_iters = len(iterations)
        # Plot each metric in a different file.
        # RMSE.
        plt.figure(1)
        plt.title('RMSE between the recommenders.')
        self_plot, = plt.plot(iterations, self.rmse,  'r-', label=self.recommender.short_str())
        eval1_plot, = plt.plot(iterations, eval1.rmse, 'b-', label=eval1.recommender.short_str())
        eval2_plot, = plt.plot(iterations, eval2.rmse, 'g-', label=eval2.recommender.short_str())
        plt.ylabel('RMSE')
        plt.xlabel('Iterations')
        plt.legend(handles=[self_plot,eval1_plot,eval2_plot])
        plt.grid(True)
        savepath = self.results_path + "Together_RMSE_{}iter_{}.png".format(n_iters,self.recommender.short_str())
        plt.savefig(savepath)
        plt.clf()

        # ROC-AUC.
        plt.figure(2)
        plt.title('ROC-AUC@{} between the recommenders.'.format(self.at))
        self_plot, = plt.plot(iterations, self.roc_auc,  'r-', label=self.recommender.short_str())
        eval1_plot, = plt.plot(iterations, eval1.roc_auc, 'b-', label=eval1.recommender.short_str())
        eval2_plot, = plt.plot(iterations, eval2.roc_auc, 'g-', label=eval2.recommender.short_str())
        plt.ylabel('ROC-AUC')
        plt.xlabel('Iterations')
        plt.legend(handles=[self_plot,eval1_plot,eval2_plot])
        plt.grid(True)
        savepath = self.results_path + "Together_ROC-AUC_{}iter_{}.png".format(n_iters,self.recommender.short_str())
        plt.savefig(savepath)
        plt.clf()

        # Precision
        plt.figure(3)
        plt.title('Precision@{} between the recommenders.'.format(self.at))
        self_plot, = plt.plot(iterations, self.precision,  'r-', label=self.recommender.short_str())
        eval1_plot, = plt.plot(iterations, eval1.precision, 'b-', label=eval1.recommender.short_str())
        eval2_plot, = plt.plot(iterations, eval2.precision, 'g-', label=eval2.recommender.short_str())
        plt.ylabel('Precision')
        plt.xlabel('Iterations')
        plt.legend(handles=[self_plot,eval1_plot,eval2_plot])
        plt.grid(True)
        savepath = self.results_path + "Together_Precision_{}iter_{}.png".format(n_iters,self.recommender.short_str())
        plt.savefig(savepath)
        plt.clf()

        # Recall
        plt.figure(4)
        plt.title('Recall@{} between the recommenders.'.format(self.at))
        self_plot, = plt.plot(iterations, self.recall,  'r-', label=self.recommender.short_str())
        eval1_plot, = plt.plot(iterations, eval1.recall, 'b-', label=eval1.recommender.short_str())
        eval2_plot, = plt.plot(iterations, eval2.recall, 'g-', label=eval2.recommender.short_str())
        plt.ylabel('Recall')
        plt.xlabel('Iterations')
        plt.legend(handles=[self_plot,eval1_plot,eval2_plot])
        plt.grid(True)
        savepath = self.results_path + "Together_Recall_{}iter_{}.png".format(n_iters,self.recommender.short_str())
        plt.savefig(savepath)
        plt.clf()

        # MAP
        plt.figure(5)
        plt.title('MAP@{} between the recommenders.'.format(self.at))
        self_plot, = plt.plot(iterations, self.map,  'r-', label=self.recommender.short_str())
        eval1_plot, = plt.plot(iterations, eval1.map, 'b-', label=eval1.recommender.short_str())
        eval2_plot, = plt.plot(iterations, eval2.map, 'g-', label=eval2.recommender.short_str())
        plt.ylabel('MAP')
        plt.xlabel('Iterations')
        plt.grid(True)
        plt.legend(handles=[self_plot,eval1_plot,eval2_plot])
        savepath = self.results_path + "Together_MAP_{}iter_{}.png".format(n_iters,self.recommender.short_str())
        plt.savefig(savepath)
        plt.clf()

        # MRR
        plt.figure(6)
        plt.title('MRR@{} between the recommenders.'.format(self.at))
        self_plot, = plt.plot(iterations, self.mrr, 'r-', label=self.recommender.short_str())
        eval1_plot, = plt.plot(iterations, eval1.mrr, 'b-', label=eval1.recommender.short_str())
        eval2_plot, = plt.plot(iterations, eval2.mrr, 'g-', label=eval2.recommender.short_str())
        plt.ylabel('MRR')
        plt.xlabel('Iterations')
        plt.legend(handles=[self_plot,eval1_plot,eval2_plot])
        plt.grid(True)
        savepath = self.results_path + "Together_MRR_{}iter_{}.png".format(n_iters,self.recommender.short_str())
        plt.savefig(savepath)
        plt.clf()

        # NDCG
        plt.figure(7)
        plt.title('NDCG@{} between the recommenders.'.format(self.at))
        self_plot, = plt.plot(iterations, self.ndcg, 'r-', label=self.recommender.short_str())
        eval1_plot, = plt.plot(iterations, eval1.ndcg, 'b-', label=eval1.recommender.short_str())
        eval2_plot, = plt.plot(iterations, eval2.ndcg, 'g-', label=eval2.recommender.short_str())
        plt.ylabel('NDCG')
        plt.xlabel('Iterations')
        plt.legend(handles=[self_plot,eval1_plot,eval2_plot])
        plt.grid(True)
        savepath = self.results_path + "Together_NDCG_{}iter_{}.png".format(n_iters,self.recommender.short_str())
        plt.savefig(savepath)
        plt.clf()
