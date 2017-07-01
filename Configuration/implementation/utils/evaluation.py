'''
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

    def __init__(self, recommender, results_path, results_file, nusers, test_set, val_set = None, at = 10, co_training=False):
        '''
            Args:
                * recommender: A Recommender Class object that represents the first
                         recommender.
                * nusers: The number of users to evaluate. It represents user indices.
        '''
        super(Evaluation, self).__init__()
        self.recommender = recommender
        self.test_set = test_set
        self.results_path = results_path
        self.results_file = results_file
        self.nusers = nusers
        self.val_set = val_set
        self.at = at
        self.rmse = list()
        self.roc_auc = list()
        self.precision = list()
        self.recall = list()
        self.map = list()
        self.mrr = list()
        self.ndcg = list()
        self.cotraining = co_training


    def __str__(self):
        return "Evaluation(Rec={}\n)".format(
            self.recommender.__str__())


    def eval(self,train_set):
        if (not isinstance(train_set,sps.csr_matrix)):
            train_set = train_set.tocsr().astype(np.float32)

        at = self.at
        n_eval = 0
        rmse_, roc_auc_, precision_, recall_, map_, mrr_, ndcg_ = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        for test_user in range(self.nusers):
            user_profile = train_set[test_user]
            relevant_items = self.test_set[test_user].indices
            if len(relevant_items) > 0:
                n_eval += 1

                # recommender recommendation.
                # this will rank **all** items
                ranked_items = self.recommender.recommend(user_id=test_user, n=self.at, exclude_seen=True)
                predicted_relevant_items = self.recommender.predict(user_id=test_user, rated_indices=relevant_items)
                # evaluate the recommendation list with ranking metrics ONLY
                rmse_ += metrics.rmse(predicted_relevant_items, self.test_set[test_user,relevant_items].toarray())
                roc_auc_ += metrics.roc_auc(ranked_items, relevant_items)
                precision_ += metrics.precision(ranked_items, relevant_items, at=at)
                recall_ += metrics.recall(ranked_items, relevant_items, at=at)
                map_ += metrics.map(ranked_items, relevant_items, at=at)
                mrr_ += metrics.rr(ranked_items, relevant_items, at=at)
                ndcg_ += metrics.ndcg(ranked_items, relevant_items, relevance=self.test_set[test_user].data, at=at)

        # Recommender evaluations
        self.rmse.append(rmse_ / n_eval)
        self.roc_auc.append(roc_auc_ / n_eval)
        self.precision.append(precision_ / n_eval)
        self.recall.append(recall_ / n_eval)
        self.map.append(map_ / n_eval)
        self.mrr.append(mrr_ / n_eval)
        self.ndcg.append(ndcg_ / n_eval)

    def log_all(self):
        for index in range(len(self.rmse)):
            self.log_by_index(index)

    def log_by_index(self,index):
        filepath = self.results_path + self.results_file
        data_utils.results_to_file(filepath=filepath,
                        cotraining=self.cotraining,
                        iterations=index,
                        recommender1=self.recommender,
                        evaluation1=[self.rmse[index], self.roc_auc[index], self.precision[index], self.recall[index], self.map[index], self.mrr[index], self.ndcg[index]],
                        at=self.at
                        )

    def plot_all(self):
        # plot with various axes scales

        # rmse
        plt.figure(1)
        plt.plot(self.rmse)
        plt.title('RMSE for {} recommender'.format(self.recommender.short_str()))
        plt.ylabel('RMSE')
        plt.xlabel('Iterations')
        plt.grid(True)
        savepath = self.results_path + "RMSE_{}iter_{}.png".format(len(self.rmse),self.recommender.short_str())
        plt.savefig(savepath)
        plt.clf()

        # roc_auc
        plt.figure(2)
        plt.plot(self.roc_auc)
        plt.title('ROC-AUC@{} for {} recommender'.format(self.at, self.recommender.short_str()))
        plt.ylabel('ROC-AUC')
        plt.xlabel('Iterations')
        plt.grid(True)
        savepath = self.results_path + "ROC-AUC_{}iter_{}.png".format(len(self.rmse),self.recommender.short_str())
        plt.savefig(savepath)
        plt.clf()

        # precision
        plt.figure(3)
        plt.plot(self.precision)
        plt.title('Precision@{} for {} recommender'.format(self.at, self.recommender.short_str()))
        plt.ylabel('Precision')
        plt.xlabel('Iterations')
        plt.grid(True)
        savepath = self.results_path + "Precision_{}iter_{}.png".format(len(self.rmse),self.recommender.short_str())
        plt.savefig(savepath)
        plt.clf()

        # recall
        plt.figure(4)
        plt.plot(self.recall)
        plt.title('Recall@{} for {} recommender'.format(self.at, self.recommender.short_str()))
        plt.ylabel('Recall')
        plt.xlabel('Iterations')
        plt.grid(True)
        savepath = self.results_path + "Recall_{}iter_{}.png".format(len(self.rmse),self.recommender.short_str())
        plt.savefig(savepath)
        plt.clf()

        # map
        plt.figure(5)
        plt.plot(self.map)
        plt.title('MAP@{} for {} recommender'.format(self.at, self.recommender.short_str()))
        plt.ylabel('MAP')
        plt.xlabel('Iterations')
        plt.grid(True)
        savepath = self.results_path + "MAP_{}iter_{}.png".format(len(self.rmse),self.recommender.short_str())
        plt.savefig(savepath)
        plt.clf()

        # mrr
        plt.figure(6)
        plt.plot(self.mrr)
        plt.title('MRR@{} for {} recommender'.format(self.at, self.recommender.short_str()))
        plt.ylabel('MRR')
        plt.xlabel('Iterations')
        plt.grid(True)
        savepath = self.results_path + "MRR_{}iter_{}.png".format(len(self.rmse),self.recommender.short_str())
        plt.savefig(savepath)
        plt.clf()

        # ndcg
        plt.figure(7)
        plt.plot(self.ndcg)
        plt.title('NDCG@{} for {} recommender'.format(self.at, self.recommender.short_str()))
        plt.ylabel('NDCG')
        plt.xlabel('Iterations')
        plt.grid(True)
        savepath = self.results_path + "NDCG_{}iter_{}.png".format(len(self.rmse),self.recommender.short_str())
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
