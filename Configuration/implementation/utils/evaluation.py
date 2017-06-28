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
import implementation.utils.metrics as metrics
import implementation.utils.data_utils as data_utils

class Evaluation(object):
    """ EVALUATION class for RecSys"""

    def __init__(self, recommender, results_path, nusers, test_set, val_set = None, at = 10, co_training=False):
        '''
            Args:
                * recommender: A Recommender Class object that represents the first
                         recommender.
                * nusers: The number of users to evaluate. It represents user indices.
        '''
        super(Evaluation, self).__init__()
        self.recommender = recommender
        self.results_path = results_path
        self.nusers = nusers
        self.test_set = test_set
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
            self.recommender.__str__)


    def eval(self, train_set):
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
                recommended_items = self.recommender.recommend(user_id=test_user, exclude_seen=True)
                # evaluate the recommendation list with ranking metrics ONLY
                rmse_ += metrics.rmse(recommended_items, relevant_items)
                roc_auc_ += metrics.roc_auc(recommended_items, relevant_items)
                precision_ += metrics.precision(recommended_items, relevant_items, at=at)
                recall_ += metrics.recall(recommended_items, relevant_items, at=at)
                map_ += metrics.map(recommended_items, relevant_items, at=at)
                mrr_ += metrics.rr(recommended_items, relevant_items, at=at)
                ndcg_ += metrics.ndcg(recommended_items, relevant_items, relevance=self.test_set[test_user].data, at=at)

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
        data_utils.results_to_file(filepath=self.results_path,
                        evaluation_type="holdout at 80%",
                        cotraining=self.cotraining,
                        iterations=index,
                        recommender1=self.recommender,
                        evaluation1=[self.rmse[index], self.roc_auc[index], self.precision[index], self.recall[index], self.map[index], self.mrr[index], self.ndcg[index]],
                        at=self.at
                        )

    def plot_all(self,number_figure):
        # plot with various axes scales
        plt.figure(number_figure)
        plt.title(self.recommender.__str__())

        # rmse
        plt.subplot(4,2,1)
        plt.plot(self.rmse)
        plt.ylabel('RMSE')
        plt.xlabel('Iterations')
        plt.grid(True)

        # roc_auc
        plt.subplot(4,2,2)
        plt.plot(self.roc_auc)
        plt.ylabel('ROC-AUC@{}'.format(self.at))
        plt.xlabel('Iterations')
        plt.grid(True)

        # precision
        plt.subplot(4,2,3)
        plt.plot(self.precision)
        plt.ylabel('PRECISION@{}'.format(self.at))
        plt.xlabel('Iterations')
        plt.grid(True)

        # recall
        plt.subplot(4,2,4)
        plt.plot(self.recall)
        plt.ylabel('RECALL@{}'.format(self.at))
        plt.xlabel('Iterations')
        plt.grid(True)

        # map
        plt.subplot(4,2,5)
        plt.plot(self.map)
        plt.ylabel('MAP@{}'.format(self.at))
        plt.xlabel('Iterations')
        plt.grid(True)

        # mrr
        plt.subplot(4,2,6)
        plt.plot(self.mrr)
        plt.ylabel('MRR@{}'.format(self.at))
        plt.xlabel('Iterations')
        plt.grid(True)

        # ndcg
        plt.subplot(4,2,7)
        plt.plot(self.ndcg)
        plt.ylabel('NDCG@{}'.format(self.at))
        plt.xlabel('Iterations')
        plt.grid(True)

        plt.show()
