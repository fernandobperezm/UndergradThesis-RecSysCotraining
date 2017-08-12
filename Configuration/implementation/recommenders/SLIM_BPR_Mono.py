#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 28 June 2017

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
import time
import os
from .Recommender import Recommender
from .Recommender_utils import similarityMatrixTopK, check_matrix
import scipy.sparse as sps
import subprocess

import pdb

class SLIM_BPR_Mono(Recommender):

    # def __init__(self, URM_train, lambda_i = 0.0025, lambda_j = 0.00025, learning_rate = 0.05, topK = False):
    def __init__(self, lambda_i = 0.0025, lambda_j = 0.00025, learning_rate = 0.05, topK = None):
        """
        WARNING: This class needs to save files on disk to call another executable. Neither this class nor the executable
        are thread safe

        :param URM_train:
        :param lambda_i:
        :param lambda_j:
        :param learning_rate:
        :param topK:
        """
        super(SLIM_BPR_Mono, self).__init__()

        # self.URM_train = URM_train
        # self.n_users = URM_train.shape[0]
        # self.n_items = URM_train.shape[1]
        self.lambda_i = lambda_i
        self.lambda_j = lambda_j
        self.learning_rate = learning_rate
        self.normalize = False
        self.sparse_weights = True
        self.topK = topK

        print(os.path.dirname(os.path.abspath(__file__)))
        print(os.getcwd())

        self.basePath = "../../../Datasets/ml10m/"
        self.executablePath = "./item_recommendation"
        self.trainFileName = "SLIM_BPR_Mono_URM_train.csv"
        self.testFileName = "SLIM_BPR_Mono_URM_test.csv"
        self.outputModelName = "SLIM_BPR_Mono_Model.txt"

        # self.basePath = "data/"
        # self.executablePath = "MyMediaLite/bin/item_recommendation"

        # Set permission to execute, code 0775, the o is needed in python to encode octal numbers
        os.chmod(self.executablePath, 0o775)

    def __str__(self):
        return "SLIM_BPR_Mono(lambda_i={},lambda_j={},learning_rate={},topK={})".format(self.lambda_i, self.lambda_j, self.learning_rate, self.topK)

    def short_str(self):
        return "SLIM_BPR_Mono"

    def _get_user_ratings(self, user_id):
        return self.URM_train[user_id]

    def writeTestToFile(self, URM_test):
        if URM_test is not None:
            self.writeSparseToFile(URM_test, open(self.basePath + self.testFileName, "w"))

    def removeTemporaryFiles(self):

        # Remove saved Model and URM

        print("Removing: {}".format(self.basePath + self.trainFileName))
        os.remove(self.basePath + self.trainFileName)

        print("Removing: {}".format(self.basePath + self.trainFileName + ".bin.PosOnlyFeedback"))
        os.remove(self.basePath + self.trainFileName + ".bin.PosOnlyFeedback")

        print("Removing: {}".format(self.basePath + self.outputModelName))
        os.remove(self.basePath + self.outputModelName)

    def writeSparseToFile(self, sparseMatrix, file):
        sparseMatrix = sparseMatrix.tocoo()

        data = sparseMatrix.data
        row = sparseMatrix.row
        col = sparseMatrix.col

        for index in range(len(data)):
            file.write("{},{},{}\n".format(row[index], col[index], data[index]))

            #if (index % 500000 == 0):
            #    print("Processed {} rows of {}".format(index, len(data)))

        file.close()

    def loadModelIntoDenseMatrix(self, filePath):
        SLIMsimilarity = open(filePath, "r")
        SLIMsimilarity.readline()  # program name
        SLIMsimilarity.readline()  # 2.99
        line = SLIMsimilarity.readline()  # size

        n_items_model = int(line.split(" ")[0])

        if n_items_model<self.n_items:
            print("The model file contains less items than the URM_train, it may be that some items do not have interactions.")

        # Shape requires the number of cells, which is the number of items
        self.W = np.zeros((self.n_items,self.n_items), dtype=np.float32)

        numCells = 0
        print("Loading SLIM model")

        for line in SLIMsimilarity:
            numCells += 1
            #if (numCells % 1000000 == 0):
            #    print("Processed {} cells".format(numCells))

            if (len(line)) > 1:
                line = line.split(" ")

                value = line[2].replace("\n", "")

                if not value == "0" and not value == "NaN":
                    row = int(line[0])
                    col = int(line[1])
                    value = float(value)

                    self.W[row, col] = value

        SLIMsimilarity.close()

        self.W = self.W.T

        if (self.topK is not None):
            self.W_sparse = similarityMatrixTopK(self.W, k=self.topK)

            self.sparse_weights = True
            del self.W

        else:
            self.sparse_weights = False


    def loadModelIntoSparseMatrix(self, filePath):

        values, rows, cols = [], [], []

        SLIMsimilarity = open(filePath, "r")
        SLIMsimilarity.readline()  # program name
        SLIMsimilarity.readline()  # 2.99
        line = SLIMsimilarity.readline()  # size

        n_items_model = int(line.split(" ")[0])

        if n_items_model<self.n_items:
            print("The model file contains less items than the URM_train, it may be that some items do not have interactions.")


        numCells = 0
        print("Loading SLIM model")

        for line in SLIMsimilarity:
            numCells += 1
            #if (numCells % 1000000 == 0):
            #    print("Processed {} cells".format(numCells))

            if (len(line)) > 1:
                line = line.split(" ")

                value = line[2].replace("\n", "")

                if not value == "0" and not value == "NaN":
                    rows.append(int(line[0]))
                    cols.append(int(line[1]))
                    values.append(float(value))

        SLIMsimilarity.close()

        self.W_sparse = sps.csr_matrix((values, (rows, cols)), shape=(self.n_items, self.n_items), dtype=np.float32)
        self.W_sparse = self.W_sparse.T
        self.sparse_weights = True

        if self.topK != False:
            self.W_sparse = similarityMatrixTopK(self.W_sparse, k=self.topK)


    def fit(self, URM_train, epochs=30, deleteFiles=False):
        """
        Train SLIM wit BPR. If the model was already trained, overwrites matrix S
        Training is performed via batch gradient descent
        :param epochs:
        :return: -
        """
        #
        self.URM_train = check_matrix(URM_train, format='csr')
        self.n_users, self.n_items = URM_train.shape
        print("Train users: {}, Train items: {}".format(self.n_users, self.n_items))

        recommenderMethod = "BPRSLIM"

        try:
            # If train file already exist, clean all data
            #trainFile = open(self.basePath + self.trainFileName, "r")
            #trainFile.close()
            print("Removing previous SLIM_BPR files")
            self.removeTemporaryFiles()

        except:
            pass

        print("Writing URM_train to {}".format(self.basePath + self.trainFileName))
        self.writeSparseToFile(self.URM_train, open(self.basePath + self.trainFileName, "w"))

        recommenderOptions = 'reg_i={reg_i} reg_j={reg_j} learn_rate={learn_rate} num_iter={num_iter}'.format(
            reg_i=self.lambda_i,
            reg_j=self.lambda_j,
            learn_rate=self.learning_rate,
            num_iter=epochs)

        command = ['"{}"'.format(self.executablePath),
                   '--training-file="{}"'.format(self.basePath + self.trainFileName),
                   #'--test-file="{}"'.format(basePath + testFileName),
                   '--recommender="{}"'.format(recommenderMethod),
                   '--no-id-mapping',
                   #'--save-user-mapping="{}"'.format(basePath + "save-user-mapping"),
                   #'--save-item-mapping="{}"'.format(basePath + "save-item-mapping"),
                   # '--test-ratio=0,25',
                   # '--rating-threshold={}'.format(ratingThreshold),
                   # '--prediction-file="{}"'.format(basePath + outputPredictionName),
                   '--save-model="{}"'.format(self.basePath + self.outputModelName),
                   #'--predict-items-number={}'.format(test_case_current[0]),
                   #'--measures="AUC prec@{} recall@{} NDCG MRR"'.format(itemsToPredict, itemsToPredict),
                   '--recommender-options="{}"'.format(recommenderOptions)
                   ]

        print("SLIM BPR hyperparameters: " + recommenderOptions)

        start_time = time.time()

        output = subprocess.check_output(' '.join(command), shell=True)

        print("\n\n" + str(output.decode()) + "\n\n")
        print("Train complete, time required {:.2f} seconds".format(time.time()-start_time))

        """
        training data: 179000 users, 4866 items, 4226544 events, sparsity 99.51476\n
        test data:     134486 users, 4820 items, 869112 events, sparsity 99.86592\n
        BPRSLIM reg_i=0.0025 reg_j=0.00025 num_iter=15 learn_rate=0.05 uniform_user_sampling=True with_replacement=False update_j=True \n
        training_time 00:02:51.6474470  prediction_time 00:12:23.3718540\n\n
        """

        # Read prediction file and calculate the scores
        #self.loadModelIntoSparseMatrix(basePath + outputModelName)
        self.loadModelIntoDenseMatrix(self.basePath + self.outputModelName)

        if deleteFiles:
            self.removeTemporaryFiles()

    def label(self, unlabeled_list, binary_ratings=False, n=None, exclude_seen=True, p_most=1, n_most=3, score_mode='user'):
        # Calculate the scores only one time.
        # users = []
        # items = []
        # for user_idx, item_idx in unlabeled_list:
        #     users.append(user_idx)
        #     items.append(item_idx)
        #
        # users = np.array(users,dtype=np.int32)
        # items = np.array(items,dtype=np.int32)
        unlabeled_list = check_matrix(unlabeled_list, 'lil', dtype=np.float32)
        users,items = unlabeled_list.nonzero()
        n_scores = len(users)
        uniq_users, user_to_idx = np.unique(users,return_inverse=True)
        if (score_mode == 'user'):
            filtered_scores = np.zeros(shape=n_scores,dtype=np.float32)
            curr_user = None
            i = 0
            for user,item in zip(users,items):
                if (curr_user != user):
                    curr_user = user
                    self.calculate_scores_user(curr_user)

                filtered_scores[i] = self.scores[item]
                i += 1

        elif (score_mode == 'batch'):
            pass
            # filtered_scores = []
            # uniq_users, user_to_idx = np.unique(users,return_inverse=True)
            # self.calculate_scores_batch(uniq_users)
            # filtered_scores = self.scores[users,items]

        elif (score_mode == 'matrix'):
            pass

        # At this point, we have all the predicted scores for the users inside
        # U'. Now we will filter the scores by keeping only the scores of the
        # items presented in U'. This will be an array where:
        # filtered_scores[i] = scores[users[i],items[i]]

        # Filtered the scores to have the n-most and p-most.
        # sorted_filtered_scores is sorted incrementally
        sorted_filtered_scores = filtered_scores.argsort()
        p_sorted_scores = sorted_filtered_scores[-p_most:]
        n_sorted_scores = sorted_filtered_scores[:n_most]

        if binary_ratings:
            scores = [(users[i], items[i], 1.0) for i in p_sorted_scores] + [(users[i], items[i], 0.0) for i in n_sorted_scores]
        else:
            scores = [(users[i], items[i], 5.0) for i in p_sorted_scores] + [(users[i], items[i], 1.0) for i in n_sorted_scores]


        meta = dict()
        meta['pos_labels'] = len(p_sorted_scores)
        meta['neg_labels'] = len(n_sorted_scores)
        meta['total_labels'] = len(p_sorted_scores) + len(n_sorted_scores)
        meta['pos_set'] = set([(users[i], items[i]) for i in p_sorted_scores])
        meta['neg_set'] = set([(users[i], items[i]) for i in n_sorted_scores])
        meta['neutral_set'] = set()

        # We sort the indices by user, then by item in order to make the
        # assignment to the LIL matrix faster.
        return sorted(scores, key=lambda triplet: (triplet[0],triplet[1])), meta

    def calculate_scores_user(self,user_id):
        user_profile = self._get_user_ratings(user_id)

        if self.sparse_weights:
            self.scores = user_profile.dot(self.W_sparse).toarray().ravel()
        else:
            self.scores = user_profile.dot(self.W).ravel()

        if self.normalize:
            # normalization will keep the scores in the same range
            # of value of the ratings in dataset
            rated = user_profile.copy()
            rated.data = np.ones_like(rated.data)
            if self.sparse_weights:
                den = rated.dot(self.W_sparse).toarray().ravel()
            else:
                den = rated.dot(self.W).ravel()
            den[np.abs(den) < 1e-6] = 1.0  # to avoid NaNs
            self.scores /= den

    def predict(self, user_id, rated_indices, score_mode='user'):
        # return the scores for the rated items.
        if (score_mode == 'user'):
            self.calculate_scores_user(user_id)
            return self.scores[rated_indices]
        elif (score_mode == 'matrix'):
            pass
