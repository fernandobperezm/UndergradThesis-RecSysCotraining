#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Politecnico di Milano.
SLIM_BPR_Mono.py

Description: This file contains the definition and implementation of a SLIM
             recommender BPR-optimized.

@author: Maurizio Ferrari Dacrema
Modified by: Fernando Pérez.

Created on 28 June 2017
Last modified on 05/09/2017.
"""

import numpy as np
import time
import os
from .Recommender import Recommender
from .Recommender_utils import similarityMatrixTopK, check_matrix
import scipy.sparse as sps
import subprocess

class SLIM_BPR_Mono(Recommender):
    """
    Train a Sparse Linear Methods (SLIM) item similarity model.
    This model is optimized for ranking, specifically, using BPR.
    This class uses MyMediaLite.

    See:
        Efficient Top-N Recommendation by Linear Regression,
        M. Levy and K. Jack, LSRS workshop at RecSys 2013.

        SLIM: Sparse linear methods for top-n recommender systems,
        X. Ning and G. Karypis, ICDM 2011.
        http://glaros.dtc.umn.edu/gkhome/fetch/papers/SLIM2011icdm.pdf

        BPR: Bayesian Personalized Ranking from Implicit Feedback,
        Steffen Rendle, Christoph Freudenthaler, Zeno Gantner and Lars Schmidt-Thieme,
        Proceedings of the Twenty-Fifth Conference on Uncertainty in Artificial
        Intelligence, UAI 2009.
        https://arxiv.org/abs/1205.2618

        MyMediaLite: A Free Recommender System Library,
        Zeno Gantner, Steffen Rendle, Christoph Freudenthaler and Lars Schmidt-Thieme,
        Proceedings of the Fifth ACM Conference on Recommender Systems.
        RecSys '11.
        http://dl.acm.org/citation.cfm?id=2043989

    Attibutes:
        * URM_train: dataset which we use to build the model.
        * lambda_i: regularization term for the positive items.
        * lambda_j: regularization term for the negative items.
        * learning_rate: learning rate for the SGD.
        * topK: top-K most similar items.
        * W: A matrix specifying the similarity between items in a dense-matrix
             format.
        * W_sparse: A matrix specifying the similarity between items in a
                    sparse-matrix format.
        * basePath: path where all the files are located.
        * executablePath: path were the MyMediaLite executable is.
        * trainFileName: name of the train file.
        * testFileName: name of the test file.
        * outputModelName: name of the model file.

    Warning:
        This class needs to save files on disk to call another executable.
        Neither this class nor the executable are thread safe.

    """
    def __init__(self, lambda_i = 0.0025, lambda_j = 0.00025, learning_rate = 0.05, topK = None):
        """Constructor of the SLIM_BPR_Mono class.

           Args:
                * URM_train: dataset which we use to build the model.
                * lambda_i: regularization term for the positive items.
                * lambda_j: regularization term for the negative items.
                * learning_rate: learning rate for the SGD.
                * topK: top-K most similar items.

           Args type:
                * URM_train: Scipy.Sparse matrix.
                * lambda_i: float
                * lambda_j: float
                * learning_rate: float
                * topK: int

        """
        super(SLIM_BPR_Mono, self).__init__()

        self.lambda_i = lambda_i
        self.lambda_j = lambda_j
        self.learning_rate = learning_rate
        self.normalize = False
        self.sparse_weights = True
        self.topK = topK

        self.basePath = "../../../Datasets/ml10m/"
        self.executablePath = "./item_recommendation"
        self.trainFileName = "SLIM_BPR_Mono_URM_train.csv"
        self.testFileName = "SLIM_BPR_Mono_URM_test.csv"
        self.outputModelName = "SLIM_BPR_Mono_Model.txt"

        # Set permission to execute, code 0775, the o is needed in python to encode octal numbers
        os.chmod(self.executablePath, 0o775)

    def __str__(self):
        """ String representation of the class. """
        return "SLIM_BPR_Mono(lambda_i={},lambda_j={},learning_rate={},topK={})".format(self.lambda_i, self.lambda_j, self.learning_rate, self.topK)

    def short_str(self):
        """ Short string used for dictionaries. """
        return "SLIM_BPR_Mono"

    def _get_user_ratings(self, user_id):
        return self.URM_train[user_id]

    def writeTestToFile(self, URM_test):
        """Writes a test set to a file.

            Args:
                * URM_test: Matrix that holds the test set.

            Args type:
                * URM_test: Scipy.Sparse matrix.
        """
        if URM_test is not None:
            self.writeSparseToFile(URM_test, open(self.basePath + self.testFileName, "w"))

    def removeTemporaryFiles(self):
        """Remove old and temporary files made by older trainings of the algorithm.

            This method removes old Training, Training with only positive feedback,
            and model files.
        """

        # Remove saved Model and URM

        print("Removing: {}".format(self.basePath + self.trainFileName))
        os.remove(self.basePath + self.trainFileName)

        print("Removing: {}".format(self.basePath + self.trainFileName + ".bin.PosOnlyFeedback"))
        os.remove(self.basePath + self.trainFileName + ".bin.PosOnlyFeedback")

        print("Removing: {}".format(self.basePath + self.outputModelName))
        os.remove(self.basePath + self.outputModelName)

    def writeSparseToFile(self, sparseMatrix, file):
        """Writes into disk a sparse matrix as a file.

            Args:
                * sparseMatrix: the sparse matrix to be saved.
                * file: an opened file where the matrix is going to be saved.

            Args type:
                * sparseMatrix: Scipy.Sparse matrix.
                * file: File instance.

        """
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
        """Loads into memory a dense matrix from a file.

            This method loads the model saved by MyMediaLite into disk, as a
            dense matrix. However, if only the top-K similar items are considered
            then a sparse matrix is created instead of a dense. The matrix is
            stored in `self.W` or `self.W_sparse`

            Args:
                * filePath: represents the name of the file that we will read the
                        matrix

            Args type:
                * filePath: str

        """
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
            print("No sparse_weights")

    def loadModelIntoSparseMatrix(self, filePath):
        """Loads into memory a sparse matrix from a file.

            This method loads the model saved by MyMediaLite into disk, as a
            sparse matrix. The matrix is stored in `self.W_sparse`.

            Args:
                * filePath: represents the name of the file that we will read the
                        matrix

            Args type:
                * filePath: str

        """

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
        """Trains and builds the model given a dataset.

            The fit function inside the SLIM_BPR_Mono class builds a similarity matrix
            between all the items, this similarity is calculated in a way that
            the ranking is optimized.

            This function executes a command-line method passing the necessary
            parameters in order to run the SLIMBPR recommender that comes into
            the MyMediaLite library. After MyMediaLite finishes the building
            of the model, this one is loaded into a dense matrix.

            It makes use of FileIO and remove old training and model files.

            Args:
                * URM_train: User-Rating Matrix for which we will train the model.
                * epochs: number of epochs to perform SGD.
                * deleteFiles: delete older files or not.

            Args type:
                * URM_train: Scipy.Sparse matrix.
                * epochs: int
                * deleteFiles: bool
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

    def label(self, unlabeled_list, binary_ratings=False, exclude_seen=True, p_most=1, n_most=3, score_mode='user'):
        """Rates new user-item pairs.

           This function is part of the Co-Training process in which we rate
           all user-item pairs inside an unlabeled pool of samples, afterwards,
           we separate them into positive and negative items based on their score.
           Lastly, we take the p-most positive and n-most negative items from all
           the rated items.

           Inside the function we also measure some statistics that help us to
           analyze the effects of the Co-Training process, such as, number of
           positive, negative and neutral items rated and sets of positive, negative
           and neutral user-item pairs to see the agreement of the recommenders.
           We put all these inside a dictionary.

           Args:
               * unlabeled_list: a matrix that holds the user-item that we must
                                 predict their rating.
               * binary_ratings: tells us if we must predict based on an implicit
                                 (0,1) dataset or an explicit.
               * exclude_seen: tells us if we need to exclude already-seen items.
               * p_most: tells the number of p-most positive items that we
                         should choose.
               * n_most: tells the number of n-most negative items that we
                         should choose.
               * score_mode: the type of score prediction, 'user' represents by
                             sequentially user-by-user, 'batch' represents by
                             taking batches of users, 'matrix' represents to
                             make the preditions by a matrix multiplication.

           Args type:
               * unlabeled_list: Scipy.Sparse matrix.
               * binary_ratings: bool
               * exclude_seen: bool
               * p_most: int
               * n_most: int
               * score_mode: str

           Returns:
               A list containing the user-item-rating triplets and the meta
               dictionary for statistics.
        """
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
        """Calculates the score for all the items for a batch of users.

           This function makes the matrix multiplication between the profile
           of the user and the item similarities. This matrix multiplication
           returns the predicted score for all the items based on the users
           preferences.

           All the scores are stored inside `self.scores`, and can be either
           normalized or not.

           Args:
                * user_id: the user index inside the system.

            Args type:
                * users: int.
        """
        user_profile = self._get_user_ratings(user_id)

        if self.sparse_weights:
            self.scores = user_profile.dot(self.W_sparse).toarray().ravel()
        else:
            self.scores = user_profile.dot(self.W).ravel()

        if self.normalize:
            print("Normalizing")
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
        """Calculates the predicted preference of a user for a list of items.

            Args:
                * user_id: user index to which we will build the top-N list.
                * rated_indices: list that holds the items for which we will
                                 predict the user preference.
                * score_mode: the score is created only for one user or it is
                              created for all the users.

            Args type:
                * user_id: int
                * rated_indices: list of int.

            Returns:
                A list of predicted preferences for each item in the list given.
        """
        # return the scores for the rated items.
        if (score_mode == 'user'):
            self.calculate_scores_user(user_id)
            return self.scores[rated_indices]
        elif (score_mode == 'matrix'):
            pass
