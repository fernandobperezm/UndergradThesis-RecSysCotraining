# theano-bpr
#
# Copyright (c) 2014 British Broadcasting Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import theano, numpy
import theano.tensor as T
import time
import sys
from collections import defaultdict
from .base import Recommender, check_matrix

import pdb

from multiprocessing import Pool
from functools import partial

# Memory consumption problem solved by:
# https://stackoverflow.com/questions/38140693/python3-multiprocessing-consumes-extensively-much-ram-and-slows-down
def _partial_uniform_sampling(tup, X):
    sgd_user,n_items = tup
    # to avoid race conditions.
    train = X.copy()
    pos_item = train[sgd_user][numpy.random.randint(len(train[sgd_user]))]
    neg_item = numpy.random.randint(n_items)
    while neg_item in train[sgd_user]:
        neg_item = numpy.random.randint(n_items)

    return sgd_user, pos_items, neg_items

class BPRMF_THEANO(Recommender):

    def __init__(self, rank, n_users, n_items, lambda_u = 0.0025, lambda_i = 0.0025, lambda_j = 0.00025, lambda_bias = 0.0, learning_rate = 0.05):
        """
          Creates a new object for training and testing a Bayesian
          Personalised Ranking (BPR) Matrix Factorisation
          model, as described by Rendle et al. in:

            http://arxiv.org/abs/1205.2618

          This model tries to predict a ranking of items for each user
          from a viewing history.
          It's also used in a variety of other use-cases, such
          as matrix completion, link prediction and tag recommendation.

          `rank` is the number of latent features in the matrix
          factorisation model.

          `n_users` is the number of users and `n_items` is the
          number of items.

          The regularisation parameters can be overridden using
          `lambda_u`, `lambda_i` and `lambda_j`. They correspond
          to each three types of updates.

          The learning rate can be overridden using `learning_rate`.

          This object uses the Theano library for training the model, meaning
          it can run on a GPU through CUDA. To make sure your Theano
          install is using the GPU, see:

            http://deeplearning.net/software/theano/tutorial/using_gpu.html

          When running on CPU, we recommend using OpenBLAS.

            http://www.openblas.net/

          Example use (10 latent dimensions, 100 users, 50 items) for
          training:

          >>> from theano_bpr import BPR
          >>> bpr = BPRMF_THEANO(10, 100, 50)
          >>> from numpy.random import randint
          >>> train_data = zip(randint(100, size=1000), randint(50, size=1000))
          >>> bpr.train(train_data)

          This object also has a method for testing, which will return
          the Area Under Curve for a test set.

          >>> test_data = zip(randint(100, size=1000), randint(50, size=1000))
          >>> bpr.test(test_data)

          (This should give an AUC of around 0.5 as the training and
          testing set are chosen at random)
        """
        super(BPRMF_THEANO, self).__init__()
        self.dataset = None
        self._rank = rank
        self._n_users = n_users
        self._n_items = n_items
        self._lambda_u = lambda_u
        self._lambda_i = lambda_i
        self._lambda_j = lambda_j
        self._lambda_bias = lambda_bias
        self._learning_rate = learning_rate
        self._train_users = set()
        self._train_items = set()
        self._train_dict = dict()
        self._configure_theano()
        self._generate_train_model_function()

    def short_str(self):
        return "BPRMF_THEANO"

    def __str__(self):
        return "BPRMF_THEANO(num_factors={},lrate={},user_reg={},pos_reg={},neg_reg={})".format(
            self._rank, self._learning_rate, self._lambda_u, self._lambda_i, self._lambda_j
        )

    def _configure_theano(self):
        """
          Configures Theano to run in fast mode
          and using 32-bit floats.
        """
        theano.config.mode = 'FAST_RUN'
        theano.config.floatX = 'float32'

    def _generate_train_model_function(self):
        """
          Generates the train model function in Theano.
          This is a straight port of the objective function
          described in the BPR paper.

          We want to learn a matrix factorisation

            U = W.H^T

          where U is the user-item matrix, W is a user-factor
          matrix and H is an item-factor matrix, so that
          it maximises the difference between
          W[u,:].H[i,:]^T and W[u,:].H[j,:]^T,
          where `i` is a positive item
          (one the user `u` has watched) and `j` a negative item
          (one the user `u` hasn't watched).
        """
        u = T.lvector('u')
        i = T.lvector('i')
        j = T.lvector('j')

        self.W = theano.shared(numpy.random.random((self._n_users, self._rank)).astype('float32'), name='W')
        self.H = theano.shared(numpy.random.random((self._n_items, self._rank)).astype('float32'), name='H')

        self.B = theano.shared(numpy.zeros(self._n_items).astype('float32'), name='B')

        x_ui = T.dot(self.W[u], self.H[i].T).diagonal()
        x_uj = T.dot(self.W[u], self.H[j].T).diagonal()

        x_uij = self.B[i] - self.B[j] + x_ui - x_uj

        obj = T.sum(T.log(T.nnet.sigmoid(x_uij)) - self._lambda_u * (self.W[u] ** 2).sum(axis=1) - self._lambda_i * (self.H[i] ** 2).sum(axis=1) - self._lambda_j * (self.H[j] ** 2).sum(axis=1) - self._lambda_bias * (self.B[i] ** 2 + self.B[j] ** 2))
        cost = - obj

        g_cost_W = T.grad(cost=cost, wrt=self.W)
        g_cost_H = T.grad(cost=cost, wrt=self.H)
        g_cost_B = T.grad(cost=cost, wrt=self.B)

        updates = [ (self.W, self.W - self._learning_rate * g_cost_W), (self.H, self.H - self._learning_rate * g_cost_H), (self.B, self.B - self._learning_rate * g_cost_B) ]

        self.train_model = theano.function(inputs=[u, i, j], outputs=cost, updates=updates)

    def train(self, train_data, epochs=30, batch_size=1000):
        """
          Trains the BPR Matrix Factorisation model using Stochastic
          Gradient Descent and minibatches over `train_data`.

          `train_data` is an array of (user_index, item_index) tuples.

          We first create a set of random samples from `train_data` for
          training, of size `epochs` * size of `train_data`.

          We then iterate through the resulting training samples by
          batches of length `batch_size`, and run one iteration of gradient
          descent for the batch.
        """
        train_data = check_matrix(train_data, format='csr', dtype=numpy.float32)
        self.dataset = train_data

        n_sgd_samples = epochs * train_data.nnz
        self.URM_train = train_data
        self.batch_size = batch_size
        self.initializeFastSampling()

        z = 0
        t2 = t1 = t0 = time.time()
        while (z+1)*batch_size < n_sgd_samples:
            sgd_users, sgd_pos_items, sgd_neg_items = self.sampleBatch()
            self.train_model(
                sgd_users,
                sgd_pos_items,
                sgd_neg_items
            )

            z += 1
            t2 = time.time()
            sys.stderr.write("\rProcessed %s ( %.2f%% ) in %.4f seconds" %(str(z*batch_size), 100.0 * float(z*batch_size)/n_sgd_samples, t2 - t1))
            sys.stderr.flush()
            t1 = t2
        if n_sgd_samples > 0:
            sys.stderr.write("\nTotal training time %.2f seconds; %e per sample\n" % (t2 - t0, (t2 - t0)/n_sgd_samples))
            sys.stderr.flush()

        # # To accomplish the specification of train data.
        # # pdb.set_trace()
        # train = check_matrix(train_data, 'lil', dtype=numpy.float32)
        # user_idx, item_idx = train.nonzero()
        # train_data = list(zip(user_idx,item_idx))
        #
        # if len(train_data) < batch_size:
        #     sys.stderr.write("WARNING: Batch size is greater than number of training samples, switching to a batch size of %s\n" % str(len(train_data)))
        #     batch_size = len(train_data)
        # self._train_dict, self._train_users, self._train_items = self._data_to_dict(train_data)
        # n_sgd_samples = len(train_data) * epochs
        # sgd_users, sgd_pos_items, sgd_neg_items = self._uniform_user_sampling(n_sgd_samples)
        #z = 0
        #t2 = t1 = t0 = time.time()
        # while (z+1)*batch_size < n_sgd_samples:
        #     self.train_model(
        #         sgd_users[z*batch_size: (z+1)*batch_size],
        #         sgd_pos_items[z*batch_size: (z+1)*batch_size],
        #         sgd_neg_items[z*batch_size: (z+1)*batch_size]
        #     )
        #     z += 1
        #     t2 = time.time()
        #     sys.stderr.write("\rProcessed %s ( %.2f%% ) in %.4f seconds" %(str(z*batch_size), 100.0 * float(z*batch_size)/n_sgd_samples, t2 - t1))
        #     sys.stderr.flush()
        #     t1 = t2
        # if n_sgd_samples > 0:
        #     sys.stderr.write("\nTotal training time %.2f seconds; %e per sample\n" % (t2 - t0, (t2 - t0)/n_sgd_samples))
        #     sys.stderr.flush()

    def initializeFastSampling(self):
        print("Initializing fast sampling")

        self.eligibleUsers = []
        self.userSeenItems = dict()

        for user_id in range(self._n_users):

            if (self.URM_train[user_id].nnz >0):

                self.eligibleUsers.append(user_id)
                self.userSeenItems[user_id] = self.URM_train[user_id].indices

        self.eligibleUsers = numpy.array(self.eligibleUsers)

    def sampleBatch(self):
        user_id_list = numpy.random.choice(self.eligibleUsers, size=(self.batch_size))
        pos_item_id_list = []
        neg_item_id_list = []

        for user_id in user_id_list:
            pos_item_id_list.append(numpy.random.choice(self.userSeenItems[user_id]))

            negItemSelected = False

            # It's faster to just try again than to build a mapping of the non-seen items
            # for every user
            while(not negItemSelected):
                neg_item_id = numpy.random.randint(0, self._n_items)

                if(neg_item_id not in self.userSeenItems[user_id]):
                    negItemSelected = True
                    neg_item_id_list.append(neg_item_id)

        return user_id_list, pos_item_id_list, neg_item_id_list

    def _uniform_user_sampling(self, n_samples):
        """
          Creates `n_samples` random samples from training data for performing Stochastic
          Gradient Descent. We start by uniformly sampling users,
          and then sample a positive and a negative item for each
          user sample.
        """
        # pdb.set_trace()
        sys.stderr.write("Generating %s random training samples\n" % str(n_samples))
        sgd_users = numpy.array(list(self._train_users))[numpy.random.randint(len(list(self._train_users)), size=n_samples)]
        # sgd_pos_items, sgd_neg_items = [], []
        sgd_pos_items = numpy.zeros(n_samples,dtype=numpy.int32)
        sgd_neg_items = numpy.zeros(n_samples,dtype=numpy.int32)
        i = 0
        # self.workers = 4
        # _partial_sampling = partial(_partial_uniform_sampling, X=self._train_users)
        # pool = Pool(processes=self.workers)
        # args_tuple = ((sgd_user,self._n_items) for sgd_user in sgd_users)
        # res = pool.map(_partial_sampling, args_tuple)
        # pool.close()
        # pool.join()
        #
        # print("The Training samples are created.")
        # # res contains a vector of (values, rows, cols) tuples
        # sgd_users, sgd_pos_items, sgd_neg_items = [], [], []
        # for user_, pos_item_, neg_item_ in res:
        #     sgd_users.append(user_)
        #     sgd_pos_items.append(pos_item_)
        #     sgd_neg_items.append(neg_item_)
        for sgd_user in sgd_users:
            if (i % 1000000 == 0):
                print("Sample: {}".format(i))
            pos_item = self._train_dict[sgd_user][numpy.random.randint(len(self._train_dict[sgd_user]))]
            # sgd_pos_items.append(pos_item)
            sgd_pos_items[i] = pos_item
            neg_item = numpy.random.randint(self._n_items)
            while neg_item in self._train_dict[sgd_user]:
                neg_item = numpy.random.randint(self._n_items)
            # sgd_neg_items.append(neg_item)
            sgd_neg_items[i] = neg_item
            i += 1
        # pool = None
        return sgd_users, sgd_pos_items, sgd_neg_items

    def predictions(self, user_index):
        """
          Computes item predictions for `user_index`.
          Returns an array of prediction values for each item
          in the dataset.
        """
        w = self.W.get_value()
        h = self.H.get_value()
        b = self.B.get_value()
        user_vector = w[user_index,:]
        return user_vector.dot(h.T) + b

    def prediction(self, user_index, item_index):
        """
          Predicts the preference of a given `user_index`
          for a gven `item_index`.
        """
        return self.predictions(user_index)[item_index]

    def top_predictions(self, user_index, topn=10,exclude_seen=True):
        """
          Returns the item indices of the top predictions
          for `user_index`. The number of predictions to return
          can be set via `topn`.
          This won't return any of the items associated with `user_index`
          in the training set.
        """

        # return [
        #     item_index for item_index in numpy.argsort(self.predictions(user_index))
        #     if item_index not in self._train_dict[user_index]
        # ][::-1][:topn]
        ranking = numpy.argsort(self.predictions(user_index))[::-1]
        if exclude_seen:
            ranking = self._filter_seen(user_index, ranking)
        return ranking[:topn]

    def fit(self, R):
        return self.train(train_data=R, epochs=10, batch_size=1000)

    def recommend(self, user_id, n=None, exclude_seen=True):
        return self.top_predictions(user_index=user_id,topn=n)

    def predict(self, user_id, rated_indices):
        return self.prediction(user_index=user_id,item_index=rated_indices)

    def label(self, unlabeled_list, binary_ratings=False, n=None, exclude_seen=True, p_most=1, n_most=3):
        unlabeled_list = check_matrix(unlabeled_list, 'lil', dtype=numpy.float32)
        users,items = unlabeled_list.nonzero()
        uniq_users, user_to_idx = numpy.unique(users,return_inverse=True)
        # At this point, we have all the predicted scores for the users inside
        # U'. Now we will filter the scores by keeping only the scores of the
        # items presented in U'. This will be an array where:
        # filtered_scores[i] = scores[users[i],items[i]]
        # scores = np.dot(self.X[uniq_users], self.Y.T)
        scores = self.predictions(user_index=uniq_users)
        filtered_scores = scores[user_to_idx,items]

        # Filtered the scores to have the n-most and p-most.
        # sorted_filtered_scores is sorted incrementally
        sorted_filtered_scores = filtered_scores.argsort()
        p_sorted_scores = sorted_filtered_scores[-p_most:]
        n_sorted_scores = sorted_filtered_scores[:n_most]

        if binary_ratings:
            scores = [(users[i], items[i], 1.0) for i in p_sorted_scores] + [(users[i], items[i], 0.0) for i in n_sorted_scores]
        else:
            scores = [(users[i], items[i], 5.0) for i in p_sorted_scores] + [(users[i], items[i], 1.0) for i in n_sorted_scores]

        # We sort the indices by user, then by item in order to make the
        # assignment to the LIL matrix faster.
        return sorted(scores, key=lambda triplet: (triplet[0],triplet[1]))


    def test(self, test_data):
        """
          Computes the Area Under Curve (AUC) on `test_data`.

          `test_data` is an array of (user_index, item_index) tuples.

          During this computation we ignore users and items
          that didn't appear in the training data, to allow
          for non-overlapping training and testing sets.
        """
        test_dict, test_users, test_items = self._data_to_dict(test_data)
        auc_values = []
        z = 0
        for user in test_dict.keys():
            if user in self._train_users:
                auc_for_user = 0.0
                n = 0
                predictions = self.predictions(user)
                for pos_item in test_dict[user]:
                    if pos_item in self._train_items:
                        for neg_item in self._train_items:
                            if neg_item not in test_dict[user] and neg_item not in self._train_dict[user]:
                                n += 1
                                if predictions[pos_item] > predictions[neg_item]:
                                    auc_for_user += 1
                if n > 0:
                    auc_for_user /= n
                    auc_values.append(auc_for_user)
                z += 1
                if z % 100 == 0 and len(auc_values) > 0:
                    sys.stderr.write("\rCurrent AUC mean (%s samples): %0.5f" % (str(z), numpy.mean(auc_values)))
                    sys.stderr.flush()
        sys.stderr.write("\n")
        sys.stderr.flush()
        return numpy.mean(auc_values)

    def _data_to_dict(self, data):
        data_dict = defaultdict(list)
        items = set()
        for (user, item) in data:
            data_dict[user].append(item)
            items.add(item)
        return data_dict, set(data_dict.keys()), items
