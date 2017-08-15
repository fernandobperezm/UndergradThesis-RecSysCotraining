# -*- coding: utf-8 -*-
# @Author: massimo
# @Date:   2016-10-24 15:27:46
# @Last Modified by:   massimo
# @Last Modified time: 2016-10-24 17:03:51

import numpy as np
import scipy.sparse as sps
from ..utils.metrics import roc_auc, precision, recall, map, ndcg, rr
from .Recommender_utils import check_matrix, areURMequals, removeTopPop
import multiprocessing
import time
import random

class Recommender(object):
    """Abstract Recommender"""

    def __init__(self):
        super(Recommender, self).__init__()
        self.URM_train = None
        self.sparse_weights = True
        self.normalize = False
        self.FastValidation_initialized = False
        self.filterTopPop = False

    def _get_user_ratings(self, user_id):
        #return self.URM_train[user_id]
        return self.URM_train_user_profile[user_id]

    def _get_item_ratings(self, item_id):
        return self.URM_train[:, item_id]

    def fit(self, URM_train):
        pass

    def _filter_TopPop(self, ranking):
        nonTopPop_mask = np.in1d(ranking, self.filterTopPop_ItemsID, assume_unique=True, invert=True)
        return ranking[nonTopPop_mask]

    def _filter_TopPop_on_scores(self, scores):
        scores[self.filterTopPop_ItemsID] = -np.inf
        return scores

    def _filter_seen_on_scores(self, user_id, scores):
        seen = self.URM_train_relevantItems[user_id]
        scores[seen] = -np.inf
        return scores

    def _filter_seen(self, user_id, ranking):
        seen = self.URM_train_relevantItems[user_id]
        unseen_mask = np.in1d(ranking, seen, assume_unique=True, invert=True)
        return ranking[unseen_mask]

    def evaluateRecommendations(self, URM_test_new, at=5, minRatingsPerUser=1, exclude_seen=True,
                                mode='sequential', filterTopPop = False,
                                fastValidation=True):
        """

        :param URM_test_new:            URM to be used for testing
        :param at: 5                    Length of the recommended items
        :param minRatingsPerUser: 1     Users with less than this number of interactions will not be evaluated
        :param exclude_seen: True       Whether to remove already seen items from the recommended items
        :param fastValidation: True     Use a faster but memory consuming dictionary to store the URM. Advisable if performing
                                        many validations on the same model

        :param mode: 'sequential', 'parallel'
        :param filterTopPop: False or decimal number        Percentage of items to be removed from recommended list and testing interactions
        :return:
        """

        if filterTopPop != False:

            self.filterTopPop = True

            _,_, self.filterTopPop_ItemsID = removeTopPop(self.URM_train, URM_2 = URM_test_new, percentageToRemove=filterTopPop)

            print("Filtering {}% TopPop items, count is: {}".format(filterTopPop*100, len(self.filterTopPop_ItemsID)))

            # Zero-out the items in order to be considered irrelevant
            URM_test_new = check_matrix(URM_test_new, format='lil')
            URM_test_new[:,self.filterTopPop_ItemsID] = 0
            URM_test_new = check_matrix(URM_test_new, format='csr')

        if self.FastValidation_initialized:

            # If all the data structures are initialized, recompute it only if URM_test changed
            recomputeFastValidationDictionary = not areURMequals(self.URM_test, URM_test_new)

        else:
            recomputeFastValidationDictionary = True

        # During testing CSR is faster
        self.URM_test = check_matrix(URM_test_new, format='csr')
        self.URM_train = check_matrix(self.URM_train, format='csr')
        self.at = at
        self.minRatingsPerUser = minRatingsPerUser
        self.exclude_seen = exclude_seen


        nusers = self.URM_test.shape[0]

        # Prune users with an insufficient number of ratings
        rows = self.URM_test.indptr
        numRatings = np.ediff1d(rows)
        mask = numRatings >= minRatingsPerUser
        usersToEvaluate = np.arange(nusers)[mask]

        usersToEvaluate = list(usersToEvaluate)

        # Generate dictionary data structure
        # - If no fast falidation required, basically recompute it anyway
        # - If fast validation required and URM_test is new
        if not fastValidation or recomputeFastValidationDictionary:
            self.initializeURMDictionary(self.URM_train, self.URM_test)
        else:
            print("URM_test fastValidation already initialised")


        # if mode=='sequential':
        #     return self.evaluateRecommendationsSequential(usersToEvaluate)
        # elif mode=='parallel':
        #     return self.evaluateRecommendationsParallel(usersToEvaluate)
        # elif mode=='random-equivalent':
        #     return self.evaluateRecommendationsRandomEquivalent(usersToEvaluate)
        # else:
        #     raise ValueError("Mode '{}' not available".format(mode))


    def evaluateRecommendationsSequential(self, usersToEvaluate):

        start_time = time.time()

        roc_auc_, precision_, recall_, map_, mrr_, ndcg_ = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        n_eval = 0

        for test_user in usersToEvaluate:

            # Calling the 'evaluateOneUser' function instead of copying its code would be cleaner, but is 20% slower

            # Being the URM CSR, the indices are the non-zero column indexes
            #relevant_items = self.URM_test[test_user].indices
            relevant_items = self.URM_test_relevantItems[test_user]

            n_eval += 1

            recommended_items = self.recommend(user_id=test_user, exclude_seen=self.exclude_seen,
                                               n=self.at, filterTopPop=self.filterTopPop)

            is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

            # evaluate the recommendation list with ranking metrics ONLY
            roc_auc_ += roc_auc(is_relevant)
            precision_ += precision(is_relevant)
            recall_ += recall(is_relevant, relevant_items)
            map_ += map(is_relevant, relevant_items)
            mrr_ += rr(is_relevant)
            #ndcg_ += ndcg(recommended_items, relevant_items, relevance=self.URM_test[test_user].data, at=self.at)
            ndcg_ += ndcg(recommended_items, relevant_items, relevance=self.URM_test_ratings[test_user], at=self.at)



            if(n_eval % 10000 == 0):
                print("Processed {} ( {:.2f}% ) in {:.2f} seconds. Users per second: {:.0f}".format(
                                  n_eval,
                                  100.0* float(n_eval)/len(usersToEvaluate),
                                  time.time()-start_time,
                                  float(n_eval)/(time.time()-start_time)))




        if (n_eval > 0):
            roc_auc_ /= n_eval
            precision_ /= n_eval
            recall_ /= n_eval
            map_ /= n_eval
            mrr_ /= n_eval
            ndcg_ /= n_eval

        else:
            print("WARNING: No users had a sufficient number of relevant items")

        results_run = {}

        results_run["AUC"] = roc_auc_
        results_run["precision"] = precision_
        results_run["recall"] = recall_
        results_run["map"] = map_
        results_run["NDCG"] = ndcg_
        results_run["MRR"] = mrr_

        return (results_run)



    def evaluateRecommendationsRandomEquivalent_oneUser(self, test_user):

        hitCount = 0

        seenItems = set(self.URM_test_relevantItems[test_user])
        seenItems.union(set(self.URM_train_relevantItems[test_user]))

        unseenItems = self.allItemsSet.difference(seenItems)

        # Being the URM CSR, the indices are the non-zero column indexes
        user_profile = self.URM_train_user_profile[test_user]

        # hits_vector = np.zeros(numRandomItems)



        if self.sparse_weights:
            scores = user_profile.dot(self.W_sparse).toarray().ravel()
            # scores = self.scoresAll[user_id].toarray().ravel()
        else:
            scores = user_profile.dot(self.W).ravel()

        ranking = scores.argsort()
        ranking = np.flip(ranking, axis=0)
        ranking = ranking[0:100]

        # For each item
        for test_item in self.URM_test_relevantItems[test_user]:

            # Randomly select a given number of items, default 1000
            other_items = random.sample(unseenItems, self.numRandomItems)
            other_items.append(test_item)

            items_mask = np.in1d(ranking, other_items, assume_unique=True)
            ranking = ranking[items_mask]

            item_position = np.where(ranking == test_item)

            if len(item_position) > 0:
                # hits_vector[item_position:numRandomItems] += 1
                hitCount += 1

        #print(test_user)
        self.evaluateRecommendationsRandomEquivalent_hit += hitCount


    def evaluateRecommendationsRandomEquivalent(self, usersToEvaluate, numRandomItems = 1000):

        start_time = time.time()

        # Initialize data structure for unseen items
        nitems = self.URM_test.shape[1]


        self.allItemsSet = set(np.arange(nitems))
        self.numRandomItems = numRandomItems
        self.evaluateRecommendationsRandomEquivalent_hit = 0

        print("Parallel evaluation starting")

        #pool = multiprocessing.Pool(processes=2, maxtasksperchild=10)
        #pool.map(self.evaluateRecommendationsRandomEquivalent_oneUser, usersToEvaluate)

        n_eval = 0
        for test_user in usersToEvaluate:
            self.evaluateRecommendationsRandomEquivalent_oneUser(test_user)

            n_eval += 1

            if(n_eval % 1000 == 0):
                print("Processed {} ( {:.2f}% ) in {:.2f} seconds. Users per second: {:.0f}".format(
                                  n_eval,
                                  100.0* float(n_eval)/len(usersToEvaluate),
                                  time.time()-start_time,
                                  float(n_eval)/(time.time()-start_time)))





        hitCount = self.evaluateRecommendationsRandomEquivalent_hit

        print("Evaluation complete in {:.2f} seconds".format(time.time()-start_time))

        recall_value = hitCount / self.URM_test.nnz

        results_run = {}

        results_run["AUC"] = 0.0
        results_run["precision"] = 0.0
        results_run["recall"] = recall_value
        results_run["map"] = 0.0
        results_run["NDCG"] = 0.0
        results_run["MRR"] = 0.0

        return (results_run)





    def evaluateRecommendationsBatch(self, URM_test, at=5, minRatingsPerUser = 1, exclude_seen=False, batch_size = 5000):

        # During testing CSR is faster
        URM_test = check_matrix(URM_test, format='csr')
        self.URM_train = check_matrix(self.URM_train, format='csr')

        roc_auc_, precision_, recall_, map_, mrr_, ndcg_ = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        n_eval = 0

        nusers = URM_test.shape[0]

        # Prune users with an insufficient number of ratings
        rows = URM_test.indptr
        numRatings = np.ediff1d(rows)
        mask = numRatings >= minRatingsPerUser
        usersToEvaluate = np.arange(nusers)[mask]

        usersToEvaluate = list(usersToEvaluate)

        #Number of blocks is rounded to the next integer
        totalNumberOfBatch = int(len(usersToEvaluate) / batch_size) + 1

        for current_batch in range(totalNumberOfBatch):

            user_first_id = current_batch*batch_size
            user_last_id = min((current_batch+1)*batch_size-1,  len(usersToEvaluate)-1)

            relevant_items_batch = URM_test[usersToEvaluate[user_first_id:user_last_id]].toarray()

            if(current_batch % 1 == 0):
                print("Testing user {} of {}".format(current_batch*batch_size, len(usersToEvaluate)))


            recommended_items_batch = self.recommendBatch(usersToEvaluate[user_first_id:user_last_id],
                                                          exclude_seen=exclude_seen,
                                                          relevant_items=relevant_items_batch,
                                                          n=at)


            for test_user in range(recommended_items_batch.shape[0]):

                n_eval += 1

                relevant_items = relevant_items_batch[test_user,:]
                recommended_items = recommended_items_batch[test_user,:]

                is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

                # evaluate the recommendation list with ranking metrics ONLY
                roc_auc_ += roc_auc(is_relevant)
                precision_ += precision(is_relevant)
                recall_ += recall(is_relevant, relevant_items)
                map_ += map(is_relevant, relevant_items)
                mrr_ += rr(is_relevant)
                ndcg_ += ndcg(recommended_items, relevant_items, relevance=relevant_items, at=at)


        if (n_eval > 0):
            roc_auc_ /= n_eval
            precision_ /= n_eval
            recall_ /= n_eval
            map_ /= n_eval
            mrr_ /= n_eval
            ndcg_ /= n_eval

        else:
            print("WARNING: No users had a sufficient number of relevant items")

        results_run = {}

        results_run["AUC"] = roc_auc_
        results_run["precision"] = precision_
        results_run["recall"] = recall_
        results_run["map"] = map_
        results_run["NDCG"] = ndcg_
        results_run["MRR"] = mrr_

        return (results_run)


    def initializeURMDictionary(self, URM_train, URM_test):

        print("Initializing fast testing")
        start_time = time.time()

        nusers = URM_test.shape[0]

        self.URM_test_relevantItems = dict()
        self.URM_test_ratings = dict()
        self.URM_train_user_profile = dict()
        self.URM_train_relevantItems = dict()

        self.FastValidation_initialized = True

        for user_id in range(nusers):

            trainData = URM_train[user_id]
            testData = URM_test[user_id]

            self.URM_train_user_profile[user_id] = trainData
            self.URM_train_relevantItems[user_id] = trainData.indices

            self.URM_test_relevantItems[user_id] = testData.indices
            self.URM_test_ratings[user_id] = testData.data

        print("Initialization complete in {:.2f} seconds".format(time.time()-start_time))



    def evaluateOneUser(self, test_user):

        # Being the URM CSR, the indices are the non-zero column indexes
        #relevant_items = self.URM_test_relevantItems[test_user]
        relevant_items = self.URM_test[test_user].indices

        # this will rank top n items
        recommended_items = self.recommend(user_id=test_user, exclude_seen=self.exclude_seen,
                                           n=self.at, filterTopPop=self.filterTopPop)

        is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

        # evaluate the recommendation list with ranking metrics ONLY
        roc_auc_ = roc_auc(is_relevant)
        precision_ = precision(is_relevant)
        recall_ = recall(is_relevant, relevant_items)
        map_ = map(is_relevant, relevant_items)
        mrr_ = rr(is_relevant)
        ndcg_ = ndcg(recommended_items, relevant_items, relevance=self.URM_test_ratings[test_user], at=self.at)

        return roc_auc_, precision_, recall_, map_, mrr_, ndcg_



    def evaluateRecommendationsParallel(self, usersToEvaluate):

        print("Evaluation of {} users begins".format(len(usersToEvaluate)))

        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count(), maxtasksperchild=1)
        resultList = pool.map(self.evaluateOneUser, usersToEvaluate)

        #for i, _ in enumerate(pool.imap_unordered(self.evaluateOneUser, usersToEvaluate), 1):
        #    if(i%1000 == 0):
        #        sys.stderr.write('\rEvaluated {} users ({0:%})'.format(i , i / usersToEvaluate))

        # Close the pool to avoid memory leaks
        pool.close()

        n_eval = len(usersToEvaluate)
        roc_auc_, precision_, recall_, map_, mrr_, ndcg_ = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # Looping is slightly faster then using the numpy vectorized approach, less data transformation
        for result in resultList:
            roc_auc_ += result[0]
            precision_ += result[1]
            recall_ += result[2]
            map_ += result[3]
            mrr_ += result[4]
            ndcg_ += result[5]


        if (n_eval > 0):
            roc_auc_ = roc_auc_/n_eval
            precision_ = precision_/n_eval
            recall_ = recall_/n_eval
            map_ = map_/n_eval
            mrr_ = mrr_/n_eval
            ndcg_ =  ndcg_/n_eval

        else:
            print("WARNING: No users had a sufficient number of relevant items")


        print("Evaluated {} users".format(n_eval))

        results = {}

        results["AUC"] = roc_auc_
        results["precision"] = precision_
        results["recall"] = recall_
        results["map"] = map_
        results["NDCG"] = ndcg_
        results["MRR"] = mrr_

        return (results)


    def recommend(self, user_id, n=None, exclude_seen=True, filterTopPop = False):
        # Get the user profile
        if self.FastValidation_initialized:
            user_profile = self.URM_train_user_profile[user_id]
        else:
            user_profile = self.URM_train[user_id]

        # compute the scores using the dot product
        if self.sparse_weights:
            print("sparse")
            scores = user_profile.dot(self.W_sparse).toarray().ravel()
            #scores = self.scoresAll[user_id].toarray().ravel()
        else:
            print("complete")
            scores = user_profile.dot(self.W).ravel()
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
            scores /= den

        if exclude_seen:
            scores = self._filter_seen_on_scores(user_id, scores)

        if filterTopPop:
            scores = self._filter_TopPop_on_scores(scores)


        # rank items and mirror column to obtain a ranking in descending score
        ranking = scores.argsort()
        ranking = np.flip(ranking, axis=0)

        """
        if exclude_seen:
            ranking = self._filter_seen(user_id, ranking)

        if filterTopPop:
            ranking = self._filter_TopPop(ranking)
        """

        return ranking[:n]

    def recommendBatch(self, users_to_recommend_list, n=None, exclude_seen=False, relevant_items=None):

        # compute the scores using the dot product
        user_profile_batch = self.URM_train[users_to_recommend_list]

        #user_profile = self.URM_train[user_id]
        if self.sparse_weights:
            scores_array = user_profile_batch.dot(self.W_sparse).toarray()

        else:
            scores_array = user_profile_batch.dot(self.W)

        if self.normalize:
            raise ValueError("Not implemented")

        # To exclude seen items perform a boolean indexing and replace their score with -inf
        # Seen items will be at the bottom of the list but there is no quarantee they'll NOT be
        # recommended
        if exclude_seen:
            if relevant_items==None:
                raise ValueError("Exclude seen selected but no relevant items provided")

            scores_array[relevant_items!=0] = -np.inf

        # rank items and mirror column to obtain a ranking in descending score
        ranking = scores_array.argsort(axis=0)
        ranking = np.fliplr(ranking)

        return ranking[:,:n]



    def recommend_new_user(self, user_profile, n=None, exclude_seen=True):
        # compute the scores using the dot product
        if self.sparse_weights:
            assert user_profile.shape[1] == self.W_sparse.shape[0], 'The number of items does not match!'
            scores = user_profile.dot(self.W_sparse).toarray().ravel()
        else:
            assert user_profile.shape[1] == self.W.shape[0], 'The number of items does not match!'
            scores = user_profile.dot(self.W).ravel()
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
            scores /= den
        # rank items
        ranking = scores.argsort()[::-1]
        if exclude_seen:
            seen = user_profile.indices
            unseen_mask = np.in1d(ranking, seen, assume_unique=True, invert=True)
            ranking = ranking[unseen_mask]
        return ranking[:n]
