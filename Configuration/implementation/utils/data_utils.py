# -*- coding: utf-8 -*-
'''
Politecnico di Milano.
item_knn.py

Description: This file contains the definition and implementation of a file-reading
             function for datasets and a dataframe-to-csr matrix function.

Created by: Massimo Quadrana.
Modified by Fernando Pérez.

Last modified on 25/03/2017.
'''

import numpy as np
import scipy.sparse as sps
import pandas as pd

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")


def read_dataset(path,
                 header=None,
                 columns=None,
                 make_binary=False,
                 binary_th=4.0,
                 user_key='user_id',
                 item_key='item_id',
                 rating_key='rating',
                 sep=',',
                 user_to_idx=None,
                 item_to_idx=None):
    """

    :param path:
    :param header:
    :param columns:
    :param make_binary:
    :param binary_th:
    :param user_key:
    :param item_key:
    :param rating_key:
    :param sep:
    :param user_to_idx:
    :param item_to_idx:
    :return:
    """
    data = pd.read_csv(path, header=header, names=columns, sep=sep)
    logger.info('Columns: {}'.format(data.columns.values))
    if make_binary:
        logger.info('Converting the dataset to binary feedback')
        logger.info('Positive feedback threshold (>= rule): {}'.format(binary_th))
        data = data.ix[data[rating_key] >= binary_th]
        data = data.reset_index()  # reset the index to remove the 'holes' in the DataFrame
    # build user and item maps
    if item_to_idx is None:
        if 'item_idx' not in data.columns:
            # these are used to map ids to indexes starting from 0 to nitems (or nusers)
            items = data[item_key].unique()
            if item_to_idx is None:
                item_to_idx = pd.Series(data=np.arange(len(items)), index=items)
            #  map ids to indices
            data['item_idx'] = item_to_idx[data[item_key].values].values
        else:
            aux = data[[item_key, 'item_idx']].drop_duplicates()
            item_to_idx = pd.Series(index=aux[0], data=aux[1])
    else:
        #  map ids to indices
        data['item_idx'] = item_to_idx[data[item_key].values].values
        if np.any(np.isnan(data['item_idx'])):
            logger.error('NaN values in item_idx (new items?)')
            raise RuntimeError('NaN values in item_idx')
    if user_to_idx is None:
        if 'user_idx' not in data.columns:
            # these are used to map ids to indexes starting from 0 to nusers (or nusers)
            users = data[user_key].unique()
            if user_to_idx is None:
                user_to_idx = pd.Series(data=np.arange(len(users)), index=users)
            #  map ids to indices
            data['user_idx'] = user_to_idx[data[user_key].values].values
        else:
            aux = data[[user_key, 'user_idx']].drop_duplicates()
            user_to_idx = pd.Series(index=aux[0], data=aux[1])
    else:
        #  map ids to indices
        data['user_idx'] = user_to_idx[data[user_key].values].values
        if np.any(np.isnan(data['user_idx'])):
            logger.error('NaN values in user_idx (new users?)')
            raise RuntimeError('NaN values in user_idx')

    return data, item_to_idx, user_to_idx


def df_to_csr(df, nrows, ncols, is_binary=False, user_key='user_idx', item_key='item_idx', rating_key='rating'):
    """
    Convert a pandas DataFrame to a scipy.sparse.csr_matrix
    """

    rows = df[user_key].values
    columns = df[item_key].values
    ratings = df[rating_key].values if not is_binary else np.ones(df.shape[0])
    # use floats by default
    ratings = ratings.astype(np.float32)
    shape = (nrows, ncols)
    # using the 4th constructor of csr_matrix
    # reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
    return sps.csr_matrix((ratings, (rows, columns)), shape=shape)

def results_to_file(filepath,
                    evaluation_type,
                    cotraining=False,
                    iterations=0,
                    recommender1=None,
                    recommender2=None,
                    evaluation1=None,
                    evaluation2=None,
                    at=5
                ):

    with open(filepath, 'w') as f:
        f.write("Evaluation type: {}\n".format(evaluation_type))
        f.write("Uses Co-Training: {}\n".format("Yes" if cotraining else "No"))
        if cotraining:
            f.write("# of Co-Training Iterations: {}\n".format(iterations))

        # Recommender 1.
        f.write("Recommender 1: \n")
        f.write("\tName: {}\n\n".format(recommender1.__str__()))
        f.write("\tEvaluation:\n")
        f.write('\t\tROC-AUC: {:.4f}\n'.format(evaluation1[0]))
        f.write('\t\tPrecision@{}: {:.4f}\n'.format(at, evaluation1[1]))
        f.write('\t\tRecall@{}: {:.4f}\n'.format(at, evaluation1[2]))
        f.write('\t\tMAP@{}: {:.4f}\n'.format(at, evaluation1[3]))
        f.write('\t\tMRR@{}: {:.4f}\n'.format(at, evaluation1[4]))
        f.write('\t\tNDCG@{}: {:.4f}\n'.format(at, evaluation1[5]))

        # Recommender 2.
        f.write("Recommender 2: \n")
        f.write("\tName: {}\n\n".format(recommender2.__str__()))
        f.write("\tEvaluation:\n")
        f.write('\t\tROC-AUC: {:.4f}\n'.format(evaluation2[0]))
        f.write('\t\tPrecision@{}: {:.4f}\n'.format(at, evaluation2[1]))
        f.write('\t\tRecall@{}: {:.4f}\n'.format(at, evaluation2[2]))
        f.write('\t\tMAP@{}: {:.4f}\n'.format(at, evaluation2[3]))
        f.write('\t\tMRR@{}: {:.4f}\n'.format(at, evaluation2[4]))
        f.write('\t\tNDCG@{}: {:.4f}\n'.format(at, evaluation2[5]))

        f.close()
