# -*- coding: utf-8 -*-
'''
Politecnico di Milano.
data_utils.py

Description: This file contains the definition and implementation of a file-reading
             function for datasets and a dataframe-to-csr matrix function.

Created by: Massimo Quadrana.
Modified by: Fernando Pérez.

Last modified on 05/09/2017.
'''

import numpy as np
import scipy.sparse as sps
import pandas as pd

import logging
import csv

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

    :param path: where the file is located
    :param header: number of the row that is the header.
    :param columns: name of the columns.
    :param make_binary: convert the explicit dataset into an implicit dataset.
    :param binary_th: which value would be considered as 1.
    :param user_key: the key to locate the user indices.
    :param item_key: the key to locate the item indices.
    :param rating_key: the key to locate the rating indices.
    :param sep: the separator in the csv file.
    :param user_to_idx: list that represents the which user has which index.
    :param item_to_idx: list that represents the which item has which index.
    :return: a Pandas.Dataframe with the dataset read, the indices of items, and
             the indices of users.
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
    """Reads the dataset and converts it into an Compressed Row Sparse format matrix.

        Args:
            * df: represents the dataset.
            * nrows: number of users.
            * ncols: number of items.
            * is_binary: implicit or explicit dataset
            * user_key: key for locating user indices.
            * item_key: key for locating item indices.
            * rating_key: key for locating the ratings.

        Args type:
            * df: Pandas.Dataframe instance.
            * nrows: int
            * ncols: int
            * is_binary: bool
            * user_key: str
            * item_key: str
            * rating_key: str

        Returns:
            An instance of Scipy.Sparse.CsrMatrix that holds in the rows the
            user indices, in the columns the item indices, and the ratings
            as values.
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


def df_to_lil(df, nrows, ncols, is_binary=False, user_key='user_idx', item_key='item_idx', rating_key='rating'):
    """Reads the dataset and converts it into a List of List sparse format matrix.

        Args:
            * df: represents the dataset.
            * nrows: number of users.
            * ncols: number of items.
            * is_binary: implicit or explicit dataset
            * user_key: key for locating user indices.
            * item_key: key for locating item indices.
            * rating_key: key for locating the ratings.

        Args type:
            * df: Pandas.Dataframe instance.
            * nrows: int
            * ncols: int
            * is_binary: bool
            * user_key: str
            * item_key: str
            * rating_key: str

        Returns:
            An instance of Scipy.Sparse.CsrMatrix that holds in the rows the
            user indices, in the columns the item indices, and the ratings
            as values.
    """

    rows = df[user_key].values
    columns = df[item_key].values
    ratings = df[rating_key].values if not is_binary else np.ones(df.shape[0])
    # use floats by default
    ratings = ratings.astype(np.float32)
    shape = (nrows, ncols)

    # Using the 3rd constructor of lil_matrix.
    # This returns an empty lil_matrix.
    # reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.html#scipy.sparse.lil_matrix
    data = sps.lil_matrix(shape)
    for i in range(len(ratings)):
        data[ rows[i], columns[i] ] = ratings[i]

    return data

def df_to_dok(df, nrows, ncols, is_binary=False, user_key='user_idx', item_key='item_idx', rating_key='rating'):
    """Reads the dataset and converts it into a Dictionary of Keys format matrix.

        Args:
            * df: represents the dataset.
            * nrows: number of users.
            * ncols: number of items.
            * is_binary: implicit or explicit dataset
            * user_key: key for locating user indices.
            * item_key: key for locating item indices.
            * rating_key: key for locating the ratings.

        Args type:
            * df: Pandas.Dataframe instance.
            * nrows: int
            * ncols: int
            * is_binary: bool
            * user_key: str
            * item_key: str
            * rating_key: str

        Returns:
            An instance of Scipy.Sparse.CsrMatrix that holds in the rows the
            user indices, in the columns the item indices, and the ratings
            as values.
    """

    rows = df[user_key].values
    columns = df[item_key].values
    ratings = df[rating_key].values if not is_binary else np.ones(df.shape[0])
    # use floats by default
    ratings = ratings.astype(np.float32)
    shape = (nrows, ncols)

    # Using the 3rd constructor of dok_matrix.
    # This returns an empty dok_matrix.
    # reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.html#scipy.sparse.lil_matrix
    data = sps.dok_matrix(shape)
    for u,i,r in zip(rows,columns,ratings):
        data.update({(u,i):r})

    return data

def results_to_df(filepath, type_res="evaluation"):
    """Reads the results file and transforms it into a dataframe.

        Args:
            * filepath: where the results are located as a .csv file.
            * type_res: the type of results file, it can be `numberlabeled` ,
                        `label_comparison` or None to indicate evaluation.

        Args type:
            * filepath: str
            * type_res: str

        Returns:
            An instance of Pandas.Dataframe that holds the results
    """
    header = 0
    sep = ' '
    if (type_res == "evaluation"):
        available_metrics = ['rmse','roc_auc','precision', 'recall', 'map', 'mrr', 'ndcg']
        columns = ['cotraining','iterations', '@k', 'recommender'] + available_metrics

    elif (type_res == "numberlabeled"):
        columns = ['iteration','recommender', 'pos_labeled', 'neg_labeled','total_labeled']

    elif (type_res == "label_comparison"):
        columns = ['iteration',
                   'both_positive', 'both_negative', 'both_neutral',
                   'pos_only_first', 'neg_only_first', 'neutral_only_first',
                   'pos_only_second', 'neg_only_second', 'neutral_only_second']

    elif (type_res == "item_pop_bin"):
        columns = ['iteration', 'pop_bin_type', 'recommender', 'bin_0', 'bin_1',
                    'bin_2', 'bin_3', 'bin_4', 'bin_5', 'bin_6', 'bin_7',
                    'bin_8', 'bin_9']
    else:
        return None

    results = pd.read_csv(filepath, header=header, names=columns, sep=sep)
    return results

def results_to_file(filepath,
                    header=False,
                    cotraining=False,
                    iterations=0,
                    recommender1=None,
                    evaluation1=None,
                    at=5
                ):
    """Writes into a csv file the results of a cotraining iteration.

        Args:
            * filepath: where the results will be located as a .csv file.
            * header: write the header or not.
            * cotraining: if cotraining was applied or not.
            * iterations: number of cotraining iteration.
            * recommender1: first recommender that was evaluated.
            * evaluation1: the evaluation of the first recommender for each metric.
            # at: the size of the top-N recommendation list.

        Args type:
            * filepath: str
            * header: bool
            * cotraining: bool
            * iterations: int
            * recommender1: Recommender instance.
            * evaluation1: list of float
            # at: int
    """
    with open(filepath, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        if header:
            available_metrics = ['rmse','roc_auc','precision', 'recall', 'map', 'mrr', 'ndcg']
            columns = ['cotraining','iterations', '@k', 'recommender'] + available_metrics
            csvwriter.writerow(columns)
        else:
            csvwriter.writerow([cotraining,
                             iterations if cotraining else "NaN",
                             at,
                             recommender1.__str__()
                             ] +
                             evaluation1)

        csvfile.close()
