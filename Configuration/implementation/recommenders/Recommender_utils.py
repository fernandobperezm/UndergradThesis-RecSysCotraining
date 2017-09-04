#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
import scipy.sparse as sps
import time

def check_matrix(X, format='csc', dtype=np.float32):
    if format == 'csc' and not isinstance(X, sps.csc_matrix):
        return X.tocsc().astype(dtype)
    elif format == 'csr' and not isinstance(X, sps.csr_matrix):
        return X.tocsr().astype(dtype)
    elif format == 'coo' and not isinstance(X, sps.coo_matrix):
        return X.tocoo().astype(dtype)
    elif format == 'dok' and not isinstance(X, sps.dok_matrix):
        return X.todok().astype(dtype)
    elif format == 'bsr' and not isinstance(X, sps.bsr_matrix):
        return X.tobsr().astype(dtype)
    elif format == 'dia' and not isinstance(X, sps.dia_matrix):
        return X.todia().astype(dtype)
    elif format == 'lil' and not isinstance(X, sps.lil_matrix):
        return X.tolil().astype(dtype)
    else:
        return X.astype(dtype)


def similarityMatrixTopK(item_weights, forceSparseOutput = True, k=100):

    assert (item_weights.shape[0] == item_weights.shape[1]), "selectTopK: ItemWeights is not a square matrix"

    start_time = time.time()
    print("Generating topK matrix")

    nitems = item_weights.shape[1]

    # for each column, keep only the top-k scored items
    sparse_weights = not isinstance(item_weights, np.ndarray)

    if not sparse_weights:

        idx_sorted = np.argsort(item_weights, axis=0)  # sort data inside each column

        W = item_weights.copy()
        # index of the items that don't belong to the top-k similar items of each column
        not_top_k = idx_sorted[:-k, :]
        # use numpy fancy indexing to zero-out the values in sim without using a for loop
        W[not_top_k, np.arange(nitems)] = 0.0

        if forceSparseOutput:
            W_sparse = sps.csr_matrix(W, shape=(nitems, nitems))
            return W_sparse

        print("TopK matrix generated in {:.2f} seconds".format(time.time()-start_time))

        return W

    else:
        # iterate over each column and keep only the top-k similar items
        values, rows, cols = [], [], []

        item_weights = item_weights.tocsc()

        for item_idx in range(nitems):

            item_column = item_weights[:,item_idx]
            dataValue = item_column.data
            dataRow = item_column.indices

            idx_sorted = np.argsort(dataValue)  # sort by column
            top_k_idx = idx_sorted[-k:]

            values.extend(dataValue[top_k_idx])
            rows.extend(dataRow[top_k_idx])
            cols.extend(np.ones(len(top_k_idx), dtype=np.int) * item_idx)

        # During testing CSR is faster
        W_sparse = sps.csr_matrix((values, (rows, cols)), shape=(nitems, nitems), dtype=np.float32)

        print("TopK matrix generated in {:.2f} seconds".format(time.time() - start_time))

        return W_sparse

def areURMequals(URM1, URM2):

    if(URM1.shape != URM2.shape):
        return False

    return (URM1-URM2).nnz ==0


def removeTopPop(URM_1, URM_2=None, percentageToRemove=0.2):
    """
    Remove the top popular items from the matrix
    :param URM_1: user X items
    :param URM_2: user X items
    :param percentageToRemove: value 1 corresponds to 100%
    :return: URM: user X selectedItems, obtained from URM_1
             Array: itemMappings[selectedItemIndex] = originalItemIndex
             Array: removedItems
    """


    item_pop = URM_1.sum(axis=0)  # this command returns a numpy.matrix of size (1, nitems)

    if URM_2 != None:

        assert URM_2.shape[1] == URM_1.shape[1], \
            "The two URM do not contain the same number of columns, URM_1 has {}, URM_2 has {}".format(URM_1.shape[1], URM_2.shape[1])

        item_pop += URM_2.sum(axis=0)


    item_pop = np.asarray(item_pop).squeeze()  # necessary to convert it into a numpy.array of size (nitems,)
    popularItemsSorted = np.argsort(item_pop)[::-1]

    numItemsToRemove = int(len(popularItemsSorted)*percentageToRemove)

    # Choose which columns to keep
    itemMask = np.in1d(np.arange(len(popularItemsSorted)), popularItemsSorted[:numItemsToRemove],  invert=True)

    # Map the column index of the new URM to the original ItemID
    itemMappings = np.arange(len(popularItemsSorted))[itemMask]

    removedItems = np.arange(len(popularItemsSorted))[np.logical_not(itemMask)]

    return URM_1[:,itemMask], itemMappings, removedItems



def loadCSVintoSparse (filePath, header = False):

    values, rows, cols = [], [], []

    fileHandle = open(filePath, "r")
    numCells = 0

    if header:
        fileHandle.readline()

    for line in fileHandle:
        numCells += 1
        if (numCells % 1000000 == 0):
            print("Processed {} cells".format(numCells))

        if (len(line)) > 1:
            line = line.split(",")

            value = line[2].replace("\n", "")

            if not value == "0" and not value == "NaN":
                rows.append(int(line[0]))
                cols.append(int(line[1]))
                values.append(float(value))

    return  sps.csr_matrix((values, (rows, cols)), dtype=np.float32)
