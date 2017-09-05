# -*- coding: utf-8 -*-
"""
Politecnico di Milano.
base.py

Description: This file contains the implementation of a matrix format checker and
             the definition of the Recommender abstract class.

Created by: Massimo Quadrana.
Modified by Fernando Pérez.

Last modified on 25/03/2017.
"""
import numpy as np
import scipy.sparse as sps


def check_matrix(X, format='csc', dtype=np.float32):
    """Checks if a sparse matrix is an instance of a specific Scipy.Sparse format.

        This function checks if a matrix is already an instance as in the
        Scipy.Sparse module, if it is not, it changes its format.

        Args:
            * X: the matrix to be checked.
            * format: specifies the format in which to check the matrix.
            * dtype: specifies the data type of the elements stored in the matrix.

        Args type:
            * X: Scipy.Sparse
            * format: str
            * dtype: numpy.dtype

        Returns:
            An instance of a Scipy.Sparse matrix with the format as in *format*
            and with the dtype as in *dtype*.
    """
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


class Recommender(object):
    """Class that serves as an abstract base for all our recommender classes.

        It includes some basic operations as getting the users ratings, getting
        the ratings for an item, and filtering seen items for a user.

        Attributes:
            * dataset: Represents the dataset which the recommender will be trained.

        Attributes types:
            * dataset: Scipy.Sparse
    """

    def __init__(self):
        """Constructor of the class.

            It sets the attribute *dataset* to None.
        """
        super(Recommender, self).__init__()
        self.dataset = None

    def _get_user_ratings(self, user_id):
        """Returns the ratings for a given user in the dataset.

            Args:
                * user_id: Represents the user index in the dataset.

            Args type:
                * user_id: int or list of int.

            Returns:
                Returns a Scipy.Sparse matrix that contains the ratings made
                by the user in user_id.
        """
        return self.dataset[user_id]

    def _get_item_ratings(self, item_id):
        """Returns the ratings given to an item in the dataset.

            Args:
                * item_id: Represents the item index in the dataset.

            Args type:
                * item_id: int or list of int.

            Returns:
                Returns a Scipy.Sparse matrix that contains the ratings given
                to the item in item_id.
        """
        return self.dataset[:, item_id]

    def _filter_seen(self, user_id, ranking):
        """Filters from a list the items that are already seen by the user.

            Args:
                * user_id: Represents the item index in the dataset.
                * ranking: a top-N recommendation list.

            Args type:
                * item_id: int.
                * ranking: list of int.

            Returns:
                Returns a ranking list without the indices of the items already
                seen by the user in the training set.
        """
        user_profile = self._get_user_ratings(user_id)
        seen = user_profile.indices
        unseen_mask = np.in1d(ranking, seen, assume_unique=True, invert=True)
        return ranking[unseen_mask]

    def fit(self, X):
        pass

    def recommend(self, user_id, n=None, exclude_seen=True):
        pass

    def label(self, unlabeled_list, n=None, exclude_seen=True, p_most=1, n_most=3):
        pass

    def predict(self, user_id):
        pass
