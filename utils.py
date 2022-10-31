#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on: 2021/06/26 15:14
@Author: Merc2
'''
import numpy as np

def one_hot(data):
    shape = (data.size, int(data.max()+1))
    one_hot = np.zeros(shape)
    rows = np.arange(data.size)
    one_hot[rows, data.astype(int)] = 1
    return one_hot

def cross_entropy(predictions, targets):
    N = predictions.shape[0]
    ce = -np.sum(targets * np.log(predictions)) / N
    return ce
    
def cal_weight(score, label):
    correct_score = np.max(score*label,1)
    wrong_max_score = np.max(score*(1-label),1)
    # mask = correct_score > wrong_max_score
    weight = 1 / np.exp(correct_score-wrong_max_score)
    return weight


def convert_matrix(matrix, force_categorical=None, force_numerical=None):
    """
    Covert the matrix in a matrix of floats.
    Use one-hot-encoding for categorical features.
    Features are categorical if at least one item is a string or it has more
        unique values than specified numerical_min_unique_values
        or it is listed in force_categorical.
    
    Arguments:
        matrix: The matrix to convert.
        force_cateogrical: The list of column indizes, which should be categorical.
        force_numerical: The list of column indizes, which should be numerical.
        
    Result:
        result: the converted matrix
        categorical: boolean vector, that specifies which columns are categorical
    """
    num_rows = len(matrix)
    is_categorical = []
    len_values_and_indices = []
    result_width = 0
    
    # iterate over the columns and get some data
    for i in range(matrix.shape[1]):
        
        # check if it is categorical or numerical
        matrix_column = matrix[0:num_rows, i]
        if matrix.dtype == np.dtype("object"):
            values_occurred = dict()
            values = []
            indices = []
            for v in matrix_column:
                if v not in values_occurred:
                    values_occurred[v] = len(values)
                    values.append(v)
                indices.append(values_occurred[v])
            indices = np.array(indices)
            values = np.array(values, dtype=object)
            nan_indices = np.array([i for i, n in enumerate(matrix_column) if n == np.nan])
            valid_value_indices = np.array([i for i, n in enumerate(values) if n != np.nan])
        else:
            values, indices = np.unique(matrix_column, return_inverse=True)
            nan_indices = np.argwhere(np.isnan(matrix_column)).flatten()
            valid_value_indices = np.argwhere(~np.isnan(values)).flatten()

        # check for missing values
        # nan values are additional category in categorical features
        if len(nan_indices) > 0:
            values = np.append(values[valid_value_indices], np.nan)
            indices[nan_indices] = values.shape[0] - 1         

        len_values_and_indices.append((len(values), indices))
        if len(values) == 1:
            is_categorical.append(None)
        elif i in force_categorical or i not in force_numerical and (
                len(values) < 1e-7 or
                any(type(value) is str for value in values)):
            # column is categorical
            is_categorical.append(True)
            result_width += 1
        else:
            # column is numerical
            is_categorical.append(False)
            result_width += 1

    # fill the result
    result = np.zeros(shape=(num_rows, result_width), dtype='float32', order='F')
    j = 0
    for i, is_cat in enumerate(is_categorical):
        len_values, indices = len_values_and_indices[i]
        if len_values == 1:
            continue
        if is_cat:
            # column is categorical: convert to int
            result[:, j] = indices
            j += 1
        else:
            # column is numerical
            result[:, j] = matrix[:, i]
            j += 1

    return result.astype('float32', copy=False), [x for x in is_categorical if x is not None]
    