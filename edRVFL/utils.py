#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on: 2021/06/26 15:14
@Author: Merc2
'''
import numpy as np
from scipy import stats
from numpy.random import choice
from sklearn.model_selection import train_test_split as Split
from functools import partial

def one_hot(data):
    shape = (data.size, int(data.max()+1))
    one_hot = np.zeros(shape)
    rows = np.arange(data.size)
    one_hot[rows, data.astype(int)] = 1
    return one_hot

def cross_entropy(predictions, targets, weight=None):
    N = predictions.shape[0]
    predictions += 1e-6
    if weight is None:
        ce = -np.sum(targets * np.log(predictions)) / N
    else:
        ce = -np.sum(targets * np.log(predictions)*weight[...,np.newaxis]) / N
    return ce

def instance_cross_entropy(predictions, targets, weight=None):
    predictions += 1e-6
    if weight is None:
        ce = -targets * np.log(predictions)
    else:
        ce = -targets * np.log(predictions)*weight[...,np.newaxis]
    return ce

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


# def cal_weight(score, label):
#     correct_score = np.max(score*label,1)
#     wrong_max_score = np.max(score*(1-label),1)
#     # mask = correct_score > wrong_max_score
#     # weight = 1 / np.exp(correct_score-wrong_max_score)
#     weight = sigmoid(wrong_max_score-correct_score)
#     return weight

def cal_weight(score, label):
    correct_score = np.max(score*label,1)
    wrong_max_score = np.max(score*(1-label),1)
    # mask = correct_score >= wrong_max_score
    # weight = 1 / np.exp(correct_score-wrong_max_score)
    weight = sigmoid(correct_score-wrong_max_score)
    weight = weight / weight.sum() * weight.shape[0]
    # if mask.all() == True:
    #     weight[mask] = 1
    # else:
    #     weight[mask] = 0.
    #     weight = weight / weight.sum() * weight.shape[0]
    #     weight[mask] = 0.01
    return weight

def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * np.where(x >= 0, x, alpha * (np.exp(x) - 1))

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_top_n_layer(loss, full_y, n, k):
    full_y = np.stack([full_y]*k)
    best_n_index = (-full_y * np.log(loss)).max(2).argsort(0)[:n,:]
    # best_n_index = (loss*full_y).max(2).argsort(0)[-n:,:]
    return best_n_index


def distance(x1, x2):
    num_test = x1.shape[0]
    num_train = x2.shape[0]
    dists = np.zeros((num_test,num_train))
    dists = np.sqrt(get_norm(x1, num_train).T + get_norm(x2,num_test)- 2*np.dot(x1,x2.T))
    dists = np.nan_to_num(dists)
    return dists

def instance_distance(xy1, xy2):
    P = np.add.outer(np.sum(xy1**2, axis=1), np.sum(xy2**2, axis=1))
    N = np.dot(xy1, xy2.T)
    dists = np.sqrt(P - 2*N)
    dists = np.nan_to_num(dists)
    return dists

def get_norm(x, num):
    return np.ones((num,1)) * np.sum(np.square(x),axis=1)

def weight_based_split(full_X, next_X, full_y, weight):
    raw_weight = weight
    if (weight == 1).all():
        weight = weight
    else:
        weight = 6.*(weight - np.min(weight))/np.ptp(weight)-3
    proba = stats.norm.cdf(weight)
    proba /= proba.sum()
    num_train = int(full_X.shape[0] * 0.8)
    draw_train = choice(full_X.shape[0] , num_train, replace=False,p=proba)
    draw_test = np.setdiff1d(range(full_X.shape[0]), draw_train)
    trainX, evalX = full_X[draw_train,:], full_X[draw_test,:]
    hidden_train_X, hidden_eval_X = next_X[draw_train,:], next_X[draw_test,:]
    weight_train, weight_eval = raw_weight[draw_train], raw_weight[draw_test]
    trainY, evalY = full_y[draw_train], full_y[draw_test]
    return trainX, evalX, hidden_train_X, hidden_eval_X, weight_train, weight_eval, trainY, evalY


def score_based_split(full_X, next_X, full_Y, loss, weight):
    num_clusters = 5
    num_samples = int(len(full_X)*0.05)
    loss = instance_cross_entropy(softmax(loss),full_Y).sum(-1)
    rank = np.argsort(loss)[::-1]
    # cat_X = np.concatenate((full_X,next_X),1)
    centers = []
    train_X, test_X, next_train_X, next_test_X, train_Y, test_Y = [],[],[],[],[],[]
    for i in range(num_clusters+1):
        if i == num_clusters-1:
            train_X.append(full_X*0.01)
            next_train_X.append(next_X*0.01)
            train_Y.append(full_Y)
            break
        center_tmp = np.concatenate((full_X,next_X),1)[rank[0]]
        dists_tmp = instance_distance(center_tmp[np.newaxis,:],np.concatenate((full_X,next_X),1))
        rank_tmp = np.argsort(dists_tmp).squeeze()[:num_samples]
        centers.append(center_tmp)
        mask = np.ones_like(loss,dtype=bool)
        mask[rank_tmp] = False
        train_X_tmp, test_X_tmp, next_train_X_tmp, next_test_X_tmp, \
            train_Y_tmp, test_Y_tmp = Split(full_X[~mask,:]*weight[~mask,np.newaxis],next_X[~mask,:]*weight[~mask,np.newaxis],full_Y[~mask,:], test_size=0.5)
        train_X.append(train_X_tmp)
        test_X.append(test_X_tmp)
        next_train_X.append(next_train_X_tmp)
        next_test_X.append(next_test_X_tmp)
        train_Y.append(train_Y_tmp)
        test_Y.append(test_Y_tmp)
        loss = loss[mask]
        weight = weight[mask]
        full_X = full_X[mask,:]
        next_X = next_X[mask,:]
        full_Y = full_Y[mask,:]

        rank = np.argsort(loss)[::-1]

    return np.concatenate(train_X), np.concatenate(test_X), \
        np.concatenate(next_train_X), np.concatenate(next_test_X), \
        np.concatenate(train_Y), np.concatenate(test_Y), \
        np.concatenate(centers)
        

def score_based_split_nodrop(full_X, next_X, full_Y, loss, weight):
    num_clusters = 5
    num_samples = int(len(full_X)*0.05)
    loss = instance_cross_entropy(softmax(loss),full_Y).sum(-1)
    rank = np.argsort(loss)[::-1]
    centers = []
    train_X, test_X, next_train_X, next_test_X, train_Y, test_Y = [],[],[],[],[],[]
    for i in range(num_clusters+1):
        if i == num_clusters-1:
            train_X.append(full_X*0.01)
            next_train_X.append(next_X*0.01)
            train_Y.append(full_Y)
            break
        center_tmp = np.concatenate((full_X,next_X),1)[rank[i]]
        dists_tmp = instance_distance(center_tmp[np.newaxis,:],np.concatenate((full_X,next_X),1))
        rank_tmp = np.argsort(dists_tmp).squeeze()[:num_samples]
        centers.append(center_tmp)
        mask = np.ones_like(loss,dtype=bool)
        mask[rank_tmp] = False
        train_X_tmp, test_X_tmp, next_train_X_tmp, next_test_X_tmp, \
            train_Y_tmp, test_Y_tmp = Split(full_X[~mask,:]*weight[~mask,np.newaxis],next_X[~mask,:]*weight[~mask,np.newaxis],full_Y[~mask,:], test_size=0.5)
        train_X.append(train_X_tmp)
        test_X.append(test_X_tmp)
        next_train_X.append(next_train_X_tmp)
        next_test_X.append(next_test_X_tmp)
        train_Y.append(train_Y_tmp)
        test_Y.append(test_Y_tmp)

        rank = np.argsort(loss)[::-1]

    return np.concatenate(train_X), np.concatenate(test_X), \
        np.concatenate(next_train_X), np.concatenate(next_test_X), \
        np.concatenate(train_Y), np.concatenate(test_Y), \
        np.concatenate(centers)