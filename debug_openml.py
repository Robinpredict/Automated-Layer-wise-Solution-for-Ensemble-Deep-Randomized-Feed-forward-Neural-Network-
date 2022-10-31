#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on: 2021/07/23 20:55
@Author: Merc2
'''

import numpy as np
from utils import *
from edRVFL.RVFL import Greedy
from sklearn.model_selection import train_test_split as Split
import openml


# openml.study.get_suite(218)

task = openml.tasks.get_task(7593) # covertype
#task = openml.tasks.get_task(168331) # volkert 
# task = openml.tasks.get_task(146821) # car
_, CV_NUM, _ = task.get_split_dimensions()
N_TYPES = len(task.class_labels)
X,y = task.get_X_and_y()
result = np.zeros(CV_NUM)
for i in range(1):
    network = Greedy(N_TYPES,0,15)
    train_indices, test_indices = task.get_train_test_split_indices(repeat=0, fold=i, sample=0)
    full_X = X[train_indices]
    full_Y = y[train_indices]
    trainX, evalX, trainY, evalY = Split(full_X, full_Y)
    testX = X[test_indices]
    testY = y[test_indices]
    network.train_layer_by_layer(trainX, one_hot(trainY), evalX, one_hot(evalY))
    acc_test = network.retrain_for_test(full_X, one_hot(full_Y), testX, one_hot(testY))
    result[i] = max(acc_test)
    print(f"Best Test Acc:{max(acc_test):.3f}")
print("Volkert")
print(f"Avg acc:{result.mean():.3f}\tStd {result.std():.3f}")
print(f"Avg acc:{result.mean():.3f}\tStd {result.std():.3f}")
print(f"Avg acc:{result.mean():.3f}\tStd {result.std():.3f}")