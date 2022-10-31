#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on: 2021/08/17 21:40
@Author: Merc2
'''
import numpy as np
from utils import *
from edRVFL.RVFL import OneShot
from sklearn.model_selection import train_test_split as Split
from sklearn.feature_selection import VarianceThreshold
import openml
import argparse
from time import time
from data.uci import UCIDataset

parser = argparse.ArgumentParser(description='Task No.')
parser.add_argument('--index', help='foo help', default=146606) # higgs
args = parser.parse_args()

# openml.study.get_suite(218)

# task = openml.tasks.get_task(7593) # covertype
# task = openml.tasks.get_task(168331) # volkert 
# task = openml.tasks.get_task(146821) # car
# task = openml.tasks.get_task(168909) 
# task = openml.tasks.get_task(3841) # feats
# task = openml.tasks.get_task(31)  # credit-g
# task = openml.tasks.get_task(3)  # krkp
# task = openml.tasks.get_task(7592)  # adult
# task = openml.tasks.get_task(145847)  # hill-vally
# task = openml.tasks.get_task(9955)  # plant-shape
# task = openml.tasks.get_task(146217)  # wine-red
task = openml.tasks.get_task(145681)  # wine-white
# task = openml.tasks.get_task(args.index)  # adult
_, CV_NUM, _ = task.get_split_dimensions()
N_TYPES = len(task.class_labels)
X,y = task.get_X_and_y()
selector = VarianceThreshold()
X = selector.fit_transform(X)
result = np.zeros(CV_NUM)
accs = []
for i in range(CV_NUM):
    network = OneShot(N_TYPES,0,20)
    a = time()
    train_indices, test_indices = task.get_train_test_split_indices(repeat=0, fold=i, sample=0)
    
    full_X = X[train_indices]
    full_Y = y[train_indices]
    testX = X[test_indices]
    testY = y[test_indices]
    shape = full_X.shape[1]
    if np.isnan(full_X).any():
        idx = np.isnan(full_X).any(1)
        # idx = np.isnan(full_X)
        # full_X[idx] = 0.01
        full_X = full_X[~idx]
        full_Y = full_Y[~idx]
        # full_Y = full_Y[~idx]
    if np.isnan(testX).any():
        idx = np.isnan(testX).any(1)
        # idx = np.isnan(testX)
        # testX[idx] = 0.01
        testX = testX[~idx]
        testY = testY[~idx]
    print(f"Data Shape:{full_X.shape}\t Classes:{N_TYPES}")
    assert full_X.shape[1] == shape
    print(f"Loading Time:{time()-a:.2f}")
    acc = network.loop(full_X, one_hot(full_Y), testX, one_hot(testY))
    accs.append(acc)
print(f"Task Index:{args.index}")
print(f"Data Shape:{full_X.shape}\t Classes:{N_TYPES}")
print(f"Final:{np.array(accs).mean():.4f}")
print(f"Final:{np.array(accs).mean():.4f}")
print(f"Final:{np.array(accs).mean():.4f}")
# print(f"Avg acc:{result.mean():.3f}\tStd {result.std():.3f}")
# print(f"Avg acc:{result.mean():.3f}\tStd {result.std():.3f}")
# print(f"Avg acc:{result.mean():.3f}\tStd {result.std():.3f}")