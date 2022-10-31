#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on: 2021/07/23 20:55
@Author: Merc2
'''
import pickle
import numpy as np
from easydict import EasyDict
from utils import *
from edRVFL.RVFL import Greedy
from bohbs import BOHB
import bohbs.configspace as cs
from functools import partial
from data.uci import UCIDataset
from tqdm import tqdm
import argparse
import pickle
from pathlib import Path


parser = argparse.ArgumentParser(description='W2edRVFL')
parser.add_argument('--data', type=str, default='glass')
args = parser.parse_args()

Path.mkdir(Path.cwd()/'ckpt', exist_ok=True)
print(f"Data:{args.data}")
# data_name = 'arrhythmia'
dataset = UCIDataset(args.data)
CV_NUM = dataset.n_CV
N_TYPES = dataset.n_types

result = np.zeros(CV_NUM)
for i in range(CV_NUM):
    network = Greedy(N_TYPES,0,15)
    trainX, trainY, evalX, evalY, testX, testY, full_X, full_Y = dataset.getitem(i)
    network.train_layer_by_layer(trainX, trainY, evalX, evalY)
    acc_test = network.retrain_for_test(full_X, full_Y, testX, testY)
    result[i] = max(acc_test)
    print(f"Best Test Acc:{max(acc_test):.3f}")
print(f"Avg acc:{result.mean():.3f}\tStd {result.std():.3f}")