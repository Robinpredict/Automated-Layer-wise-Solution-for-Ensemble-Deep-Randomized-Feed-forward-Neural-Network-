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
from pathlib import Path

import logging


def get_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s', "%Y-%m-%d %H:%M:%S")
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)

    return logging.getLogger(logger_name)


# openml.study.get_suite(218)
parser = argparse.ArgumentParser(description='edRVFL')
# parser.add_argument('--data', type=str, default='wine-quality-white')
parser.add_argument('-s', type=int, default=0)
parser.add_argument('-e', type=int, default=8)
args = parser.parse_args()
data_list = sorted(Path("/home/hu/database/UCIdata/").glob('*'))
# dataname = data_list[args.data].stem
# print(f'Data Name is : {dataname}')

Path("./exps/").mkdir(exist_ok=True)
for item in range(args.s, args.e):
    dataname = data_list[item].stem
    # logging.basicConfig(filename=f'./exps/{item}_{dataname}.log', filemode='a',level=logging.DEBUG, format=LOG_FORMAT)

    logger = get_logger(logger_name=f'{item}_{dataname}',log_file=f'./exps/{item}_{dataname}.log')
    logger.info(f'Data Name is : {dataname}')
    # dataset = UCIDataset(args.data)
    dataset = UCIDataset(dataname)
    CV_NUM = dataset.n_CV
    N_TYPES = dataset.n_types

    result = np.zeros(CV_NUM)
    a, b, c, d = [],[],[],[]
    for seed in [0,18,318,1996,2021,200008]:
        acc_soft_new = []
        acc_soft_old = []
        acc_hard_new = []
        acc_hard_old = []
        for i in range(CV_NUM):
            network = OneShot(N_TYPES,seed,20,logger)
            # a = time()
            _,_,_,_, testX, testY, full_X, full_Y = dataset.getitem(i)
            shape = full_X.shape[1]
            # print(f"Data Shape:{full_X.shape}\t Classes:{N_TYPES}")
            logging.info(f"Data Shape:{full_X.shape}\t Classes:{N_TYPES}")
            # print(f"Loading Time:{time()-a:.2f}")
            acc = network.loop(full_X, full_Y, testX, testY)

            acc_hard_new.append(acc[0])
            acc_soft_new.append(acc[1])
            acc_soft_old.append(acc[2])
            acc_hard_old.append(acc[3])
            logger.info(f"Soft Voting RAW Space Accuracy for FOLD{i}: {acc[1]*100}%")
            logger.info(f"Hard Voting RAW Space Accuracy for FOLD{i}: {acc[0]*100}%")
            logger.info(f"Soft Voting HID Space Accuracy for FOLD{i}: {acc[2]*100}%")
            logger.info(f"Hard Voting HID Space Accuracy for FOLD{i}: {acc[3]*100}%")

        # print(f"Data Shape:{full_X.shape}\t Classes:{N_TYPES}")
        # print(f"Final soft raw:{np.array(acc_soft_old).mean():.4f}")
        # print(f"Final hard raw:{np.array(acc_hard_old).mean():.4f}")
        # print(f"Final soft new:{np.array(acc_soft_new).mean():.4f}")
        # print(f"Final hard new:{np.array(acc_hard_new).mean():.4f}")
        logger.info(f"Soft raw for seed-{seed}:{np.array(acc_soft_old).mean():.4f}")
        logger.info(f"Hard raw for seed-{seed}:{np.array(acc_hard_old).mean():.4f}")
        logger.info(f"Soft new for seed-{seed}:{np.array(acc_soft_new).mean():.4f}")
        logger.info(f"Hard new for seed-{seed}:{np.array(acc_hard_new).mean():.4f}")
        # print(f"Avg acc:{result.mean():.3f}\tStd {result.std():.3f}")
        # print(f"Avg acc:{result.mean():.3f}\tStd {result.std():.3f}")
        # print(f"Avg acc:{result.mean():.3f}\tStd {result.std():.3f}")
        a.append(np.array(acc_soft_old).mean())
        b.append(np.array(acc_hard_old).mean())
        c.append(np.array(acc_soft_new).mean())
        d.append(np.array(acc_hard_new).mean())
    logger.info(f"Data Shape:{full_X.shape}\t Classes:{N_TYPES}")
    logger.info(f"Final soft raw :{np.array(a).mean():.4f}, Details:{np.array(a)}")
    logger.info(f"Final hard raw :{np.array(b).mean():.4f}, Details:{np.array(b)}")
    logger.info(f"Final soft new :{np.array(c).mean():.4f}, Details:{np.array(c)}")
    logger.info(f"Final hard new :{np.array(d).mean():.4f}, Details:{np.array(d)}")