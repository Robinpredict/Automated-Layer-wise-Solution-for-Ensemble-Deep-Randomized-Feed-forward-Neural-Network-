#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on: 2021/07/23 20:56
@Author: Merc2
'''
from cgi import test
from functools import partial
from easydict import EasyDict
import numpy as np
from sklearn.utils.extmath import softmax
from scipy import stats
from bohb import BOHB
import bohb.configspace as cs
from .utils import one_hot, cross_entropy, cal_weight, score_based_split_nodrop, sigmoid, relu, selu, get_top_n_layer, distance, weight_based_split, score_based_split
from sklearn.model_selection import train_test_split as Split
from einops import rearrange
import time
import logging
class RVFL_layer(object):
    def __init__(self, classes, attr, raw_input, raw_eval):
        super().__init__()
        self.attr = attr
        self.attr.lamb = 2**self.attr.C
        self.attr.RandState = np.random.RandomState(self.attr.randseed)
        self.classes = classes
        self.raw_X = raw_input
        self.raw_X_eval = raw_eval
        self.Params = EasyDict()

    def selu(self, x):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * np.where(x >= 0, x, alpha * (np.exp(x) - 1))
    
    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    
    def matrix_inverse(self, x, target):
        n_sample, n_D = x.shape
        
        if self.attr.C == 0:
            beta = np.dot(np.linalg.pinv(x), target)

        elif n_D<n_sample:
            beta = np.matmul(np.matmul(np.linalg.inv(np.identity(x.shape[1])/self.attr.lamb+np.matmul(x.T,x)),x.T),target)
        else:
            beta = np.matmul(x.T,np.matmul(np.linalg.inv(np.identity(x.shape[0])/self.attr.lamb+np.matmul(x,x.T)),target))

        return beta
    
    def train(self, X, target):
        attr = self.attr
        n_sample, n_D = X.shape
        # print(X.shape)
        w = 2 * self.attr.RandState.rand(self.attr.N, n_D) - 1
        b = self.attr.RandState.rand(1, self.attr.N)


        w = attr.S * w.T /  np.expand_dims(np.linalg.norm(w,axis=0), 1)

        self.w = w
            
        self.b = b

        A_ = X @ w
        # layer normalization
        A_mean = np.mean(A_, axis=0)
        A_std = np.std(A_, axis=0)
        self.mean = A_mean
        self.std = A_std

        A_ = (A_ - A_mean) / A_std
        A_ = A_ + np.repeat(b, n_sample, 0)
        # A_ = attr.gama * A_ + attr.alpha

        if attr.activation == 0:

            A_ = self.relu(A_)

        elif attr.activation == 1:

            A_ = self.selu(A_)

        elif attr.activation == 2:

            A_ = self.sigmoid(A_)

        elif attr.activation == 3:

            A_ = 1.6732632423543772848170429916717 * A_ * self.sigmoid(A_)

        A_merge = np.concatenate([self.raw_X, A_, np.ones((n_sample, 1))], axis=1)

        self.A_ = A_
        beta_ = self.matrix_inverse(A_merge, target)
        self.beta = beta_
        X = np.concatenate([self.raw_X,  A_], axis=1)
        predict_score = A_merge @ beta_

        self.Params['w'] = self.w
        self.Params['b'] = self.b
        self.Params['beta'] = self.beta
        self.Params['mean'] = self.mean
        self.Params['std'] = self.std

        return predict_score
    
    def eval(self, X, params=None):
        attr = self.attr
        n_sample, n_D = self.raw_X_eval.shape

        if params is not None:
            self.Params = params
        
        A_ = X @ self.Params.w
        A_ = (A_ - self.Params.mean) / self.Params.std
        A_ = A_ + np.repeat(self.Params.b, n_sample, 0)
        if attr.activation == 0:

            A_ = self.relu(A_)

        elif attr.activation == 1:

            A_ = self.selu(A_)

        elif attr.activation == 2:

            A_ = self.sigmoid(A_)
        
        elif attr.activation == 3:

            A_ = 1.6732632423543772848170429916717 * A_ * self.sigmoid(A_)
        
        self.A_t = A_

        A_merge = np.concatenate([self.raw_X_eval, A_, np.ones((n_sample, 1))], axis=1)
        predict_score = A_merge @ self.Params.beta
        return predict_score
    
    def rvfl(self, previous_X, Y, previous_Xt, Yt):
        # train_score = softmax(self.train(previous_X, Y))
        # eval_score = softmax(self.eval(previous_Xt))
        train_score = self.train(previous_X, Y)
        eval_score = self.eval(previous_Xt)
        train_acc = np.mean(np.argmax(train_score,-1).ravel()==np.argmax(Y,axis=1))
        eval_acc = np.mean(np.argmax(eval_score,-1).ravel()==np.argmax(Yt,axis=1))
        next_X = self.A_
        next_Xt = self.A_t
        # print(train_acc, eval_acc)
        return [train_acc, eval_acc], [next_X, next_Xt], [train_score, eval_score]


class Greedy(object):
    def __init__(self, classes, randseed, max_L):
        self.randseed = randseed
        self.state_init()
        self.recored_init()
        self.classes = classes
        self.max_L = max_L

    def recored_init(self):
        record = EasyDict()
        record.eval = [0.]
        record.train = [0.]
        self.record = record


    def state_init(self):
        self.L = 0
        self.state_dict=EasyDict()
        self.state_dict['Configs'] = []
        self.scores = EasyDict()
        self.scores.t = []
        self.scores.e = []


    def RVFL_fun(self, attr, trainx, trainy, evalx, evaly):
        attr = EasyDict(attr)
        attr.L = self.L
        attr.randseed = self.randseed
        layer = RVFL_layer(classes=self.classes, attr=attr, raw_input=self.rawX, raw_eval=self.rawXe)
        acc,next_in,score = layer.rvfl(trainx, trainy, evalx,evaly)
        if acc[1] > max(self.record.eval):
            self.X = next_in[0]
            self.Xe = next_in[1]
            self.score_t = score[0]
            self.score_e = score[1]
        elif acc[1] == max(self.record.eval):
            if acc[0] > max(self.record.train):
                self.X = next_in[0]
                self.Xe = next_in[1]
                self.score_t = score[0]
                self.score_e = score[1]
            else:
                self.X = next_in[0]
                self.Xe = next_in[1]
                self.score_t = score[0]
                self.score_e = score[1]
        else:
            pass
        self.record.eval.append(acc[1])
        self.record.train.append(acc[0])

        return acc

    def train_layer_by_layer(self, X, target, Xe, targete):
        self.X = X
        self.rawX = X.copy()
        self.Xe = Xe
        self.rawXe = Xe.copy()
        for i in range(self.max_L):
            print(f"Layer:{i+1}")
            self.recored_init()
            N = cs.IntegerUniformHyperparameter('N', 100, 1024)
            C = cs.UniformHyperparameter('C', -12, 12)
            S = cs.NormalHyperparameter('S', 0, 5)
            activation = cs.CategoricalHyperparameter('activation', [0,1,2,3])
            configspace = cs.ConfigurationSpace([N, C, S, activation], seed=1)
            opt = BOHB(configspace, 
                    partial(self.RVFL_fun,
                    trainx=self.X, trainy=target,
                    evalx=self.Xe, evaly=targete), 
                    max_budget=20, min_budget=1)
            logs = opt.optimize()
            best_configs = logs.best['hyperparameter'].to_dict()
            self.state_dict.Configs.append(best_configs)
            self.L += 1
            print(logs.best)
            self.scores.t.append(np.argmax(self.score_t,-1).ravel())
            self.scores.e.append(np.argmax(self.score_e,-1).ravel())
        self.scores.t = np.array(self.scores.t).T
        self.scores.e = np.array(self.scores.e).T
        accs_t = []
        accs_e = []
        for i in range(self.max_L):
            accuracy_t = majorityVoting(target, self.scores.t[:,:i+1])
            accuracy_e = majorityVoting(targete, self.scores.e[:,:i+1])
            accs_e.append(accuracy_e)
            accs_t.append(accuracy_t)
        # print(accs_t, accs_e)

    def retrain_for_test(self, X, target, Xe, targete):
        X_ = X
        Xe_ = Xe
        results_t = []
        results_e = []
        for i in range(self.max_L):
            config = EasyDict(self.state_dict.Configs[i])
            config.L = self.L
            config.randseed = self.randseed
            net = RVFL_layer(classes=self.classes, attr=config, raw_input=X, raw_eval=Xe)
            _ , [next_X, next_Xe], [train_score, eval_score] = net.rvfl(X_, target, Xe_, targete)
            X_ = next_X
            Xe_ = next_Xe
            results_e.append(np.argmax(eval_score,-1).ravel())
            results_t.append(np.argmax(train_score,-1).ravel())
        results_e = np.array(results_e).T
        results_t = np.array(results_t).T
        accs_t = []
        accs_e = []
        for i in range(self.max_L):
            accuracy_t = majorityVoting(target, results_t[:,:i+1])
            accuracy_e = majorityVoting(targete, results_e[:,:i+1])
            accs_e.append(accuracy_e)
            accs_t.append(accuracy_t)
        # print(accs_t, accs_e)
        return accs_e



class OneShot(object):
    def __init__(self, classes, randseed, max_L,logger):
        self.randseed = randseed
        self.state_init()
        self.recored_init()
        self.classes = classes
        self.max_L = max_L
        self.logger = logger

    def recored_init(self):
        record = EasyDict()
        record.eval = [0.]
        record.train = [0.]
        self.record = record

    def state_init(self):
        self.L = 0
        self.state_dict=EasyDict()
        self.state_dict['Configs'] = []
        self.state_dict['Params'] = []
        self.scores = EasyDict()
        self.scores.t = []
        self.scores.e = []
        self.wweight = 0
        self.centers = [[0]]

    def RVFL_fun(self, attr, trainx, trainy, evalx, evaly, weight=None):
        attr = EasyDict(attr)
        attr.L = self.L
        attr.randseed = self.randseed
        layer = RVFL_layer(classes=self.classes, attr=attr, raw_input=self.rawX, raw_eval=self.rawXe)
        acc,next_in, score = layer.rvfl(trainx, trainy, evalx,evaly)
        self.X = next_in[0]
        self.Xe = next_in[1]
        self.score_t = score[0]
        self.score_e = score[1]
        # if acc[1] > max(self.record.eval):
        #     self.X = next_in[0]
        #     self.Xe = next_in[1]
        #     self.score_t = score[0]
        #     self.score_e = score[1]
        # elif acc[1] == max(self.record.eval):
        #     if acc[0] > max(self.record.train):
        #         self.X = next_in[0]
        #         self.Xe = next_in[1]
        #         self.score_t = score[0]
        #         self.score_e = score[1]
        #     else:
        #         self.X = next_in[0]
        #         self.Xe = next_in[1]
        #         self.score_t = score[0]
        #         self.score_e = score[1]
        # else:
        #     pass
        self.record.eval.append(acc[1])
        self.record.train.append(acc[0])
        loss = cross_entropy(softmax(score[1]),evaly, weight)
        return loss

    def find_optim(self, full_X, full_y, next_X=None, weight=None, pre_loss=None):
        self.fullx = full_X
        self.fully = full_y
        if next_X is None:
            trainX, evalX, trainY, evalY = Split(full_X, full_y)
            hidden_train_X = trainX
            hidden_eval_X = evalX
            self.rawX = trainX
            self.rawXe = evalX
            weight_eval = None
        else:
            # # trainX, evalX, hidden_train_X, hidden_eval_X, weight_train, weight_eval, trainY, evalY = Split(full_X, next_X, weight, full_y)
            # trainX, evalX, hidden_train_X, hidden_eval_X, weight_train, weight_eval, trainY, evalY = weight_based_split(full_X, next_X, full_y, weight)
            # # tx = weight_based_split(full_X, weight)
            # self.rawX = trainX*weight_train[...,np.newaxis]
            # # TODO SELECT WEIGHTS
            # self.rawXe = evalX
            trainX, evalX, hidden_train_X, hidden_eval_X, trainY, evalY, centers = score_based_split_nodrop(full_X, next_X, full_y, pre_loss, weight)
            self.rawX = trainX
            self.rawXe = evalX
            self.centers.append(centers)
        self.logger.info(f"Layer:{self.L+1}")
        self.recored_init()
        N = cs.IntegerUniformHyperparameter('N', 256, 2048)
        C = cs.UniformHyperparameter('C', -12, 12)
        S = cs.NormalHyperparameter('S', 0, 5) 
        activation = cs.CategoricalHyperparameter('activation', [0,1,2,3])
        configspace = cs.ConfigurationSpace([N, C, S, activation], seed=1)
        opt = BOHB(configspace, 
                partial(self.RVFL_fun,
                trainx=hidden_train_X, trainy= trainY,
                evalx=hidden_eval_X, evaly=evalY), 
                max_budget=50, min_budget=1)
        logs = opt.optimize()
        best_configs = logs.best['hyperparameter'].to_dict()
        self.state_dict.Configs.append(best_configs)
        self.logger.info(logs.best)
        self.logger.info(logs.best['hyperparameter'])

    def train_optim(self, X, target, Xt, targett):
        X_ = X
        Xe_ = Xt
        # results_t = []
        # results_e = []

        config = EasyDict(self.state_dict.Configs[self.L])
        config.L = self.L
        config.randseed = self.randseed
        net = RVFL_layer(classes=self.classes, attr=config, raw_input=self.fullx, raw_eval=self.Xt)
        _ , [next_X, next_Xt], [train_score, test_score] = net.rvfl(X_, target, Xe_, targett)
        self.state_dict.Params.append(net.Params)

        return next_X, next_Xt, train_score, test_score
    
    def test_loop(self, data, X_test, configs, params):
        attr = EasyDict(configs)
        n_sample, n_D = X_test.shape
        
        A_ = data @ params.w
        A_ = (A_ - params.mean) / params.std
        A_ = A_ + np.repeat(params.b, n_sample, 0)
        if attr.activation == 0:

            A_ = relu(A_)

        elif attr.activation == 1:

            A_ = selu(A_)

        elif attr.activation == 2:

            A_ = sigmoid(A_)
        
        elif attr.activation == 3:

            A_ = 1.6732632423543772848170429916717 * A_ * sigmoid(A_)
        
        A_t = A_

        A_merge = np.concatenate([X_test, A_, np.ones((n_sample, 1))], axis=1)
        predict_score = A_merge @ params.beta
        return predict_score, A_t        

    def final_test(self, trainX, trainY, testX, testy, desired_index):
        scores = []
        ensemble_idc = []
        decision = 0
        for l in range(self.max_L):
            if l == 0:
                testx = testX
                trainx = trainX
            score, nextx = self.test_loop(testx, testX, self.state_dict.Configs[l], self.state_dict.Params[l])
            _,train_temp = self.test_loop(trainx, trainX, self.state_dict.Configs[l], self.state_dict.Params[l])
            testx, trainx = nextx, train_temp
            test_cat = np.concatenate((testX, nextx),1)
            train_cat = np.concatenate((trainX, train_temp),1)
            dists = distance(test_cat, train_cat)
            choose_train_idx= dists.argsort(1)[:,:3]
            ensemble_idx = desired_index[:,choose_train_idx]
            ensemble_idx = rearrange(ensemble_idx,'c b n -> b (c n)')
            ensemble_idc.append(ensemble_idx)
            scores.append(score)
        scores = np.stack(scores)
        ensemble_idc = np.concatenate(ensemble_idc,1)
        scores = rearrange(scores, 'c b d -> b c d')
        if decision == 0:
            dists = distance(testX, trainX)
            choose_train_idx= dists.argsort(1)[:,:3]
            ensemble_idx = desired_index[:,choose_train_idx]
            ensemble_idx = rearrange(ensemble_idx,'c b n -> b (c n)')
            final_score_raw = np.array(list(map(lambda a,b: a[b], scores,ensemble_idx)))
            pred_soft_raw = final_score_raw.sum(1).argmax(1)
            pred_hard_raw = stats.mode(final_score_raw.argmax(2),1)[0]
            self.logger.info(f'Soft OS Result RAW Space:{(pred_soft_raw==testy.argmax(1)).sum() / len(testy):.4f}({(pred_soft_raw==testy.argmax(1)).sum()}/{len(testy)})')
            self.logger.info(f'Hard OS Result RAW Space:{(pred_hard_raw.squeeze()==testy.argmax(1)).sum() / len(testy):.4f}({(pred_hard_raw.squeeze()==testy.argmax(1)).sum()}/{len(testy)})')

        final_score = np.array(list(map(lambda a,b: a[b], scores,ensemble_idc)))
        pred_soft = final_score.sum(1).argmax(1)
        pred_hard = stats.mode(final_score.argmax(2),1)[0].squeeze()
        self.logger.info(f'Soft OS Result New Space:{(pred_soft==testy.argmax(1)).sum() / len(testy):.4f}({(pred_soft==testy.argmax(1)).sum()}/{len(testy)})')
        self.logger.info(f'Hard OS Result New Space:{(pred_hard==testy.argmax(1)).sum() / len(testy):.4f}({(pred_hard==testy.argmax(1)).sum()}/{len(testy)})')
        accs =[]
        for n in range(self.max_L):
            predict = scores[:,n].argmax(1)
            acc = (predict==testy.argmax(1)).sum() / len(testy)
            accs.append(acc)
            self.logger.info(f'Total Result:{np.array(accs).mean()}')
        return (pred_hard.squeeze()==testy.argmax(1)).sum() / len(testy), (pred_soft==testy.argmax(1)).sum() / len(testy), \
                (pred_soft_raw==testy.argmax(1)).sum() / len(testy), (pred_hard_raw.squeeze()==testy.argmax(1)).sum() / len(testy)
        
    def loop(self,full_X, full_y,Xt,yt):
        self.Xt = Xt
        losses = np.zeros((self.max_L,)+full_y.shape)
        for l in range(self.max_L):
            ti = time.time()
            self.L = l
            if l == 0:
                self.find_optim(full_X, full_y)
            else:
                self.find_optim(full_X, full_y, nextX, weight_for_next, pre_loss = train_score)
            if l == 0:
                nextX, nextXt, train_score, test_score = self.train_optim(full_X,full_y,Xt,yt)
            else:
                nextX, nextXt, train_score, test_score = self.train_optim(nextX,full_y,nextXt,yt)
            losses[l] = softmax(train_score)
            weight = cal_weight(train_score, full_y)
            self.wweight += weight
            # weight_for_next = (self.wweight - np.min(self.wweight))/np.ptp(self.wweight)
            weight_for_next = self.wweight / (l+1)
            self.logger.info(f'one layer take {(time.time()-ti)/60.}min')
            self.logger.info(f'Layer{l+1} Training ACC: {((train_score.argmax(-1) == full_y.argmax(-1)).sum())/train_score.shape[0]}')
        desired_index = get_top_n_layer(losses ,full_y, n=3, k = self.max_L)
        acc = self.final_test(full_X, full_y, Xt, yt, desired_index)
        return acc
            # acc=(test_score.argmax(1) == yt.argmax(1)).sum() / test_score.shape[0]
            # print(f"Test Acc for each layer: {acc}")
        



    

