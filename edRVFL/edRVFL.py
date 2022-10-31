from functools import partial
from easydict import EasyDict
import easydict
import numpy as np
from scipy import stats


def majorityVoting(Y,pred_idx):

    Nsample= Y.shape[0]
    Ind_corrClass=np.argmax(Y,axis=1)
    indx=np.zeros(Nsample)
    for i in range (Nsample):
        Y=pred_idx[i,:]
        indx[i]=stats.mode(Y)[0][0] 
        
    acc=np.mean(indx==Ind_corrClass)

    return acc
        

class edRVFL(object):

    def __init__(self, classes, attr_train, attr_fixed, state_dict=EasyDict()):
        super().__init__()
        self.attr = EasyDict(attr_train, **attr_fixed)  
        self.attr.lamb = 2**self.attr.C
        self.attr.RandState = np.random.RandomState(self.attr.randseed)
        self.classes = classes
        self.state_dict = state_dict
        if not self.state_dict:
            self.state_dict['w'] = []
            self.state_dict['b'] = []
            self.state_dict['beta'] = []
            self.state_dict['A_'] = []
            self.state_dict['pred_eval'] = []
            self.state_dict['mean'] = []
            self.state_dict['std'] = []
        
        self.drop_amount = int(np.floor(self.attr.drop * self.attr.N))
        self.selected_amount = int(np.floor(self.attr.select * self.attr.N))
        self.TrainingAccuracy = np.zeros(self.attr.L)

    def state_init(self):
        self.state_dict=EasyDict()
        self.state_dict['w'] = []
        self.state_dict['b'] = []
        self.state_dict['beta'] = []
        self.state_dict['A_'] = []
        self.state_dict['pred_eval'] = []
        self.state_dict['mean'] = []
        self.state_dict['std'] = []

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
    
    def _ensemble(self, target, X):
        accs = []
        for i in range(self.attr.L):
            accuracy = majorityVoting(target, X[:,:i+1])
            accs.append(accuracy)
        return accs

    def train(self, X, target):
        self.state_init()
        attr = self.attr
        self.raw_X = X
        predictions = []
        for i in range(attr.L):

            n_sample, n_D = X.shape
            if i == 0:
                w = 2 * attr.RandState.rand(attr.N, n_D) - 1
                b = attr.RandState.rand(1, attr.N)
            else:
                w = 2 * attr.RandState.rand(attr.N + self.selected_amount - self.drop_amount + n_D, n_D) - 1
                b = attr.RandState.rand(1, attr.N + self.selected_amount - self.drop_amount + n_D)
            
            w = attr.S * w.T /  np.expand_dims(np.linalg.norm(w,axis=0), 1)
            # w = w.T
            self.w = w
            
            self.b = b

            self.state_dict['w'].append(w)
            self.state_dict['b'].append(b)

            A_ = X @ w
            # layer normalization
            A_mean = np.mean(A_, axis=0)
            A_std = np.std(A_, axis=0)

            self.mean = A_mean
            self.std = A_std

            self.state_dict['mean'].append(self.mean)
            self.state_dict['std'].append(self.std)

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


            if i == 0:
                A_merge = np.concatenate([self.raw_X, A_, np.ones((n_sample, 1))], axis=1)
            else:
                A_merge = np.concatenate([self.raw_X, A_selected, A_, np.ones((n_sample, 1))], axis=1)

            beta_ = self.matrix_inverse(A_merge, target)
            self.beta = beta_
            self.state_dict['beta'].append(beta_)

            if attr.fs == 0:

                significance = np.linalg.norm(beta_, ord=1, axis=1)
                ranked_index = np.argsort(significance[n_D:-1])[::-1]

            elif attr.fs == 1:

                significance = np.linalg.norm(beta_, ord=2, axis=1)
                ranked_index = np.argsort(significance[n_D:-1])[::-1]

            selected_index = ranked_index[:self.selected_amount]
            self.state_dict.s_idx.append(selected_index)
            left_amount = attr.N - self.drop_amount
            left_index = ranked_index[:left_amount]
            self.state_dict.d_idx.append(left_index)
            A_except_trainX = A_merge[:, n_D: -1]
            A_selected = A_except_trainX[:, selected_index]
            A_ = A_except_trainX[:,left_index]

            X = np.concatenate([self.raw_X, A_selected, A_], axis=1)


            predict_score = A_merge @ beta_

            prediction = np.argmax(predict_score,-1).ravel()
            predictions.append(prediction)
        pred = np.array(predictions)
        train_acc = np.array(self._ensemble(target, pred.T))
        return train_acc
    
    def evaluate(self, X, target):
        self.raw_Xe = X
        predictions = []
        attr = self.attr
        for i in range(attr.L):
            n_sample, n_D = X.shape

            A_ = X @ self.state_dict.w[i]
            A_ = (A_ - self.state_dict.mean[i]) / self.state_dict.std[i]
            A_ = A_ + np.repeat(self.state_dict.b[i], n_sample, 0)
            # A_ = attr.gama * A_ + attr.alpha
            if attr.activation == 0:

                A_ = self.relu(A_)

            elif attr.activation == 1:

                A_ = self.selu(A_)

            elif attr.activation == 2:

                A_ = self.sigmoid(A_)
            
            elif attr.activation == 3:

                A_ = 1.6732632423543772848170429916717 * A_ * self.sigmoid(A_)
            

            if i == 0:
                A_merge = np.concatenate([self.raw_Xe, A_, np.ones((n_sample, 1))], axis=1)
            else:
                A_merge = np.concatenate([self.raw_Xe, A_select, A_, np.ones((n_sample, 1))], axis=1)

            A_except_testX = A_merge[:, n_D: -1]
            A_ = A_except_testX[:,self.state_dict.d_idx[i]]
            A_select = A_except_testX[:, self.state_dict.s_idx[i]]
            X = np.concatenate([self.raw_Xe, A_select, A_], axis=1)
            predict_score = A_merge @ self.state_dict.beta[i]

            prediction = np.argmax(predict_score,-1).ravel()
            predictions.append(prediction)

        pred = np.array(predictions)
        test_acc = np.array(self._ensemble(target, pred.T))

        return test_acc
    
    def rvfl(self, x, target, xe, targete, xt=None, targett=None): 
        acc_train = self.train(x, target)
        acc_eval = self.evaluate(xe, targete)
        if xt is not None and targett is not None:
            acc_test = self.evaluate(xt, targett)
        print(f"Train:{acc_train.max():.5f}\tEval :{acc_eval.max():.5f}")
        if xt is not None and targett is not None:
            print(f"Test :{acc_test.max():.5f}")
            return acc_train, acc_eval, acc_test
        return acc_train, acc_eval

