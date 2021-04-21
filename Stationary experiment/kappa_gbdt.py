#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 20:58:00 2020

@author: kunwang
"""

from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from scipy.io import arff
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import time
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score, roc_auc_score, roc_curve
import os
from skmultiflow.drift_detection.adwin import ADWIN
from sklearn.ensemble import GradientBoostingRegressor
import arff
from tqdm import tqdm
import pixiedust
from copy import deepcopy
from operator import itemgetter
import warnings

warnings.filterwarnings('ignore')


class Kappa_GBDT(object):
    def __init__(self,
                 max_iter=50,
                 sample_rate=0.8,
                 learn_rate=0.01,
                 max_depth=4,
                 new_tree_max_iter=10):

        self.max_iter = max_iter
        self.sample_rate = sample_rate 
        self.learn_rate = learn_rate
        self.max_depth = max_depth 
        self.dtrees = []
        self.original_f = None
        self.new_tree_max_iter = new_tree_max_iter



    def fit(self, x_train, y_train):
        np.random.seed(0)
        n, m = x_train.shape        
        f = np.ones(n) * np.mean(y_train)
        self.original_f = np.mean(y_train)
        self.residual_mean = np.zeros(self.max_iter)
        n_sample = int(n * self.sample_rate)

        for iter_ in range(self.max_iter): 
            sample_idx = np.random.permutation(n)[:n_sample]
            x_train_subset, y_train_subset = x_train[sample_idx, :], y_train[sample_idx]
            y_predict_subset = np.zeros(n_sample)
            
            for j in range(n_sample):
                k = sample_idx[j]
                y_predict_subset[j] = f[k]

            residual = y_train_subset - y_predict_subset

            dtree = DecisionTreeRegressor(max_depth=self.max_depth)
            # fit to negative gradient
            dtree.fit(x_train_subset, residual * self.learn_rate)
            # append new tree
            self.dtrees.append(dtree)  

            # update prediction score
            for j in range(n):
                pre = dtree.predict(np.array([x_train[j]]))
                f[j] += pre
                


    def predict(self, x):
        n = x.shape[0]
        y = np.zeros([n, len(self.dtrees)])
        
        for iter_ in range(len(self.dtrees)):
            dtree = self.dtrees[iter_]
            y[:, iter_] = dtree.predict(x)

        init_residual = np.ones(y.shape[0]) * self.original_f
        self.cumulated_pred_score = np.cumsum(y, axis=1)
        return np.sum(y, axis=1) + init_residual.reshape(1, -1)
    
    
    def kappa_pruning (self,y_test,n):
        
        time_start=time.time()
        
        init_residual = np.ones(y_test.shape[0]) * self.original_f 
        init_residual = init_residual.reshape(1, -1)
        init_score = np.repeat(init_residual, len(self.dtrees), axis=0).T
        tree_score = init_score + self.cumulated_pred_score
        
        # establish the label
        tree_label = (tree_score >= 0.5)
        
        # calculate the kappa score of pair of trees
        k = 0
        k_pair = []
        for a in range(tree_label.shape[1]):
            for b in range(a+1, tree_label.shape[1]):#列数
                tree_label_a = tree_label[:,a]
                tree_label_b = tree_label[:,b]
                k = metrics.cohen_kappa_score(tree_label_a, tree_label_b)
                k_pair.append([k,a,b])
                
        # rank the k_pair by sort the small kappa score
        k_pair.sort()
        k_pair = pd.DataFrame(k_pair,columns=['k','a','b'])

        copy_dtrees = self.dtrees.copy()
        l = len(self.dtrees)
        for i in range (0,l-n-1,2):
            v1=k_pair.iloc[0,1]
            v2=k_pair.iloc[0,2]
            self.dtrees[i] = copy_dtrees[int(v1)]
            self.dtrees[i+1] = copy_dtrees[int(v2)]
            k_pair = k_pair[~k_pair['a'].isin([v1])]
            k_pair = k_pair[~k_pair['a'].isin([v2])]
            k_pair = k_pair[~k_pair['b'].isin([v1])]
            k_pair = k_pair[~k_pair['b'].isin([v2])]
        
        self.dtrees = self.dtrees[:l-n]
        
        time_end=time.time()
        print('time cost',time_end-time_start,'s')
    
    
    
    def incremental_fit(self, x_test, y_test, pred_score, new_tree_max_iter):
        n, m = x_test.shape        
        f = pred_score      
        n_sample = int(n*self.sample_rate)
        np.random.seed(0)
        for iter_ in range(new_tree_max_iter):            
            sample_idx = np.random.permutation(n)[:n_sample]            
            y_residual = y_test - f
            x_train_subset, residual_train_subset = x_test[sample_idx, :], y_residual[sample_idx]
            
            new_tree = DecisionTreeRegressor(max_depth = self.max_depth)
            new_tree.fit(x_train_subset, residual_train_subset * self.learn_rate)
            self.dtrees.append(new_tree)
            self.max_iter += 1
            
            for j in range(n):
                pre = new_tree.predict(np.array([x_test[j]]))
                f[j] += pre