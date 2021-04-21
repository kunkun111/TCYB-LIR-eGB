#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 21:05:58 2020

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
from scipy import stats
import seaborn as sns



def takeSecond(elem):
    return elem[1]



def new_list(ab,n):
    ab.sort(key = takeSecond, reverse=False)
    # print(ab)
    ab_list= []
    for i in range (n):
        ab_list.append(ab[i][0])
    # print(ab_list)
    return ab_list




class Portion_Stat_GBDT(object):
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
        #print(self.original_f)
        self.residual_mean = np.zeros(self.max_iter)
        n_sample = int(n * self.sample_rate)
        f_array = np.zeros([n,self.max_iter])

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
            #print(f,f.shape)
            f_array[:,iter_] = f
            
        return f_array



    def predict(self, x):
        n = x.shape[0]
        y = np.zeros([n, len(self.dtrees)])
        
        for iter_ in range(len(self.dtrees)):
            dtree = self.dtrees[iter_]
            y[:, iter_] = dtree.predict(x)

        init_residual = np.ones(y.shape[0]) * self.original_f
        self.cumulated_pred_score = np.cumsum(y, axis=1)
        return np.sum(y, axis=1) + init_residual.reshape(1, -1)
    
    
    
     def portion_stat_pruning(self, y_test, n):#stationary      
        time_start=time.time()
        init_residual = np.ones(y_test.shape[0]) * self.original_f
        residual = y_test.reshape(1, -1) - init_residual.reshape(1, -1)
        residual_mat = np.repeat(residual, len(self.dtrees), axis=0).T
        tree_pruning_residual = np.abs(residual_mat - self.cumulated_pred_score)
             
        tree_portion = []
        prune_tree_list = []
        drift_list = []
        for i in range (tree_pruning_residual.shape[1]-2):
            tree_residual_a = tree_pruning_residual[:,i]
            tree_residual_b = tree_pruning_residual[:,i+1]
            tree_residual_c = tree_pruning_residual[:,i+2]
            portion1 = (tree_residual_b - tree_residual_a)/tree_residual_a
            portion2 = (tree_residual_c - tree_residual_b)/tree_residual_b
            
            portion_1 = (np.mean(tree_residual_b) - np.mean(tree_residual_a))/np.mean(tree_residual_a)
            portion_2 = (np.mean(tree_residual_c) - np.mean(tree_residual_b))/np.mean(tree_residual_b)

            t,p_two = stats.ks_2samp(portion1, portion2, alternative='greater',mode='auto')
            tree_portion.append((i+2,p_two))
            
        prune_tree_list = new_list(tree_portion,n)       
        self.dtrees = [self.dtrees[i] for i in range(0, len(self.dtrees), 1) if i not in prune_tree_list]   
        
        time_end=time.time()
        print('time cost',time_end-time_start,'s')
        
        return prune_tree_list
        
        
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