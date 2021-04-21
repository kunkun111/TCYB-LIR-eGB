#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 16:33:49 2021

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
from math import exp,log
from sklearn.preprocessing import  OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from copy import deepcopy
from scipy import stats



# load dataset
def load_arff(path, dataset_name, num_copy):
    if num_copy == -1:
        file_path = path + dataset_name + '/'+ dataset_name + '.arff'
        dataset = arff.load(open(file_path), encode_nominal=True)
    else:
        file_path = path + dataset_name + '/'+ dataset_name + str(num_copy) + '.arff'
        dataset = arff.load(open(file_path), encode_nominal=True)
    return np.array(dataset["data"])


# label to one hot
def label_to_one_hot(y, num_label):    
    """Convert class labels from scalars to one-hot vectors."""    
    num_labels = y.shape[0]   
    index_offset = np.arange(num_labels) * num_label    
    labels_one_hot = np.zeros((num_labels, num_label))    
    labels_one_hot.flat[[index_offset+y.ravel()]] = 1    
    return labels_one_hot



# GBDT main class
class multi_GBDT(object):
    def __init__(self, max_iter=100, sample_rate=0.8, learn_rate=0.1, max_depth=4, new_tree_max_iter=23, num_label=7):
        
        self.max_iter = max_iter # number of trees
        self.sample_rate = sample_rate # 0 < sample_rate <= 1
        self.learn_rate = learn_rate # learn rate
        self.max_depth = max_depth 
        self.original_f = None
        self.new_tree_max_iter = new_tree_max_iter
        self.num_label = num_label
        self.trees = dict()
        self.f = dict()
        #self.new_trees = dict()
 
       
    def initial_f(self,f, y): # prior probability
        y = label_to_one_hot(y,self.num_label)
        n,m = y.shape
        for label in range(y.shape[1]): # label数量
            f[label] = dict()
            for id in range(y.shape[0]): # instance数量
                f[label][id] = 0
        return f
        
    
    def compute_residual(self, y, f): # computer the residual
        residual = {}
        subset = label_to_one_hot(y, self.num_label)
        n = subset.shape[0]
        for label in range(subset.shape[1]): # number of labels
            if subset.shape[0] > 100: # chunk size
                n=100
                
            residual[label] = {}
            p_sum = sum([exp(f[label][x]) for x in range(n)])
            for id in range(n): # instance数量
                p = exp(f[label][id])/p_sum # calculate probability
                residual[label][id] = subset.T[label,id] - p
        return residual
    
    
    def compute_loss(self, y, f):
        loss = 0
        subset = label_to_one_hot(y, self.num_label)
        n = subset.shape[0]
        loss_array = np.zeros(n)
        for id in range(n):
            exp_values = {}
            for label in f: 
                exp_values[label] = exp(f[label][id])
            
            probs = {}
            loss_instance = 0
            for label in f:
                probs[label] = exp_values[label]/sum(exp_values.values())
                loss_instance += subset[id,label]*log(probs[label])
            loss = loss - loss_instance
            loss_array[id] = - loss_instance
            
        return loss/n, loss_array
    
        
    def fit(self, x_train, y_train):
        np.random.seed(0)
        f = dict()
        n, m = x_train.shape
        n_sample = int (n * self.sample_rate)
        self.f = self.initial_f(f,y_train) # probability
        # self.init_residual = self.initial_f(f,y_train)
        loss = []
    
        for iter in range(self.max_iter):       
            sample_idx = np.random.permutation(n)[:n_sample]
            x_train_subset, y_train_subset = x_train[sample_idx, :], y_train[sample_idx]
            
            self.trees[iter] = dict()
            residual = self.compute_residual(y_train_subset, self.f) 
            for label in range(self.num_label):        
                tree = DecisionTreeRegressor(max_depth=4,random_state=0)           
                residual_array = list(residual[label].values())
                residual_array = np.array(residual_array)
                tree.fit(x_train_subset, self.learn_rate * residual_array)
                self.trees[iter][label] = tree
            
                for id in range(n): # update f_value
                    pre_tree = tree.predict([x_train[id]])
                    self.f[label][id] = self.f[label][id] + pre_tree
            
            train_loss, train_loss_array = self.compute_loss(y_train,self.f)
            loss.append(train_loss)

                  
    def predict(self, x_test, y_test):
        n = x_test.shape[0]
        y_pre = np.zeros([n,self.num_label])
        f = dict()
        f_pre = self.initial_f(f, y_test)
        y_cum = []
        loss_array = np.zeros([n,len(self.trees)])
        tree_pred_array = np.zeros([n,self.num_label,len(self.trees)])

        for iter in range(len(self.trees)):
            
            for label in range(self.num_label):                       
                tree = self.trees[iter][label]
                y_pre[:,label] = y_pre[:, label] + tree.predict(x_test)
                
                for id in range(n): # update f _value
                    pre_tree = tree.predict([x_test[id]])
                    f_pre[label][id] = f_pre[label][id] + pre_tree
                    
            test_loss, test_loss_array = self.compute_loss(y_test, f_pre)  
            # print(test_loss, test_loss_array)
            y_cum.append(test_loss)
            loss_array[:,iter] = test_loss_array
            tree_pred_array[:,:,iter] = y_pre

        pre_total = y_pre
    
        return pre_total, y_cum, loss_array, tree_pred_array 
    
    
    def incre_fit(self,x_test,y_test,pred_score,new_tree_max_iter):
        n, m = x_test.shape
        n_sample = int (n * self.sample_rate) 
        
        for iter in range(new_tree_max_iter):
            
            if n > 100:
                n = 100
                
            sample_idx = np.random.permutation(n)[:n_sample]
            y_residual = self.compute_residual(y_test,self.f)
            new_trees = dict()
        
            residual_new = []
            for i in range(self.num_label):
                residual_list = list(y_residual[i].values())
                residual_new.append(residual_list)         
            y_residual = np.array(residual_new).T  
            
            x_train_subset, residual_train_subset = x_test[sample_idx,:], y_residual[sample_idx,:]       
            for label in range(self.num_label):         
                residual_subset = residual_train_subset[:,label] 
                new_tree = DecisionTreeRegressor(max_depth = 4)
                new_tree.fit(x_train_subset, residual_subset * self.learn_rate)
                new_trees[label] = new_tree
                
                for id in range(n): # update f _value
                    pre_tree = new_tree.predict([x_test[id]])
                    self.f[label][id] = self.f[label][id] + pre_tree 
                    
            self.trees.append(new_trees)
        
        
               
    def kappa_pruning(self, tree_pred_array):       
        n = tree_pred_array.shape[0]
        iter = len(self.trees)
        
        # establish tree label
        tree_label = np.zeros([n,iter])
        for i in range(iter):
            tree_label[:,i] = self.pred_label(tree_pred_array[:,:,i])
        
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

        # reorder trees
        copy_trees = self.trees.copy()
        l = len(self.trees)
        a = 0
        for i in range (0,l-1,2):
            if k_pair.iloc[0,0] >=0.81:
                break
            else:
                v1=k_pair.iloc[0,1]
                v2=k_pair.iloc[0,2]
                self.trees[i] = copy_trees[int(v1)]
                self.trees[i+1] = copy_trees[int(v2)]
                k_pair = k_pair[~k_pair['a'].isin([v1])]
                k_pair = k_pair[~k_pair['a'].isin([v2])]
                k_pair = k_pair[~k_pair['b'].isin([v1])]
                k_pair = k_pair[~k_pair['b'].isin([v2])]
                a += 2
        
        print('keep',a)
        #prune trees
        tree_prune_list = range(a,l)
        self.trees = [self.trees[i] for i in range(0, len(self.trees), 1) if i not in tree_prune_list] 
        print ('len model',len(self.trees))
        
               
    def pred_label(self, pre_prob):
        n, m = pre_prob.shape
        exp_values = dict()
        predict_label = None
        final_label = np.zeros(n)
        
        # get probability
        for id in range(n):
            for label in range(m):
                exp_values[label] = exp(pre_prob[id,label])
            exp_sum = sum(exp_values.values())
            probs = dict()
            
            for label in exp_values:
                probs[label] = exp_values[label] / exp_sum
    
	    # establish label
            for label in probs:
                if predict_label == None or probs[label]>probs[predict_label]:
                    predict_label = label
            final_label[id] = predict_label
            
        return final_label
    

 