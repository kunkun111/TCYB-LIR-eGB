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
        
        self.max_iter = max_iter#树的最大数量
        self.sample_rate = sample_rate
        self.learn_rate = learn_rate#学习率
        self.max_depth = max_depth#树的最大深度
        self.original_f = None
        self.new_tree_max_iter = new_tree_max_iter
        self.num_label = num_label
        self.trees = dict()
        self.f = dict()
        #self.new_trees = dict()
 
       
 
    def initial_f(self,f, y):#prior probability
        y = label_to_one_hot(y,self.num_label)
        n,m = y.shape
        for label in range(y.shape[1]):#label
            f[label] = dict()
            for id in range(y.shape[0]):
                #instance
                f[label][id] = 0
        return f
        
    
        
    def compute_residual(self, y, f):# computer the residual
        residual = {}
        subset = label_to_one_hot(y, self.num_label)
        #print('shape',subset.shape)
        #print('f',f)
        
        n = subset.shape[0]       
        for label in range(subset.shape[1]):#label
            
            if subset.shape[0] > 100:
                n=100
                
            residual[label] = {}
            p_sum = sum([exp(f[label][x]) for x in range(n)])
            #print('p_sum',p_sum)
            for id in range(n):#instance
                p = exp(f[label][id])/p_sum#calculate probability
                residual[label][id] = subset.T[label,id] - p
        return residual
    
    
    
    def compute_loss(self, y, f):
        loss = 0.0
        subset = label_to_one_hot(y, self.num_label)
        #print(subset)
        n = subset.shape[0]
        #print('n',n) 
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
        self.f = self.initial_f(f,y_train)#probability
        #print(self.f)
        self.init_residual = self.initial_f(f,y_train)
        #print(self.init_residual)
        loss = []
    
        for iter in range(self.max_iter):       
            sample_idx = np.random.permutation(n)[:n_sample]
            x_train_subset, y_train_subset = x_train[sample_idx, :], y_train[sample_idx]
            
            self.trees[iter] = dict()
            residual = self.compute_residual(y_train_subset, self.f) 
            #print('residual',residual)
            for label in range(self.num_label):        
                tree = DecisionTreeRegressor(max_depth=4,random_state=0)           
                residual_array = list(residual[label].values())
                residual_array = np.array(residual_array)
                tree.fit(x_train_subset, self.learn_rate * residual_array)
                self.trees[iter][label] = tree
            
                for id in range(n):#update f _value
                    pre_tree = tree.predict([x_train[id]])
                    self.f[label][id] = self.f[label][id] + pre_tree
            
            train_loss, train_loss_array = self.compute_loss(y_train,self.f)
            #print(train_loss)
            loss.append(train_loss)

        
                  
    def predict(self, x_test, y_test):
        n = x_test.shape[0]
        y_pre = np.zeros([n,self.num_label])
        f = dict()
        f_pre = self.initial_f(f, y_test)
        y_cum = []
        loss_array = np.zeros([n,len(self.trees)])

        for iter in range(len(self.trees)):
            
            for label in range(self.num_label):                       
                tree = self.trees[iter][label]
                y_pre[:,label] = y_pre[:, label] + tree.predict(x_test)
                
                for id in range(n):#update f _value
                    pre_tree = tree.predict([x_test[id]])
                    f_pre[label][id] = f_pre[label][id] + pre_tree

            test_loss, test_loss_array = self.compute_loss(y_test, f_pre)            

            y_cum.append(test_loss)
            loss_array[:,iter] = test_loss_array

        pre_total = y_pre
    
        return pre_total, y_cum, loss_array
    
    
    
    def incre_fit(self,x_test,y_test,pred_score,new_tree_max_iter):
    
        n, m = x_test.shape
        n_sample = int (n * self.sample_rate) 
    
        for iter in range(new_tree_max_iter):
            
            if n > 100:
                n = 100
                    
            sample_idx = np.random.permutation(n)[:n_sample]
            y_residual = self.compute_residual(y_test, self.f)
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
                
                for id in range(n): #update f _value
                    pre_tree = new_tree.predict([x_test[id]])
                    self.f[label][id] = self.f[label][id] + pre_tree
                    
            #print('len trees',len(self.trees))
            
            self.trees.append(new_trees)
        
        

    def portion_pruning(self, y_test_loss):

        iter = len(y_test_loss)
        tree_portion = []
        prune_tree_list = []
        
        for i in range(iter-1):
            loss_a = y_test_loss[i]
            loss_b = y_test_loss[i+1]
            ratio = (loss_b - loss_a)/loss_a
            tree_portion.append((i,ratio))
            
            if ratio >= 0:
                prune_tree_list.append(i)
            
        self.trees = [self.trees[i] for i in range(0, len(self.trees), 1) if i not in prune_tree_list]   
        
    

    def portion_stat_pruning (self, y_loss_array):

        iter = len(self.trees)
        tree_portion = []
        prune_tree_list = []
        
        for i in range (iter-2):
            tree1 = y_loss_array[:,i]
            tree2 = y_loss_array[:,i+1]
            tree3 = y_loss_array[:,i+2]
            
            portion1 = (tree2 - tree1) / tree1
            portion2 = (tree3 - tree2) / tree2
            
            portion_1 = (np.mean(tree2) - np.mean(tree1))/np.mean(tree1)
            portion_2 = (np.mean(tree3) - np.mean(tree2))/np.mean(tree2)
             
            
            t,p_two = stats.ks_2samp(portion1, portion2, alternative='greater', mode='auto')
            tree_portion.append((i+1,p_two))
                
            if portion_2 >= 0 and p_two < 0.001:
            # if portion_2 >= 0 and p_two < 0.005:
            # if portion_2 >= 0 and p_two < 0.01:
            # if portion_2 >= 0 and p_two < 0.05:
                prune_tree_list.append(i+2)

        self.trees = [self.trees[i] for i in range(0, len(self.trees), 1) if i not in prune_tree_list]   
        
        
        
    def pred_label(self, pre_prob):
        n, m = pre_prob.shape
        #print(n,m)
        exp_values = dict()
        predict_label = None
        final_label = np.zeros(n)
        for id in range(n):
            for label in range(m):
                exp_values[label] = exp(pre_prob[id,label])
            exp_sum = sum(exp_values.values())
            probs = dict()
            
            for label in exp_values:
                probs[label] = exp_values[label] / exp_sum
    

            for label in probs:
                if predict_label == None or probs[label]>probs[predict_label]:
                    predict_label = label
            final_label[id] = predict_label
            
        return final_label
    
 