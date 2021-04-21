#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 14:52:55 2021

@author: kunwang
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from scipy.io import arff
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import arff
from tqdm import tqdm
import pixiedust
from copy import deepcopy
from portion_stat_gbdt import Portion_Stat_GBDT
from portion_gbdt import Portion_GBDT
from kappa_gbdt import Kappa_GBDT



def load_arff(path, dataset_name, num_copy):
    if num_copy == -1:
        file_path = path + dataset_name + '/'+ dataset_name + '.arff'
        dataset = arff.load(open(file_path), encode_nominal=True)
    else:
        file_path = path + dataset_name + '/'+ dataset_name + str(num_copy) + '.arff'
        dataset = arff.load(open(file_path), encode_nominal=True)
    return np.array(dataset["data"])




def kappa_digram(y_train_score,y_train,prune_list):
         # establish the label
        tree_label = (y_train_score >= 0.5)
        
        # calculate kappa score and error rate of pair of trees
        k = 0
        k_pair = []
        #print(tree_label.shape)
        for a in range(tree_label.shape[1]):
            for b in range(a+1, tree_label.shape[1]):
                tree_label_a = tree_label[:,a]
                tree_label_b = tree_label[:,b]
                k = metrics.cohen_kappa_score(tree_label_a, tree_label_b)
                err_rate = 1-(metrics.accuracy_score(y_train,tree_label_a)+metrics.accuracy_score(y_train,tree_label_b))/2
                k_pair.append([k,err_rate,a,b])
        
        # plot the kappa digram
        k_pair = np.array(k_pair)
        #np.savetxt('pair.out', k_pair , delimiter=',')
        x = k_pair[:,0]
        y = k_pair[:,1]
        
        plt.figure(figsize=(4, 3))
        plt.xlim((0.7, 1.02))
        plt.ylim((0.05, 0.15))
        plt.scatter(x,y,s=2, label='Initial trees')
        
        # establish the label
        tree_label = (y_train_score >= 0.5)
        
        # calculate kappa score and error rate of pair of trees
        k = 0
        k_pair1 = []
        #print(tree_label.shape)
        for a1 in range(len(prune_list)):
            for b1 in range(a1+1, len(prune_list)):
                tree_label_a1 = tree_label[:,prune_list[a1]]
                tree_label_b1 = tree_label[:,prune_list[b1]]
                k1 = metrics.cohen_kappa_score(tree_label_a1, tree_label_b1)
                err_rate1 = 1-(metrics.accuracy_score(y_train,tree_label_a1)+metrics.accuracy_score(y_train,tree_label_b1))/2
                k_pair1.append([k1,err_rate1,a1,b1])
        
        # plot the kappa digram
        k_pair1 = np.array(k_pair1)
        #print(k_pair1.shape)
        #np.savetxt('pair.out', k_pair , delimiter=',')
        x1 = k_pair1[:,0]
        y1 = k_pair1[:,1]
        #print(x1)
        
        #plt.figure(figsize=(4, 3))
        plt.xlim((0.7, 1.02))
        plt.ylim((0.05, 0.15))
        plt.scatter(x1,y1,s=2,c='r',marker='*',label='Naive prune')
        plt.xlabel('Kappa score')             
        plt.ylabel('Error rate')
        plt.legend(loc = 'lower left')
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.show()
        
        

def evaluation_Prune_GBDT_Stationary(data, n, **GBDT_pram):
# def evaluation_Prune_GBDT_Stationary(data, **GBDT_pram):
    
    # Load data 
    x = data[:, :-1]
    y = data[:, -1]
    
    # Data split
    x_learn, x_test, y_learn, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    x_train, x_prune, y_train, y_prune = train_test_split(x_learn, y_learn, test_size=0.3, random_state=0)
    
    # Training GBDT
    # model = Portion_GBDT(**GBDT_pram)
    # model = Portion_Stat_GBDT(**GBDT_pram)
    model = Kappa_GBDT(**GBDT_pram)
    y_train_score = model.fit(x_train, y_train)
    
    # Pruning GBDT
    model.predict(x_prune)
    # model.portion_pruning(y_prune,n)
    # model.portion_stat_pruning(y_prune,n)
    model.kappa_pruning(y_prune,n)
    
    # Testing GBDT
    y_pred_score2 = model.predict(x_test)
    y_pred_score2 = np.squeeze(y_pred_score2)
    y_pred_label2 = (y_pred_score2 >= 0.5)
    
    accuracy2 = metrics.accuracy_score(y_test, y_pred_label2.T)
    f12 = metrics.f1_score(y_test,y_pred_label2.T,average='macro')
    
    return accuracy2, f12




if __name__ == '__main__':
    
    # Data path
    path = '/Pruning_Real/'
    
    # initial parameter
    GBDT_pram = {
        'max_iter': 150,
        'sample_rate': 0.8,
        'learn_rate': 0.01,
        'max_depth': 4
    }
          
    
    # Data sets
    data_name = ['Australian Credit Approval', 'Boston', 'Chess', 'Diabetes', 
                  'EEG_eye_state','German Credit','Ionosphere','Ringnorm','Spambase','Twonorm']
    
    acc22=[]
    i = 10
    
    for j in range (len(data_name)):
        print(data_name[j])
        dataset_name = data_name[j]
        data = load_arff(path, dataset_name, -1)
        acc2, f12 = evaluation_Prune_GBDT_Stationary(data, i, **GBDT_pram)
        print('Accuracy:',acc2,'F1-score:',f12)

    
    
    