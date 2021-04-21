#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 14:52:22 2021

@author: kunwang
"""


import xgboost as xgb
import arff
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
from sklearn.model_selection import train_test_split



# load dataset
def load_arff(path, dataset_name, num_copy):
    if num_copy == -1:
        file_path = path + dataset_name + '/'+ dataset_name + '.arff'
        dataset = arff.load(open(file_path), encode_nominal=True)
    else:
        file_path = path + dataset_name + '/'+ dataset_name + str(num_copy) + '.arff'
        dataset = arff.load(open(file_path), encode_nominal=True)
    return np.array(dataset["data"])


path = '/Pruning_Real/'
num_copy = -1
# dataset_name = ['Australian Credit Approval', 'Boston', 'Chess', 'Diabetes', 
              # 'EEG_eye_state','German Credit','Ionosphere','Ringnorm','Spambase','Twonorm']
dataset_name = ['Glass','Image','Iris','Letters','Lymph','Thyroid','Vehicle','Vowel','Waveform','Wine']
classes = [7,7,3,26,4,3,4,11,3,3]

for i in range (len(dataset_name)):
    data = load_arff(path,dataset_name[i],num_copy)
    print(str(dataset_name[i]))
    x = data[:,:-1]
    y = data[:,-1]
    x_learn, x_test, y_learn, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    x_train, x_prune, y_train, y_prune= train_test_split(x_learn, y_learn, test_size=0.3, random_state=0)

    # read in data
    dtrain = xgb.DMatrix(x_train, label = y_train)
    dtest = xgb.DMatrix(x_test,label = y_test)
    
    # specify parameters via map---Binary
    param1 = {'booster': 'dart',
             'max_depth': 4, 
             'learning_rate': 0.01,
             'objective': 'binary:logistic', 
             'sample_type': 'uniform',
             'normalize_type': 'tree',
             'rate_drop': 0.1,
             'skip_drop': 0.5}
    

    # specify parameters via map---multi
    param2 = {'booster': 'dart',
             'max_depth': 4, 'learning_rate': 0.01,
             'objective': 'multi:softmax', 
             'sample_type': 'uniform',
             'normalize_type': 'tree',
              'num_class': classes[i],
             'rate_drop': 0.1,
             'skip_drop': 0.5}

    num_round = 150
    # bst = xgb.train(param1, dtrain, num_round) # Binary
    bst = xgb.train(param2, dtrain, num_round) # Multi
    
    preds = bst.predict(dtest)
    # y_label = (preds >= 0.5)*1

    # multi
    print('acc:', accuracy_score(preds, y_test))
    print('f1:', f1_score(preds, y_test, average='macro'))
    
    # binary
    # print('acc:', accuracy_score(y_label,y_test))
    # print('f1:', f1_score(y_label,y_test,average='macro'))



