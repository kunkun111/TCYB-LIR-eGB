#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 21:07:50 2020

@author: kunwang
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from scipy.io import arff
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import arff
from tqdm import tqdm
import pixiedust
from copy import deepcopy

import lightgbm as lgb




def load_arff(path, dataset_name, num_copy):
    if num_copy == -1:
        file_path = path + dataset_name + '/'+ dataset_name + '.arff'
        dataset = arff.load(open(file_path), encode_nominal=True)
    else:
        file_path = path + dataset_name + '/'+ dataset_name + str(num_copy) + '.arff'
        dataset = arff.load(open(file_path), encode_nominal=True)
    return np.array(dataset["data"])



def evaluation_LGBM(data, ini_train_size, win_size):
    
    x_train = data[0:ini_train_size, :-1]
    y_train = data[0:ini_train_size, -1]
    
    lgb_train1 = lgb.Dataset(x_train, y_train)

    kf = KFold(int((data.shape[0] - ini_train_size) / win_size))
    stream = data[ini_train_size:, :]
    pred = np.zeros(stream.shape[0])
    accuracy = []
    f1 = []
        
    gbm = None
    params = { 'task': 'train', 
               # 'application': 'multiclass',  
                'application': 'binary', 
               'boosting_type': 'gbdt',
                # 'num_class': 10, # multiclass
               'learning_rate': 0.01, 
               'max_depth': 4,
               'verbose': -1} 
    
    gbm = lgb.train(params,
                    lgb_train1,
                    num_boost_round = 200,
                    verbose_eval = False,
                    keep_training_booster = True)
    # print(gbm.num_trees(),'ABC')
    

    for train_index, test_index in tqdm(kf.split(stream), total=kf.get_n_splits(), desc="#batch"):

        x_test = stream[test_index, :-1]
        y_test = stream[test_index, -1]
               
        #lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)
        
        y_pred_score = gbm.predict(x_test)
        # y_pred_label = [list(x).index(max(x)) for x in y_pred_score] # multiclass
        
        y_pred_score = np.squeeze(y_pred_score) # binary
        y_pred_label = (y_pred_score >= 0.5) # binary

        accuracy.append(metrics.accuracy_score(y_test, y_pred_label))
        f1.append(metrics.f1_score(y_test,y_pred_label,average='macro'))
        
        pred[test_index] = y_pred_label
        
        lgb_train = lgb.Dataset(x_test, y_test)
        
        #params = {'verbose': -1}
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round = 25,
                        #valid_sets = lgb_eval,
                        init_model = gbm,  # 如果gbm不为None，那么就是在上次的基础上接着训练
                        verbose_eval = False,
                        keep_training_booster = True)
        
        # print(gbm.num_trees())      
    
    return accuracy, f1, pred



def exp_synthetic(path, dataset_name, num_run, num_eval, exp_function, **exp_parm):
    np.random.seed(0)
    batch_acc = np.zeros([num_run, num_eval])

    for num_copy in range(num_run):
        print(num_copy, '/', num_run)
        data = load_arff(path, dataset_name, num_copy)
        #data = data[2400:2700,:]
        acc,f1,pred = exp_function(data, **exp_parm)
        batch_acc[num_copy] = acc
        print('Total acc, ',metrics.accuracy_score(data[exp_parm['ini_train_size']:, -1],pred))

    print("%4f" % (batch_acc.mean()))
    print("%4f" % (batch_acc.mean(axis=1).std()))
    #plt.plot(batch_acc.mean(axis=0))
    #plt.title(str(dataset_name))
    #plt.xlabel('Chunks')
    #plt.ylabel('Accuracy')
    #plt.show()
    


def exp_realworld(path, dataset_name, num_run, exp_function, **exp_parm):

    aver_total_acc = np.zeros(num_run)
    aver_total_f1 = np.zeros(num_run)
    np.random.seed(0)
    data = load_arff(path, dataset_name, -1)
    num_eval = int(
        (data.shape[0]-exp_parm['ini_train_size']) / exp_parm['win_size'])
    batch_acc = np.zeros([num_run, num_eval])
    batch_f1 = np.zeros([num_run, num_eval])


    tqdm.write('='*20)
    tqdm.write((dataset_name + str(0)).center(20))
    batch_acc[0], batch_f1[0], pred = exp_function(data, **exp_parm)
    aver_total_acc[0] = metrics.accuracy_score(
        data[exp_parm['ini_train_size']:, -1], pred)
    aver_total_f1[0] = metrics.f1_score(
        data[exp_parm['ini_train_size']:, -1], pred, average='macro')
    #tqdm.write('Current r_seed acc,' + str(aver_total_acc[0]))
    #tqdm.write('Current r_seed f1,' + str(aver_total_f1[0]))
    #plt.plot(batch_acc[0])
    #plt.title(str(dataset_name))
    #plt.xlabel('Chunks')
    #plt.ylabel('Accuracy')
    #plt.show()
 

    for r_seed in range(1, num_run):
        np.random.seed(r_seed)
        data = load_arff(path, dataset_name, -1)
        num_eval = int((data.shape[0] - exp_parm['ini_train_size']) /
                       exp_parm['win_size'])
        tqdm.write('='*20)
        #tqdm.write((dataset_name + str(r_seed)).center(20))
        batch_acc[r_seed], batch_f1[r_seed], pred = exp_function(data, **exp_parm)
        aver_total_acc[r_seed] = metrics.accuracy_score(
            data[exp_parm['ini_train_size']:, -1], pred)
        #tqdm.write('Current r_seed acc,' + str(aver_total_acc[r_seed]))
        aver_total_f1[r_seed] = metrics.f1_score(
            data[exp_parm['ini_train_size']:, -1], pred, average='macro')
        
        #tqdm.write('Current r_seed acc,' + str(aver_total_acc[r_seed]))
        #tqdm.write('Current r_seed f1,' + str(aver_total_f1[r_seed]))
        
    tqdm.write('Average acc,' + str(np.mean(aver_total_acc)))
    tqdm.write('Average f1,' + str(np.mean(aver_total_f1)))
    #tqdm.write('Std acc,' + str(np.std(aver_total_acc)))
    
    print(pred.shape)
    print(data[exp_parm['ini_train_size']:, -1].shape)
    result = np.zeros([pred.shape[0], 2])
    result[:, 0] = pred
    result[:, 1] = data[exp_parm['ini_train_size']:, -1]
    #result = np.hstack([data[exp_parm['ini_train_size']:, -1], pred.T])
    #np.savetxt(str(dataset_name)+'_stat_base200incre25_05.out', result , delimiter=',')
    


# Parameter Setting
path = '/Synthetic/'
num_run = 15
num_eval = 99


#initial parameter
IE_single_parm = {
    'ini_train_size': 100,
    'win_size': 100
}



# Run Drift Synthetic Datasets

data_name=['SEAa','RTG','RBF','RBFr','AGRa','HYP']
# data_name=['LEDa','LEDg']
for i in range (len(data_name)):
    print (data_name[i])
    dataset_name = data_name[i]
    exp_synthetic(path, dataset_name, num_run, num_eval, evaluation_LGBM,
                  **IE_single_parm)
'''   
# Pruning Real
path_rel = '/Pruning_Real/'
num_run = 1


datasets = ['spam_corpus_x2_feature_selected','elecNorm']

# datasets = ['airline']

# datasets = ['weather']

# datasets = ['usenet1','usenet2']

# datasets = ['powersupply']


for j in range (len(datasets)):
    dataset_name = datasets[j]
    exp_realworld(path_rel, dataset_name, num_run, evaluation_LGBM, **IE_single_parm)
'''
