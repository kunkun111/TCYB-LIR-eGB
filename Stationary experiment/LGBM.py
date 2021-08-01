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
import time
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


# def evaluation_LGBM(data, ini_train_size, win_size, seeds):
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
                'application': 'multiclass',  
                # 'application': 'binary', 
               'boosting_type': 'gbdt',
                'num_class': 10, # multiclass
               'learning_rate': 0.01, 
               'max_depth': 4,
               # 'bagging_fraction' : 0.8,  
               # 'bagging_freq' : 5,        
               # 'bagging_seed' : 0,
               'verbose': -1,
               'random_seed': 0} 
               # 'random_seed': 0}
    
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
        y_pred_label = [list(x).index(max(x)) for x in y_pred_score] # multiclass
        
        # y_pred_score = np.squeeze(y_pred_score) # binary
        # y_pred_label = (y_pred_score >= 0.5) # binary

        accuracy.append(metrics.accuracy_score(y_test, y_pred_label))
        f1.append(metrics.f1_score(y_test,y_pred_label,average='macro'))
        
        pred[test_index] = y_pred_label
        
        lgb_train = lgb.Dataset(x_test, y_test)
        
        #params = {'verbose': -1}
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round = 25,
                        init_model = gbm,  # 如果gbm不为None，那么就是在上次的基础上接着训练
                        verbose_eval = False,
                        keep_training_booster = True)
        
        # print(gbm.num_trees())      
    
    return accuracy, f1, pred



def exp_synthetic(path, dataset_name, num_run, num_eval, exp_function, **exp_parm):
    np.random.seed(0)
    batch_acc = np.zeros([num_run, num_eval])
    runtime = []

    for num_copy in range(num_run):
        print(num_copy, '/', num_run)
        data = load_arff(path, dataset_name, num_copy)
        #data = data[2400:2700,:]
        
        start = time.time()
        acc,f1,pred = exp_function(data, **exp_parm)
        end = time.time()
        print ('CPU time', end-start)
        runtime.append(end-start)
        
        batch_acc[num_copy] = acc
        print('Total acc, ',metrics.accuracy_score(data[exp_parm['ini_train_size']:, -1],pred))
    
    print('****************************************')
    print("%4f" % (batch_acc.mean()))
    print("%4f" % (batch_acc.mean(axis=1).std()))
    print('average runtime', np.mean(runtime))
    print('std runtime', np.std(runtime))
    print('****************************************')
    


def exp_realworld(path, dataset_name, exp_function, **exp_parm):

    data = load_arff(path, dataset_name, -1)


    tqdm.write('='*20)
    tqdm.write((dataset_name + str(0)).center(20))
    
    start = time.time()
    batch_acc, batch_f1, pred = exp_function(data, **exp_parm)
    end = time.time()
    print ('CPU time', end-start)

    
    aver_total_acc = metrics.accuracy_score(
        data[exp_parm['ini_train_size']:, -1], pred)
    aver_total_f1 = metrics.f1_score(
        data[exp_parm['ini_train_size']:, -1], pred, average='macro')
    
    return aver_total_acc, aver_total_f1, end-start
    


if __name__ == '__main__':

    # Parameter Setting
    path = '/TCYB-LIR-eGB/Synthetic/'
    num_run = 15
    num_eval = 99
    
   
    #initial parameter
    IE_single_parm = {
        'ini_train_size': 100,
        'win_size': 100
    }
    
    
    
    # Run Drift Synthetic Datasets
    # data_name=['SEAa','RTG','RBF','RBFr','AGRa','HYP']
    data_name=['LEDa','LEDg']
    for i in range (len(data_name)):
        print (data_name[i])
        dataset_name = data_name[i]
        exp_synthetic(path, dataset_name, num_run, num_eval, evaluation_LGBM,
                      **IE_single_parm)
    '''   
    # Pruning Real
    path_rel = '/home/kunwang/Data/TCYB-LIR-eGB/Pruning_Real/'
    num_run = 5
    
    
    # datasets = ['usenet1','usenet2']
    
    # datasets = ['powersupply']
    
    # datasets = ['weather']
    
    datasets = ['spam_corpus_x2_feature_selected','elecNorm','imdb','mnista','mnistg','airline']
    
    
    
    #initial parameter
    for j in range (len(datasets)):
        dataset_name = datasets[j]
        
        total_acc = []
        total_f1 = []
        runtime = []

        for i in range (0, num_run):
            IE_single_parm = {
                'ini_train_size': 100,
                'win_size': 100,
                'seeds': i
            }

            acc, f1, times = exp_realworld(path_rel, dataset_name, evaluation_LGBM, **IE_single_parm)

            total_acc.append(acc)
            total_f1.append(f1)
            runtime.append(times)
            
            print('accuracy', acc, 'f1_score',f1, 'random_seed:', i)
            
        print('==============================')
        print('average acc:', np.mean(total_acc))
        print('std acc:', np.std(total_acc))
        print('average f1:', np.mean(total_f1))
        print('std f1:', np.std(total_f1))
        print('average time:', np.mean(runtime))
        print('==============================')
 '''       

