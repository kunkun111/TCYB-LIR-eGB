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
from syn_gbdt_multi import multi_GBDT



def load_arff(path, dataset_name, num_copy):
    if num_copy == -1:
        file_path = path + dataset_name + '/'+ dataset_name + '.arff'
        dataset = arff.load(open(file_path), encode_nominal=True)
    else:
        file_path = path + dataset_name + '/'+ dataset_name + str(num_copy) + '.arff'
        dataset = arff.load(open(file_path), encode_nominal=True)
    return np.array(dataset["data"])



def evaluation_Prune_GBDT(data, ini_train_size, win_size, max_tree, num_ince_tree, **GBDT_pram):
    x_train = data[0:ini_train_size, :-1]
    y_train = data[0:ini_train_size, -1]
    
    model1 = multi_GBDT(**GBDT_pram)
    model1.fit(x_train, y_train)
    
    model2 = multi_GBDT(**GBDT_pram)
    model2.fit(x_train, y_train)

    kf = KFold(int((data.shape[0] - ini_train_size) / win_size))
    stream = data[ini_train_size:, :]
    pred = np.zeros(stream.shape[0])
    accuracy = []
    f1 = []

    for train_index, test_index in tqdm(kf.split(stream), total=kf.get_n_splits(), desc="#batch"):

        x_test = stream[test_index, :-1]
        y_test = stream[test_index, -1]
        
        # Step 1. Make Predictionmars
        y_pred_score1, y_loss1, y_loss_array1 = model1.predict(x_test, y_test)
        y_pred_label1 = model1.pred_label(y_pred_score1)
        
        y_pred_score2, y_loss2, y_loss_array2 = model2.predict(x_test, y_test)
        y_pred_label2 = model2.pred_label(y_pred_score2)
        
        acc1 = metrics.accuracy_score(y_test,y_pred_label1.T)
        acc2 = metrics.accuracy_score(y_test,y_pred_label2.T)
        
        
        # Step 2. NP-SP switch
        if acc1 > acc2:
            accuracy.append(acc1)
            f1.append(metrics.f1_score(y_test,y_pred_label1.T,average='macro'))
            pred[test_index] = y_pred_label1
        else:
            accuracy.append(acc2)
            f1.append(metrics.f1_score(y_test,y_pred_label2.T,average='macro'))
            pred[test_index] = y_pred_label2

        
        # Step 3. Purning GBDT
        model1.portion_pruning(y_loss1)        
        model2.portion_stat_pruning(y_loss_array2)
        
        
        # Step 3. Incremental GBDT
        if len(model1.trees) < 200:
            model1.incre_fit(x_test, y_test, y_pred_score1, 200-len(model1.trees)+num_ince_tree)
        else:
            model1.incre_fit(x_test, y_test, y_pred_score1, num_ince_tree)
            
        if len(model2.trees) < 200:
            model2.incre_fit(x_test, y_test, y_pred_score2, 200-len(model2.trees)+num_ince_tree)
        else:
            model2.incre_fit(x_test, y_test, y_pred_score2, num_ince_tree)
            
    tqdm.write('Num tree at the end,' + str(len(model1.trees)))
    tqdm.write('Num tree at the end,' + str(len(model2.trees)))
    
    return accuracy, f1, pred



def exp_realworld(path, dataset_name, num_run, exp_function, **exp_parm):

    aver_total_acc = np.zeros(num_run)
    aver_total_f1 = np.zeros(num_run)
    data = load_arff(path, dataset_name, -1)
    # data = data[0:300, :]
    num_eval = int(
        (data.shape[0] - exp_parm['ini_train_size']) / exp_parm['win_size'])
    batch_acc = np.zeros([num_run, num_eval])
    batch_f1 = np.zeros([num_run, num_eval])
 

    for r_seed in range(0, num_run):
        
        print(str(dataset_name), 'randomn_seeds:', str(r_seed))
        
        np.random.seed(r_seed)
        
        start = time.time()
        batch_acc[r_seed], batch_f1[r_seed], pred = exp_function(data, **exp_parm)
        end = time.time()
        
        aver_total_acc[r_seed] = metrics.accuracy_score(
            data[exp_parm['ini_train_size']:, -1], pred)
        aver_total_f1[r_seed] = metrics.f1_score(
            data[exp_parm['ini_train_size']:, -1], pred, average='macro')
        
        tqdm.write('Current r_seed acc,' + str(aver_total_acc[r_seed]))
        tqdm.write('Current r_seed f1,' + str(aver_total_f1[r_seed]))
        
        print ('run_time:',end-start)
        print('=========================')
        
        result = np.zeros([pred.shape[0], 2])
        result[:, 0] = pred
        result[:, 1] = data[exp_parm['ini_train_size']:, -1]
        # np.savetxt(str(dataset_name)+ str(r_seed) +'_naive_base200incre25.out', result , delimiter=',')
        
    
    print('***********************************************')
    tqdm.write('Average acc,' + str(np.mean(aver_total_acc)))
    tqdm.write('Average f1,' + str(np.mean(aver_total_f1)))
    tqdm.write('Std acc,' + str(np.std(aver_total_acc)))
    print('***********************************************')
    


if __name__=="__main__":
   
    #initial parameter
    IE_single_parm = {
        'ini_train_size': 100,
        'win_size': 100,
        'max_tree': 10000,
        'num_ince_tree': 25
    }
    
    # run real-world datasets
    path_rel = '/home/kunwang/Data/TCYB-LIR-eGB/Pruning_Real/'
    num_run = 5
                    
    datasets = ['powersupply']
    classes = [24]
    
    for i in range (len(datasets)):
        GBDT_pram = {
            'max_iter': 200,
            'sample_rate': 0.8,
            'learn_rate': 0.01,
            'max_depth': 4,
            'num_label': classes[i]
        }
        
        IE_single_parm.update(GBDT_pram)
        dataset_name = datasets[i]
        exp_realworld(path_rel, dataset_name, num_run, evaluation_Prune_GBDT,
                  **IE_single_parm)
        