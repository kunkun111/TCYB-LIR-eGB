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
from copy import deepcopy
from portion_stat_gbdt import Portion_Stat_GBDT
import time



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
    model = Portion_Stat_GBDT(**GBDT_pram)
    model.fit(x_train, y_train)

    kf = KFold(int((data.shape[0] - ini_train_size) / win_size))
    stream = data[ini_train_size:, :]
    pred = np.zeros(stream.shape[0])
    accuracy = []
    f1 = []

    for train_index, test_index in tqdm(kf.split(stream), total=kf.get_n_splits(), desc="#batch"):

        x_test = stream[test_index, :-1]
        y_test = stream[test_index, -1]
        
        # Step 1. Make Predictionmars
        y_pred_score = model.predict(x_test)
        y_pred_score = np.squeeze(y_pred_score)
        y_pred_label = (y_pred_score >= 0.5)

        accuracy.append(metrics.accuracy_score(y_test, y_pred_label.T))
        f1.append(metrics.f1_score(y_test,y_pred_label.T,average='macro'))
        
        pred[test_index] = y_pred_label
        
        # Step 2. Purning GBDT
        model.portion_stat_pruning(y_test)
        y_pred_score = np.squeeze(model.predict(x_test))
        
        # Step 3. Incremental GBDT
        if len(model.dtrees) < 200:
            model.incremental_fit(x_test, y_test, y_pred_score, 200-len(model.dtrees)+num_ince_tree)
        else:
            model.incremental_fit(x_test, y_test, y_pred_score, num_ince_tree)
    tqdm.write('Num tree at the end,' + str(len(model.dtrees)))
    
    return accuracy, f1, pred



def exp_synthetic(path, dataset_name, num_run, num_eval, exp_function, **exp_parm):
    np.random.seed(0)
    batch_acc = np.zeros([num_run, num_eval])
    run_time = []

    for num_copy in range(num_run):
        print(num_copy, '/', num_run)
        data = load_arff(path, dataset_name, num_copy)
        
        start = time.time()
        acc,f1,pred = exp_function(data, **exp_parm)
        end = time.time()
        print ('run_time:',end-start)
        run_time.append(end-start)
        
        batch_acc[num_copy] = acc
        print('Total acc, ',metrics.accuracy_score(data[exp_parm['ini_train_size']:, -1],pred))

    print('***********************************************')
    print('batch_acc_mean:', (batch_acc.mean()))
    print('batch_acc_std:' , (batch_acc.mean(axis=1).std()))
    
    # print(run_time)
    print('run_time_mean:' , np.mean(run_time))
    print('run_time_std:' , np.std(run_time))
    print('***********************************************')
    
    #plt.plot(batch_acc.mean(axis=0))
    #plt.title(str(dataset_name))
    #plt.xlabel('Chunks')
    #plt.ylabel('Accuracy')
    #plt.show()
    


def exp_realworld(path, dataset_name, num_run, exp_function, **exp_parm):

    aver_total_acc = np.zeros(num_run)
    aver_total_f1 = np.zeros(num_run)
    data = load_arff(path, dataset_name, -1)
    num_eval = int(
        (data.shape[0] - exp_parm['ini_train_size']) / exp_parm['win_size'])
    batch_acc = np.zeros([num_run, num_eval])
    batch_f1 = np.zeros([num_run, num_eval])
 
    for r_seed in range(0, num_run):
        
        print(str(dataset_name), 'randome_seeds:', str(r_seed))
        
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
        np.savetxt(str(dataset_name)+ str(r_seed) +'_state_005_base200incre25.out', result , delimiter=',')
        
    
    print('***********************************************')
    tqdm.write('Average acc,' + str(np.mean(aver_total_acc)))
    tqdm.write('Std acc,' + str(np.std(aver_total_acc)))
    tqdm.write('Average f1,' + str(np.mean(aver_total_f1)))
    tqdm.write('Std f1,' + str(np.std(aver_total_f1)))
    print('***********************************************')

 


if __name__ == '__main__':
    
    # Data path
    path = '/TCYB-LIR-eGB/Synthetic/'
    num_run = 15
    num_eval = 99
    
    
    #initial parameter
    IE_single_parm = {
        'ini_train_size': 100,
        'win_size': 100,
        'max_tree': 10000,
        'num_ince_tree': 25
    }
    
    GBDT_pram = {
        'max_iter': 200,
        'sample_rate': 0.8,
        'learn_rate': 0.01,
        'max_depth': 4
    }
    
    '''
    # Run Drift Synthetic Datasets
    data_name=['SEAa','RTG','RBF','RBFr','AGRa','HYP']
    for i in range (len(data_name)):
        print (data_name[i])
        IE_single_parm.update(GBDT_pram)
        dataset_name = data_name[i]
        exp_synthetic(path, dataset_name, num_run, num_eval, evaluation_Prune_GBDT,
                      **IE_single_parm)
    
    '''   
    # Pruning Real
    path_rel = '/TCYB-LIR-eGB/Pruning_Real/'
    num_run = 5
    
    # datasets = ['usenet1','usenet2']
    
    # datasets = ['weather']
    
    datasets = ['spam_corpus_x2_feature_selected','elecNorm','imdb','mnista','mnistg','airline']
    
    for j in range (len(datasets)):
        IE_single_parm.update(GBDT_pram)
        dataset_name = datasets[j]
        exp_realworld(path_rel, dataset_name, num_run, evaluation_Prune_GBDT, **IE_single_parm)
    
    
    