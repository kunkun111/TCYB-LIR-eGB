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



def exp_synthetic(path, dataset_name, num_run, num_eval, exp_function, **exp_parm):
    #print(**exp_parm)
    np.random.seed(0)
    batch_acc = np.zeros([num_run, num_eval])

    for num_copy in range(num_run):
        print(num_copy, '/', num_run)
        data = load_arff(path, dataset_name, num_copy)
        acc,f1,pred = exp_function(data, **exp_parm)
        batch_acc[num_copy] = acc
        print('Total acc, ',metrics.accuracy_score(data[exp_parm['ini_train_size']:, -1],pred))

    print("%4f" % (batch_acc.mean()))
    print("%4f" % (batch_acc.mean(axis=1).std()))


def exp_realworld(path, dataset_name, num_run, exp_function, **exp_parm):

    aver_total_acc = np.zeros(num_run)
    aver_total_f1 = np.zeros(num_run)
    np.random.seed(0)
    data = load_arff(path, dataset_name, -1)
    num_eval = int(
        (data.shape[0] - exp_parm['ini_train_size']) / exp_parm['win_size'])
    batch_acc = np.zeros([num_run, num_eval])
    batch_f1 = np.zeros([num_run, num_eval])


    tqdm.write('='*20)
    tqdm.write((dataset_name + str(0)).center(20))
    batch_acc[0], batch_f1[0], pred = exp_function(data, **exp_parm)
    aver_total_acc[0] = metrics.accuracy_score(
        data[exp_parm['ini_train_size']:, -1], pred)
    aver_total_f1[0] = metrics.f1_score(
        data[exp_parm['ini_train_size']:, -1], pred,average='macro')
 

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

    


if __name__=="__main__":
    # Data path
    path = '/Synthetic/'
    num_run = 15
    num_eval = 99
    
    #initial parameter
    IE_single_parm = {
        'ini_train_size': 100,
        'win_size': 100,
        'max_tree': 10000,
        'num_ince_tree': 25
    }
    
    '''
    GBDT_pram = {
        'max_iter': 200,
        'sample_rate': 0.8,
        'learn_rate': 0.01,
        'max_depth': 4
    }
    '''
    
    # Run Drift Synthetic Datasets
    # data_name = ['LEDa']
    data_name = ['LEDg']
    classes = [10]
    
    for i in range (len(data_name)):
        print (data_name[i])
        
        GBDT_pram = {
            'max_iter': 200,
            'sample_rate': 0.8,
            'learn_rate': 0.01,
            'max_depth': 4,
            'num_label': classes[i]
        }
        
        IE_single_parm.update(GBDT_pram)
        dataset_name = data_name[i]
        exp_synthetic(path, dataset_name, num_run, num_eval, evaluation_Prune_GBDT,
                      **IE_single_parm)
    
