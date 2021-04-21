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
    model = multi_GBDT(**GBDT_pram)
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
        y_pred_score, y_loss, y_loss_array = model.predict(x_test, y_test)
        y_pred_label = model.pred_label(y_pred_score)

        accuracy.append(metrics.accuracy_score(y_test, y_pred_label.T))
        f1.append(metrics.f1_score(y_test,y_pred_label.T,average='macro'))
        
        pred[test_index] = y_pred_label
        
        # Step 2. Purning GBDT
        model.portion_pruning(y_loss)
        
        # Step 3. Incremental GBDT
        if len(model.trees) < 200:
            model.incre_fit(x_test, y_test, y_pred_score, 200-len(model.trees)+num_ince_tree)
        else:
            model.incre_fit(x_test, y_test, y_pred_score, num_ince_tree)
        tqdm.write('Num tree at the end,' + str(len(model.trees)))
    
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
    #data =  data[0:503,:]
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
        #data =  data[0:503,:]
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
    #np.savetxt(str(dataset_name)+'_re_multi_naive.out', result , delimiter=',')
    


if __name__=="__main__":
   
    #initial parameter
    IE_single_parm = {
        'ini_train_size': 100,
        'win_size': 100,
        'max_tree': 10000,
        'num_ince_tree': 25
    }
    
    # run real-world datasets
    path_rel = '/Pruning_Real/'
    num_run = 1
                    
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
        