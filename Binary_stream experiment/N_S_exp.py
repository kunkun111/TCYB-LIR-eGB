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
from N_S_gbdt import N_S_GBDT



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
    model1 = N_S_GBDT(**GBDT_pram)
    model1.fit(x_train, y_train)
    
    model2 = N_S_GBDT(**GBDT_pram)
    model2.fit(x_train, y_train)

    kf = KFold(int((data.shape[0] - ini_train_size) / win_size))
    stream = data[ini_train_size:, :]
    pred = np.zeros(stream.shape[0])
    accuracy = []
    f1 = []
    
    frequency = []

    for train_index, test_index in tqdm(kf.split(stream), total=kf.get_n_splits(), desc="#batch"):

        x_test = stream[test_index, :-1]
        y_test = stream[test_index, -1]
        
        # Step 1. Make Predictionmars
        y_pred_score1 = model1.predict(x_test)
        y_pred_score1 = np.squeeze(y_pred_score1)
        y_pred_label1 = (y_pred_score1 >= 0.5)
        
        y_pred_score2 = model2.predict(x_test)
        y_pred_score2 = np.squeeze(y_pred_score2)
        y_pred_label2 = (y_pred_score2 >= 0.5)
        
        acc1 = metrics.accuracy_score(y_test,y_pred_label1.T)
        acc2 = metrics.accuracy_score(y_test,y_pred_label2.T)
        
        
        # Step 2. NP-SP switch
        if acc1 > acc2:
            accuracy.append(acc1)
            f1.append(metrics.f1_score(y_test,y_pred_label1.T,average='macro'))
            pred[test_index] = y_pred_label1
            frequency.append(0)
        else:
            accuracy.append(acc2)
            f1.append(metrics.f1_score(y_test,y_pred_label2.T,average='macro'))
            pred[test_index] = y_pred_label2
            frequency.append(1)
        
        
        # Step 3. Purning GBDT      
        model1.portion_pruning(y_test) 
        model2.portion_stat_pruning(y_test)
          
        
        # Step 3. Incremental GBDT
        # model 1
        if len(model1.dtrees) < 200:
            model1.incremental_fit(x_test, y_test, y_pred_score1, 200-len(model1.dtrees)+num_ince_tree)
        else:
            model1.incremental_fit(x_test, y_test, y_pred_score1, num_ince_tree)
            
        # model 2
        if len(model2.dtrees) < 200:
            model2.incremental_fit(x_test, y_test, y_pred_score2, 200-len(model2.dtrees)+num_ince_tree)
        else:
            model2.incremental_fit(x_test, y_test, y_pred_score2, num_ince_tree)
            
    tqdm.write('Num tree at the end,' + str(len(model1.dtrees)))
    tqdm.write('Num tree at the end,' + str(len(model2.dtrees))) 
#    print(frequency)
    
    return accuracy, f1, pred


# modified SEA dataset
def SEA():
    data = np.empty((0,4))
    # theta = [10,7,3,7,10,13,16,13,7,10]
    theta = [1,3,5,7,9,11,13,15,17,19]
    for t in range (len(theta)):   
        a = np.zeros([1000,4])    
        a[:,0] = np.random.rand(1000)*10
        a[:,1] = np.random.rand(1000)*10
        a[:,2] = np.random.normal(1,0.1,1000)
    
        for i in range (a.shape[0]):
            if a[i,0]+a[i,1] <= theta[t]:
                a[i,3]=0
            else:
                a[i,3]=1
    
        plt.figure(figsize=(6,4))
        plt.scatter(a[:,0], a[:,1], marker='o', c=a[:,3], cmap='coolwarm')
        font_size = 12
        plt.tick_params(labelsize=font_size)
        plt.show()
        
        data = np.vstack((data,a))
        
    return data



def exp_synthetic(path, dataset_name, num_run, num_eval, exp_function, **exp_parm):
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
    result = np.zeros([pred.shape[0], 2])
    result[:, 0] = pred
    result[:, 1] = data[exp_parm['ini_train_size']:, -1]
    #result = np.hstack([data[exp_parm['ini_train_size']:, -1], pred.T])
    # np.savetxt(str(dataset_name)+'_stat_base200incre25_05.out', result , delimiter=',')
    


if __name__ == '__main__':
    
    # Data path
    path = '/Synthetic/'
    num_run = 1
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
    path_rel = '/data/kunwang/Work2/Pruning_Real/'
    num_run = 1
    
    # datasets = ['airline']
    
    datasets = ['spam_corpus_x2_feature_selected','elecNorm']
    
    # datasets = ['usenet1','usenet2']
    
    # datasets = ['weather']
    
    
    
    for j in range (len(datasets)):
        IE_single_parm.update(GBDT_pram)
        dataset_name = datasets[j]
        exp_realworld(path_rel, dataset_name, num_run, evaluation_Prune_GBDT, **IE_single_parm)
    '''