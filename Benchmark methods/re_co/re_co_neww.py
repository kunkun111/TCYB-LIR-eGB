#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 10:28:25 2021

@author: kunwang
"""


from skmultiflow.data import SEAGenerator
from skmultiflow.meta import AdaptiveRandomForestClassifier
from skmultiflow.meta import LearnPPNSEClassifier
from skmultiflow.meta import OnlineAdaC2Classifier
from skmultiflow.meta import OnlineBoostingClassifier
from skmultiflow.meta import OnlineRUSBoostClassifier
from skmultiflow.evaluation import EvaluatePrequential
import numpy as np
import arff
import pandas as pd
from skmultiflow.data.data_stream import DataStream
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score, roc_auc_score, roc_curve
import time


# load .arff dataset
def load_arff(path, dataset_name):
    file_path = path + dataset_name + '/'+ dataset_name + '.arff'
    dataset = arff.load(open(file_path), encode_nominal=True)
    return pd.DataFrame(dataset["data"])



def ARF_run (dataset_name, batch, random_seeds):
    data = load_arff(path, dataset_name)
    #print (data.shape)
    # data transform
    stream = DataStream(data)
    #print(stream)

    # Setup variables to control loop and track performance
    n_samples = 0
    max_samples = data.shape[0]

    # Train the classifier with the samples provided by the data stream
    pred = np.empty(0)
    np.random.seed(0)
    
    model = AdaptiveRandomForestClassifier(n_estimators=24, random_state=random_seeds)
    while n_samples < max_samples and stream.has_more_samples():
        X, y = stream.next_sample(batch)
        y_pred = model.predict(X)
        pred = np.hstack((pred,y_pred))
        model.partial_fit(X, y,stream.target_values)
        n_samples += batch

    # evaluate
    data = data.values
    Y = data[:,-1]
    acc = accuracy_score(Y[batch:], pred[batch:])
    f1 = f1_score(Y[batch:], pred[batch:], average='macro')
    #print (Y[batch:].shape, pred[batch:].shape)
    print("acc:",acc)
    print("f1:",f1)
    
    # save results
    result = np.zeros([pred[batch:].shape[0], 2])
    result[:, 0] = pred[batch:]
    result[:, 1] = Y[batch:]
    np.savetxt(dataset_name +'_' + 'ARF' + str(random_seeds) +'.out', result, delimiter=',')
    
    
    
def NSE_run (dataset_name, batch, random_seeds):
    data = load_arff(path, dataset_name)
    #print (data.shape)
    # data transform
    stream = DataStream(data)
    #print(stream)

    # Setup variables to control loop and track performance
    n_samples = 0
    max_samples = data.shape[0]

    # Train the classifier with the samples provided by the data stream
    pred = np.empty(0)
    np.random.seed(random_seeds)
    
    model = LearnPPNSEClassifier(n_estimators=24)
    while n_samples < max_samples and stream.has_more_samples():
        X, y = stream.next_sample(batch)
        y_pred = model.predict(X)
        pred = np.hstack((pred,y_pred))
        model.partial_fit(X, y,stream.target_values)
        n_samples += batch

    # evaluate
    data = data.values
    Y = data[:,-1]
    acc = accuracy_score(Y[batch:], pred[batch:])
    f1 = f1_score(Y[batch:], pred[batch:], average='macro')
    #print (Y[batch:].shape, pred[batch:].shape)
    print("acc:",acc)
    print("f1:",f1)
    
    # save results
    result = np.zeros([pred[batch:].shape[0], 2])
    result[:, 0] = pred[batch:]
    result[:, 1] = Y[batch:]
    np.savetxt(dataset_name +'_'+ 'NSE.out', result, delimiter=',')
    
    
def ADA_run (dataset_name, batch, random_seeds):
    data = load_arff(path, dataset_name)
    #print (data.shape)
    # data transform
    stream = DataStream(data)
    #print(stream)

    # Setup variables to control loop and track performance
    n_samples = 0
    max_samples = data.shape[0]

    # Train the classifier with the samples provided by the data stream
    pred = np.empty(0)
    np.random.seed(0)
    
    model = OnlineAdaC2Classifier(n_estimators=24, random_state=random_seeds)
    while n_samples < max_samples and stream.has_more_samples():
        X, y = stream.next_sample(batch)
        y_pred = model.predict(X)
        pred = np.hstack((pred,y_pred))
        model.partial_fit(X, y,stream.target_values)
        n_samples += batch

    # evaluate
    data = data.values
    Y = data[:,-1]
    acc = accuracy_score(Y[batch:], pred[batch:])
    f1 = f1_score(Y[batch:], pred[batch:], average='macro')
    #print (Y[batch:].shape, pred[batch:].shape)
    print("acc:",acc)
    print("f1:",f1)
    
    # save results
    result = np.zeros([pred[batch:].shape[0], 2])
    result[:, 0] = pred[batch:]
    result[:, 1] = Y[batch:]
    np.savetxt(dataset_name +'_'+ 'ADA' + str(random_seeds) +'.out', result, delimiter=',')
    
    
def OBC_run (dataset_name, batch, random_seeds):
    data = load_arff(path, dataset_name)
    #print (data.shape)
    # data transform
    stream = DataStream(data)
    #print(stream)

    # Setup variables to control loop and track performance
    n_samples = 0
    max_samples = data.shape[0]

    # Train the classifier with the samples provided by the data stream
    pred = np.empty(0)
    np.random.seed(0)
    
    model1 = OnlineBoostingClassifier(n_estimators=24, random_state=random_seeds)
    while n_samples < max_samples and stream.has_more_samples():
        X, y = stream.next_sample(batch)
        y_pred = model1.predict(X)
        pred = np.hstack((pred,y_pred))
        model1.partial_fit(X, y,stream.target_values)
        n_samples += batch

    # evaluate
    data = data.values
    Y = data[:,-1]
    acc = accuracy_score(Y[batch:], pred[batch:])
    f1 = f1_score(Y[batch:], pred[batch:], average='macro')
    #print (Y[batch:].shape, pred[batch:].shape)
    print("acc:",acc)
    print("f1:",f1)
    
    # save results
    result = np.zeros([pred[batch:].shape[0], 2])
    result[:, 0] = pred[batch:]
    result[:, 1] = Y[batch:]
    np.savetxt(dataset_name +'_'+ 'OBC_new' + str(random_seeds) +'.out', result, delimiter=',')
    

def RUS_run (dataset_name, batch, random_seeds):
    data = load_arff(path, dataset_name)
    #print (data.shape)
    # data transform
    stream = DataStream(data)
    #print(stream)

    # Setup variables to control loop and track performance
    n_samples = 0
    max_samples = data.shape[0]

    # Train the classifier with the samples provided by the data stream
    pred = np.empty(0)
    np.random.seed(0)
    
    model = OnlineRUSBoostClassifier(n_estimators=24, random_state=random_seeds)
    while n_samples < max_samples and stream.has_more_samples():
        X, y = stream.next_sample(batch)
        y_pred = model.predict(X)
        pred = np.hstack((pred,y_pred))
        model.partial_fit(X, y,stream.target_values)
        n_samples += batch

    # evaluate
    data = data.values
    Y = data[:,-1]
    acc = accuracy_score(Y[batch:], pred[batch:])
    f1 = f1_score(Y[batch:], pred[batch:], average='macro')
    print("acc:",acc)
    print("f1:",f1)
    
    # save results
    result = np.zeros([pred[batch:].shape[0], 2])
    result[:, 0] = pred[batch:]
    result[:, 1] = Y[batch:]
    np.savetxt(dataset_name +'_'+ 'RUS' + str(random_seeds) +'.out', result, delimiter=',')

    
    
    
if __name__ == '__main__':
    
    path = '/TCYB-LIR-eGB/Pruning_Real/'

# task1
    datasets = ['usenet1','usenet2']
    batch = [40,40]
 
# task2
    # datasets = ['spam_corpus_x2_feature_selected']
    # batch = [100]
    
# task3
    # datasets = ['elecNorm']
    # batch = [100]
    
# task4
    # datasets = ['weather']
    # batch = [365]

# task5
    # datasets = ['airline']
    # batch = [100]

# task6  
    # datasets = ['powersupply'] 
    # batch = [100]
    
# task7  
    # datasets = ['imdb'] 
    # batch = [100]
    
# task8   
    # datasets = ['mnista'] 
    # batch = [100]
    
# task9  
    # datasets = ['mnistg'] 
    # batch = [100]

        
    for i in range (len(datasets)):
        dataset_name = datasets[i]
        batch_size = batch[i]
        
        for j in range (0,5):
            
            print ('===================')
            start = time.time()
            NSE_run(dataset_name, batch_size, j)
            end = time.time()
            print (dataset_name, batch_size, 'NSE', 'random_seeds:', j, 'CPU_runtime:', end-start)
            
            
            print ('===================')
            start = time.time()
            OBC_run(dataset_name, batch_size, j)
            end = time.time()
            print (dataset_name, batch_size,'OBC','random_seeds:', j, 'CPU_runtime:', end-start)  
            
            
            print ('===================')
            start = time.time()
            RUS_run(dataset_name, batch_size, j)
            end = time.time()
            print (dataset_name, batch_size,'RUS','random_seeds:', j, 'CPU_runtime:', end-start)  
            
            
            print ('===================')
            start = time.time()
            ARF_run(dataset_name, batch_size, j)
            end = time.time()
            print (dataset_name, batch_size,'ARF','random_seeds:', j, 'CPU_runtime:', end-start)  
            
            
            print ('===================')
            start = time.time()
            ADA_run(dataset_name, batch_size, j)
            end = time.time()
            print (dataset_name, batch_size,'ADA','random_seeds:', j, 'CPU_runtime:', end-start) 
            


    
    