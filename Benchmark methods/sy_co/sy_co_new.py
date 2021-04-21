#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 20:56:58 2021

@author: kunwang
"""

 
# Imports
from skmultiflow.meta import AdaptiveRandomForestClassifier
from skmultiflow.meta import LearnPPNSEClassifier
from skmultiflow.meta import OnlineAdaC2Classifier
from skmultiflow.meta import OnlineBoostingClassifier
from skmultiflow.meta import OnlineRUSBoostClassifier
import numpy as np
import arff
import pandas as pd
from skmultiflow.data.data_stream import DataStream
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score, roc_auc_score, roc_curve



# load .arff dataset
def load_arff(path, dataset_name, num_copy):
    file_path = path + dataset_name + '/'+ dataset_name + str(num_copy) + '.arff'
    dataset = arff.load(open(file_path), encode_nominal=True)
    return pd.DataFrame(dataset["data"])




def ARF_run (dataset_name, batch, num_copy):
    data = load_arff(path, dataset_name, num_copy)
    # data transform
    stream = DataStream(data)
    #print(stream)

    # Setup variables to control loop and track performance
    n_samples = 0
    max_samples = data.shape[0]

    # Train the classifier with the samples provided by the data stream
    pred = np.empty(0)
    np.random.seed(0)
    
    model = AdaptiveRandomForestClassifier()
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
    # np.savetxt(dataset_name +'_' + str(num_copy)+ '_ARF.out', result, delimiter=',')
    
    
def NSE_run (dataset_name, batch, num_copy):
    data = load_arff(path, dataset_name, num_copy)
    # data transform
    stream = DataStream(data)
    #print(stream)

    # Setup variables to control loop and track performance
    n_samples = 0
    max_samples = data.shape[0]

    # Train the classifier with the samples provided by the data stream
    pred = np.empty(0)
    np.random.seed(0)
    
    model = LearnPPNSEClassifier()
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
    # np.savetxt(dataset_name +'_' + str(num_copy)+ '_NSE.out', result, delimiter=',')
    

    
    
def ADA_run (dataset_name, batch, num_copy):
    data = load_arff(path, dataset_name, num_copy)
    # data transform
    stream = DataStream(data)
    #print(stream)

    # Setup variables to control loop and track performance
    n_samples = 0
    max_samples = data.shape[0]

    # Train the classifier with the samples provided by the data stream
    pred = np.empty(0)
    np.random.seed(0)
    
    model = OnlineAdaC2Classifier()
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
    # np.savetxt(dataset_name +'_' + str(num_copy)+ '_ADA.out', result, delimiter=',')
    
    
def OBC_run (dataset_name, batch, num_copy):
    data = load_arff(path, dataset_name, num_copy)
    # data transform
    stream = DataStream(data)
    #print(stream)

    # Setup variables to control loop and track performance
    n_samples = 0
    max_samples = data.shape[0]

    # Train the classifier with the samples provided by the data stream
    pred = np.empty(0)
    np.random.seed(0)
    
    model = OnlineBoostingClassifier()
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
    # np.savetxt(dataset_name +'_' + str(num_copy)+ '_OBC.out', result, delimiter=',')
    

def RUS_run (dataset_name, batch, num_copy):
    data = load_arff(path, dataset_name, num_copy)
    # data transform
    stream = DataStream(data)
    #print(stream)

    # Setup variables to control loop and track performance
    n_samples = 0
    max_samples = data.shape[0]

    # Train the classifier with the samples provided by the data stream
    pred = np.empty(0)
    np.random.seed(0)
    
    model = OnlineRUSBoostClassifier()
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
    # np.savetxt(dataset_name +'_' + str(num_copy)+ '_RUS.out', result, delimiter=',')

    
    
    
if __name__ == '__main__':
    
    
    #datasets = ['AGRa','HYP','RBF','RBFr','RTG','SEAa'] 
    datasets = ['LEDg'] 
    path = '/Synthetic/'
    batch_size = 100
    num_run = 15


    for i in range (len(datasets)):
        dataset_name = datasets[i]
        for num_copy in range(num_run):
            print(num_copy, '/', num_run)
            print('dataset', dataset_name, num_copy, 'batch_size', batch_size)
            
        
            print (dataset_name, batch_size,'ARF')
            ARF_run(dataset_name, batch_size, num_copy)
                 
            print (dataset_name, batch_size,'NSE')
            NSE_run(dataset_name, batch_size, num_copy)
                    
            print (dataset_name, batch_size,'ADA')
            ADA_run(dataset_name, batch_size, num_copy)
                   
            print (dataset_name, batch_size,'OBC')
            OBC_run(dataset_name, batch_size, num_copy)
                     
            print (dataset_name, batch_size,'RUS')
            RUS_run(dataset_name, batch_size, num_copy) 
            
            

    