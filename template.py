# -*- coding: utf-8 -*-
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel, mutual_info_classif, SelectPercentile
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import model_selection
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
import pickle
import os
import sys
import argparse
from sklearn.neural_network import MLPClassifier

def RMS(data):
    return np.sqrt(np.sum(data ** 2)/len(data))

def Energy(data):
    return np.sum(data ** 2)
    
def extract_features_group(df, columns, win_size):
    df_mean = df.groupby('id')[columns].apply(pd.rolling_mean, win_size, min_periods=1)
    df_std = df.groupby('id')[columns].apply(pd.rolling_std, win_size, min_periods=1)
    df_std = df_std.fillna(0)
    df_median = df.groupby('id')[columns].apply(pd.rolling_median, win_size, min_periods=1)
    df_min = df.groupby('id')[columns].apply(pd.rolling_min, win_size, min_periods=1)
    df_max = df.groupby('id')[columns].apply(pd.rolling_max, win_size, min_periods=1)
    df_quantile = df.groupby('id')[columns].apply(lambda x: pd.rolling_quantile(x,win_size,0.9,min_periods=1))
    df_rms = df.groupby('id')[columns].apply(pd.rolling_apply, win_size, lambda x:RMS(x), min_periods=1)
    df_energy = df.groupby('id')[columns].apply(pd.rolling_apply, win_size, lambda x:Energy(x), min_periods=1)
    df_features = pd.concat([df[columns],df_mean,df_std,df_median,df_max,df_min,df_quantile,df_rms,df_energy], axis=1).dropna()
    features = np.array(df_features)
    return features

def extract_features(df, columns, win_size):
    df_mean = pd.rolling_mean(df[columns], win_size, min_periods=1)
    df_std = pd.rolling_std(df[columns], win_size, min_periods=1)
    df_std = df_std.fillna(0)
    df_median = pd.rolling_median(df[columns], win_size, min_periods=1)
    df_min = pd.rolling_min(df[columns], win_size, min_periods=1)
    df_max = pd.rolling_max(df[columns], win_size, min_periods=1)
    df_quantile = pd.rolling_quantile(df[columns],win_size,0.9)
    df_rms = pd.rolling_apply(df[columns],win_size,lambda x:RMS(x))             
    df_energy = pd.rolling_apply(df[columns],win_size,lambda x:Energy(x))                         
    df_features = pd.concat([df[columns],df_mean,df_std,df_median,df_max,df_min,df_quantile,df_rms,df_energy], axis=1).dropna()
    features = np.array(df_features)
    return features
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('args', nargs='+')
    ns = vars(parser.parse_args())
    path = ns['args'][0]
    win_size = int(ns['args'][1])
    columns_temp = ns['args'][2]
    columns = columns_temp.split(',')
    model_path = ns['args'][3]

    #### load data
    #filename = '/data/preventive_maintenance/turbofan_engine.csv'
    #df = pd.read_csv(filename) #load data as pandas dataframe
    ### if multiple files
    files = os.listdir(path)
    df = pd.read_csv(path+files[0])
    for i in range(1,len(files)):
        temp = pd.read_csv(path+files[i])
        df = pd.concat([df,temp]).reset_index(drop=True)
    #### ------------------------------------------------------
    
    #### feature engineering   
    #win_size = 5     
    #columns = ['s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21'] #specify which columns to perform feature engineering
    features = extract_features_group(df, columns, win_size) #if input data include groups
    labels = np.array(df['label'])
    #### ------------------------------------------------------
    
    #### feature normalization
    min_max_scaler = MinMaxScaler()
    features = min_max_scaler.fit_transform(features)
    #### ------------------------------------------------------
    
    #### feature selection
    print('Feature selection...')    
    tree = RandomForestClassifier(n_estimators=200)
    tree = tree.fit(features, labels)
    model = SelectFromModel(tree, prefit=True)
    features_selected = model.transform(features)
    #### ------------------------------------------------------

    ### divide training and testing data
    X_train, X_test, y_train, y_test = model_selection.train_test_split(features_selected, labels, test_size=0.15, random_state=42)
    #### ------------------------------------------------------
    
    #### define model
    score = 'f1'
    ## logistic regression
    clf = LogisticRegression(random_state=123)
    params = {'C': [.01, 0.1, 1.0, 10], 'solver': ['liblinear', 'lbfgs']}
    
    ## random forest
    #clf = RandomForestClassifier(n_estimators=200, max_features='auto', random_state=123)
    #params = {'max_depth': [4, 5, 6, 7, 8]}
    
    ## gradient boosting
    #clf = GradientBoostingClassifier(n_estimators=200, max_features='auto', random_state=123)
    #params = {'max_depth': [4, 5, 6, 7, 8]}
    
    ## svm
    #clf = SVC(kernel='rbf', random_state=123)
    #params = {'C': [1.0]}
    
    ## svm linear
    #clf = LinearSVC(random_state=123)
    #params = {'C': [.001, .01 ,.1 ]}
    
    ## naive bayes
    #clf = GaussianNB()
    #params = {} 
    
    ## neural network
    #clf = MLPClassifier(solver='adam', random_state=123)
    #params = {'alpha': [0.001, 0.01], 'hidden_layer_sizes': [(100,), (100,100)]}
    #### ------------------------------------------------------
    
    #### training and hyperparameter tuning using grid search cross validation
    print('Training...')
    grid_search = model_selection.GridSearchCV(estimator=clf, param_grid=params, cv=5, scoring=score)
    grid_search.fit(X_train, y_train)
    print('\nBest Parameters:\n',grid_search.best_params_)
    means = grid_search.cv_results_['mean_test_score']
    stds = grid_search.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    #### ------------------------------------------------------
        
    #### testing and performance evaluation
    y_pred = grid_search.predict(X_test)
    ## binary classification
    binclass_metrics = {
                        'Accuracy' : metrics.accuracy_score(y_test, y_pred),
                        'Precision' : metrics.precision_score(y_test, y_pred),
                        'Recall' : metrics.recall_score(y_test, y_pred),
                        'F1 Score' : metrics.f1_score(y_test, y_pred),
                       }
    print(binclass_metrics)
    #### ------------------------------------------------------
    
    #### save model
    with open(model_path,"wb") as f:
        pickle.dump([grid_search,win_size,columns,model,min_max_scaler],f)
    #### ------------------------------------------------------
    

