# -*- coding: utf-8 -*-

import pickle
import pandas as pd
import numpy as np
from sklearn import metrics
import argparse

def RMS(data):
    return np.sqrt(np.sum(data ** 2)/len(data))

def Energy(data):
    return np.sum(data ** 2)

def extract_features_group(df, columns, win_size):
    #df_mean = df.groupby('id')[columns].apply(pd.rolling_mean, win_size, min_periods=1)
    df_mean = df.groupby('id')[columns].rolling(window=win_size,min_periods=1,center=False).mean().reset_index().drop(['id','level_1'], axis=1)

    #df_std = df.groupby('id')[columns].apply(pd.rolling_std, win_size, min_periods=1)
    df_std = df.groupby('id')[columns].rolling(window=win_size,min_periods=1,center=False).std().reset_index().drop(['id','level_1'], axis=1)

    df_std = df_std.fillna(0)
    #df_median = df.groupby('id')[columns].apply(pd.rolling_median, win_size, min_periods=1)
    df_median = df.groupby('id')[columns].rolling(window=win_size,min_periods=1,center=False).median().reset_index().drop(['id','level_1'], axis=1)

    #df_min = df.groupby('id')[columns].apply(pd.rolling_min, win_size, min_periods=1)
    df_min = df.groupby('id')[columns].rolling(window=win_size,min_periods=1,center=False).min().reset_index().drop(['id','level_1'], axis=1)

    #df_max = df.groupby('id')[columns].apply(pd.rolling_max, win_size, min_periods=1)
    df_max = df.groupby('id')[columns].rolling(window=win_size,min_periods=1,center=False).max().reset_index().drop(['id','level_1'], axis=1)

    df_quantile = df.groupby('id')[columns].apply(lambda x: pd.rolling_quantile(x,win_size,0.9,min_periods=1))

    df_rms = df.groupby('id')[columns].apply(pd.rolling_apply, win_size, lambda x:RMS(x), min_periods=1)
    #df_rms = df.groupby('id')[columns].rolling(window=win_size,center=False,min_periods=1).apply(func= lambda x:RMS(x))


    df_energy = df.groupby('id')[columns].apply(pd.rolling_apply, win_size, lambda x:Energy(x), min_periods=1)


    df_features = pd.concat([df[columns],df_mean,df_std,df_median,df_max,df_min,df_quantile,df_rms,df_energy], axis=1).dropna()
    features = np.array(df_features)
    return features

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('args', nargs='+')
    ns = vars(parser.parse_args())
    model_path = ns['args'][0]
    filename = ns['args'][1]

    #### load model
    with open(model_path,"rb") as f:
        [clf,win_size,columns,fs_model,min_max_scaler] = pickle.load(f)

    #### load testing data
    df = pd.read_csv(filename) #load data as pandas dataframe

    #### feature engineering
    features = extract_features_group(df, columns, win_size)
    labels = np.array(df['label'])
    #### ------------------------------------------------------

    #### feature normalization
    features = min_max_scaler.fit_transform(features)
    #### ------------------------------------------------------

    #### feature selection
    features_selected = fs_model.transform(features)
    #### ------------------------------------------------------

    y_pred = clf.predict(features_selected)
    ## binary classification
    binclass_metrics = {
                        'Accuracy' : metrics.accuracy_score(labels, y_pred),
                        'Precision' : metrics.precision_score(labels, y_pred),
                        'Recall' : metrics.recall_score(labels, y_pred),
                        'F1 Score' : metrics.f1_score(labels, y_pred),
                       }
    print(binclass_metrics)