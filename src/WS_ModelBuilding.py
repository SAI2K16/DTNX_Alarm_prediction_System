#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 09:06:00 2018

@author: Prodapt-ML Team

"""

from data.WS_DataPreparation import Utility
from WS_ParentScript import dataPreparation
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import configparser
import warnings
import logging
import pickle
import os


def alertTag(pm, thd_down, alert_metric):

    if(pm[alert_metric] < thd_down):
        pm['tag'] = 1
    else:
        pm['tag'] = 0
    return pm

def knnAlgo_build(input_df, metric_list, model_dir):
   
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(input_df[metric_list], input_df[["tag"]]) 
    
    model_pickle = open('/app/env/AlarmPrediction/models/new_model.pkl', 'wb')
    pickle.dump(neigh, model_pickle)
    model_pickle.close()


if __name__ == "__main__":

    warnings.filterwarnings("ignore")    
   
    config_dir = os.getcwd() + '/WS_configfile.properties'
    config = configparser.RawConfigParser()
    config.read(config_dir)
    
    loggername = config.get('General', 'model.build.log')
    logging.basicConfig(level=logging.INFO, filename=loggername, filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    
    
    logging.info("####################################################")
    logging.info("                 Started ...                          ")
    logging.info("####################################################\n")

    logging.info("Started feteching PM data from Hive")
    util = Utility(loggername,config)
    util.hiveReadData("modelbuilding")
    logging.info("Retrived PM data from Hive")
    
    print('Hive data fetched')                 
    tmp_cols = config.get('ModelBuilding', 'hive.columns')   
    hive_cols = tmp_cols.split(',')  
    metric_val = config.get('Parameters', 'metrics')   
    metric_list = metric_val.split(',')    
    hive_dir = config.get('Hive', 'hive.model.dir')
    convertedpm_dir = config.get('ModelBuilding', 'converted.pm.dir')
    alert_tag = config.get('Parameters', 'alert.tagging.metric')
    timestamp = config.get('Parameters', 'timestamp')
    module = config.get('Parameters', 'devices')
    model_dir = config.get('ModelBuilding', 'ad.knn.dir')
          
    skip = 0    
    nrow = 1000000    
        
    while(1):    
        df = pd.read_csv(config.get('ModelBuilding', 'raw.pm.dir'),header=0,skiprows=range(1,skip),nrows=nrow,index_col=False)
        skip = skip+nrow
    
        if(len(df)==0):            
            break
        
        
        converted_pm = util.dataConversion(df, convertedpm_dir, "modelbuilding")
        logging.info("Cycle completed " + str(skip))
   
    print('Data conversion completed')
    logging.info("PM data from hive converted & successfully written")
    
    converted_pm = pd.read_csv(convertedpm_dir, low_memory=False, names=[timestamp, module]+metric_list)
    train_data = dataPreparation(converted_pm,metric_list)
    logging.info("Data preparation successful")

    thd_down = np.mean(train_data[alert_tag]) - (3 * np.std(train_data[alert_tag]))
    train_data=train_data.apply(alertTag, args = (thd_down,alert_tag, ), axis=1)
    logging.info("Data tagging successful")
    
    knnAlgo_build(train_data, metric_list, model_dir)
    print('Model building completed')
    logging.info("Anomaly model building successful\n")
    logging.info("####################################################")
    logging.info("                Completed ...                         ")
    logging.info("####################################################\n")

