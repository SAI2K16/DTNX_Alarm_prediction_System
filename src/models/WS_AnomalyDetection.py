#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 13:14:25 2018

@author: Prodapt-ML Team
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import pandas as pd
import configparser
import pickle 
import os

class AnomalyDetection():

    def __init__(self):    
        config_dir = os.getcwd() + '/WS_configfile.properties'
        self.config = configparser.RawConfigParser()
        self.config.read(config_dir)

        with open(self.config.get('ModelBuilding', 'ad.knn.dir'), 'rb') as handle:
            self.ad_model = pickle.load(handle)
        
    def knnAlgo_predict(self, input_df):
        """
             Load the trained anomaly model and make prediction
    
             Parameters
             ----------
             input_df  : Forecasted PM metric for a device
    
             Returns
             -------
             anomaly[0] : 1/0 - Anomaly/Non-Anomaly
    
             Examples
             --------
             >> knnAlgo_predict(input_df)
        """
        

        anomaly = self.ad_model.predict(input_df[['BerPreFecMax','PhaseCorrectionAve','PmdMin','Qmin','SoPmdAve']].values.reshape(1,-1))

        return anomaly[0]
