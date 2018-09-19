#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 11:07:48 2018

@author: Prodapt-ML Team
"""

from data.WS_DataPreparation import Utility
from models.WS_Forecast import Forecast
from datetime import datetime
import pandas as pd
import configparser
import warnings
import logging
import os


def dataReadAndConversion(raw_dir, converted_dir, util):
    """
         Read the pm data fetched from hive and convert to required format

         Parameters
         ----------
         raw_dir        : Directory path where the raw PM data will be stored
         converted_dir  : Directory path where the converted data will be stored

         Returns
         -------
         converted_pm : converted pm data

         Examples
         --------
         >> dataReadAndConversion("/home/iotadmin")
    """
    pm_data = pd.read_csv(raw_dir, low_memory=False)
    converted_pm = util.dataConversion(pm_data, converted_dir, "forecast")
    return converted_pm


def dataPreparation(converted_pm, metric_list):
    """
         Addressing any missing values using mean

         Parameters
         ----------
         converted_pm : converted pm data

         Returns
         -------
         converted_pm : Prepared pm data

         Examples
         --------
         >> dataPreparation(converted_pm)
    """   
    for metric in metric_list:
        converted_pm[metric] = converted_pm[metric].fillna(converted_pm[metric].mean())
    return converted_pm


def machineLearning(train_data, pred_dir, pqd_dir, logger_name,config):
    """
         Makes the forecast and anomaly prediction for one hour - 30 devices

         Parameters
         ----------
         train_data  : Prepared pm data
         pred_dir : Storage directory for the predicted value
         pqd_dir : Storage directory for the arima parameters

         Returns
         -------
         prediction_df : Dataframe with forecasted value & anomaly prediction

         Examples
         --------
         >> machineLearning(train_data)
    """
    fc = Forecast(logger_name,config)
    prediction_df = fc.arimaForecast(train_data, pred_dir, pqd_dir)
    return prediction_df


def dataAvailabilityCheck(hive_data, converted_data, loggername, timestamp, col_names):
    data_available = False
    logger = logging.getLogger(loggername)

    if os.path.isfile(hive_data) & (len(open(hive_data).readlines()) > 2):
        data_available = True
        if os.path.isfile(converted_data):
                hive_pm_data = pd.read_csv(hive_data, low_memory=False, infer_datetime_format=True)
                hive_pm_data.sort_values(timestamp,inplace=True)
                hive_lasttime = hive_pm_data.tail(1)[timestamp].values
                
                conv_pm_data = pd.read_csv(converted_data, low_memory=False, infer_datetime_format=True, names=col_names)
                conv_pm_data.sort_values(timestamp,inplace=True)
                conv_lasttime = conv_pm_data.tail(1)[timestamp].values
    
                fmt = '%Y-%m-%d %H:%M:%S.%f'
                tstamp1 = datetime.strptime(hive_lasttime[0], fmt)
                tstamp2 = datetime.strptime(conv_lasttime[0], fmt)
 

                if tstamp1 > tstamp2:
                    data_available = True
                    logger.info("PM Data Available .. ")
                else:
                    data_available = False
                    logger.info("Last Data Available on:" + str(hive_lasttime[0]))
                    print("Latest data unavailable - Check log for further details")
    return data_available
        

# Assumption - PM data (CHO6Y/OCHCTP) will be fetched from hive & stored as a 
# CSV in a tmp folder
######################### Execute the following ##############################

if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    
    config_dir = os.getcwd() + '/WS_configfile.properties'
    config = configparser.RawConfigParser()
    config.read(config_dir)
    
    hive_dir = config.get('General', 'raw.pm.dir')
    converted_dir = config.get('General','converted.pm.dir')
    mloutput_dir = config.get('General','prediction.dir')
    arima_metric = config.get('General','pqd.dir')
    loggername =   config.get('General', 'forecast.log')
    timestamp = config.get('Parameters', 'timestamp')
    module = config.get('Parameters', 'devices')   
    metric_val = config.get('Parameters', 'metrics')   
    metric_list = metric_val.split(',')
    col_names=[timestamp, module]+metric_list
    
    logging.basicConfig(level=logging.INFO, filename=loggername, filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    
    logging.info("####################################################")
    logging.info("                 Started ...                          ")
    logging.info("####################################################\n")
                         
    logging.info("Started feteching PM data from Hive")  
    util = Utility(loggername,config)    
    hive_df = util.hiveReadData("forecast")
#        sendMail('Hive data not read and process has terminated')
 
   
    if(dataAvailabilityCheck(hive_dir, converted_dir, loggername, timestamp, col_names)):
        
        flag=0
        converted_pm = dataReadAndConversion(hive_dir, converted_dir, util)
        if len(converted_pm)!=0:
            logging.info("PM data from hive converted & successfully written")
        else:
            flag=1
#            util.sendMail('Data Conversion has not been completed and process has terminated')
        
        train_data = dataPreparation(converted_pm, metric_list)
        if len(train_data)!=0:
            logging.info("Data preparation successful")
        else:
            flag=1
 #           util.sendMail('Data Preparation has not been completed and process has terminated')
        
        forecast_df = machineLearning(train_data,mloutput_dir,arima_metric,loggername,config)
        if len(forecast_df)!=0:
            logging.info("ML Process Completed For The Hour\n")
        else:
            flag=1
  #          util.sendMail('Forecast and anomaly prediction has not been completed and process has terminated')
        
        if flag == 0:
            print("\n\nProcess Successfully Completed\n\n")
    
            logging.info("####################################################")
            logging.info("                Completed ...                         ")
            logging.info("####################################################\n")
    else:

        print("\n\nProcess Terminated\n\n")

        logging.info("####################################################")
        logging.info("    Process terminated - Data Unavailable ...       ")
        logging.info("####################################################\n")
