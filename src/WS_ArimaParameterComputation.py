# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 10:36:16 2018

@author: Prodapt-ML Team
"""

from models.WS_Forecast import Forecast
import pandas as pd
import os
import configparser
import warnings
import logging
import pickle
import multiprocessing

def calcPDQ(config):
          
    timestamp = config.get('Parameters', 'timestamp')
    module = config.get('Parameters', 'devices')   
    metric_val = config.get('Parameters', 'metrics')   
    metric_list = metric_val.split(',')
    col_names=[timestamp, module]+metric_list
    
    logger_name = config.get('General', 'arima.log')
    
    logger = logging.getLogger(logger_name)
    
    converted_pm = pd.read_csv(config.get('General','converted.pm.dir'), names=col_names)
    
    device_list = list(converted_pm[module].unique())
    device_dict = {}    
    pdq_dict = {}
    for device in device_list:
        temp_dict={}
        temp_df = converted_pm[converted_pm[module]==device]
        for metric in metric_list:            
            temp_dict[metric] = temp_df[metric]
        device_dict[device] = temp_dict

    with open(config.get('General','pqd.dir'), 'rb') as pdq:
            pdq_dict = pickle.load(pdq) 
                   
    logger.info("Started computing the ARIMA parameters")

    threads = []

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    fc = Forecast(logger_name,config)
    for dev, value in device_dict.items():
        tmp_dict = {}
        tmp_dict[dev]= value      
        p = multiprocessing.Process(target=fc.selectPDQ, args=(tmp_dict, config.get('General','pqd.dir'),metric_list, pdq_dict,return_dict,))
        threads.append(p)

    for x in threads:
     x.start()
     
    for x in threads:
     x.join()
     
    pqd = dict(return_dict)
    
    with open(config.get('General','pqd.dir'), 'wb') as pdq:
            pickle.dump(pqd, pdq, protocol=pickle.HIGHEST_PROTOCOL)
        
    logger.info("Parametrs have been computed for all devices successfully")
    

if __name__ == "__main__":
    
    warnings.filterwarnings("ignore")
    config_dir = os.getcwd() + '/WS_configfile.properties'
    config = configparser.RawConfigParser()
    config.read(config_dir)
    
    
    logging.basicConfig(level=logging.DEBUG, filename=config.get('General', 'arima.log'), filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    calcPDQ(config)
    
