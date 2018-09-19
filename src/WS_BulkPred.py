# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 09:47:22 2018

@author: Prodapt-ML Team
"""

from models.WS_Forecast import Forecast
from models.WS_AnomalyDetection import AnomalyDetection
from statsmodels.tsa.arima_model import ARIMA
from datetime import timedelta
import pandas as pd
import pickle
import datetime
import configparser
import os
import numpy as np
import warnings
import logging


def fpHandling(prediction_df, device_list, dev_metric):
    pm_list = []
    for device in device_list:
        tmp = prediction_df[prediction_df[dev_metric]==device]
        index = 0
        flag = 0
        alert_list=[]
        while (flag==0):
            alerts = list(tmp.iloc[index:index+4]["anomaly"])
            if alerts.count(1) > 1:
                fp_alerts = [0,0,0,0]
                fp_alerts[alerts.index(1)] = 1
                alert_list.extend(fp_alerts)
            else:
                alert_list.extend(alerts)
            index=index+4
            if index >= len(prediction_df):
                flag=1
        tmp["anomaly"]=alert_list
        pm_list.append(tmp)
    prediction_df = pd.concat(pm_list)
    return prediction_df

def bulkPred(actual_dir, start_date, end_date, config, loggername): 
    """
         Forecast the pm metrics for 30 devices for the missing hours
         
         Parameters
         ----------
         input_df  : performance data
         start_date : Date and time from when the forecast should start
         end_date : Date and time upto when the forecast should be done

         Returns
         -------
         prediction_df : Dataframe with forecasted values & anomaly prediction    
         
         Examples
         --------
         >> bulkPred(input_df)
    """

    anomaly_dir = config.get('General', 'anomaly.op.dir')
    pqd_dir = config.get('General', 'pqd.dir')
    timestamp = config.get('Parameters', 'timestamp')
    module = config.get('Parameters', 'devices')   
    metric_val = config.get('Parameters', 'metrics')   
    metric_list = metric_val.split(',')
    pred_dir = config.get('General','prediction.dir')
    
    start_date=datetime.datetime.strptime(start_date,'%Y-%m-%d %H:%M:%S')
    end_date=datetime.datetime.strptime(end_date,'%Y-%m-%d %H:%M:%S')

    diff = end_date-start_date
    time_diff = diff.days*24+(diff.seconds//3600)
    
    logger = logging.getLogger(loggername)
    
    input_df = pd.read_csv(actual_dir)
    input_df[timestamp]=pd.to_datetime(input_df[timestamp],infer_datetime_format=True)
    input_df.sort_values(timestamp,inplace=True)
    
    traindata = input_df[(input_df[timestamp] >= (start_date-timedelta(hours=24))) & (input_df[timestamp] < start_date)]
    testdata = input_df[(input_df[timestamp] >= start_date) & (input_df[timestamp] <= end_date)]
    
    with open(pqd_dir, 'rb') as pdq:
        pdq_dict = pickle.load(pdq)    
        
   
       
    device_list = list(input_df[module].unique())
    
    prediction_df=pd.DataFrame(columns=[timestamp,module]+metric_list)
    timelist = list(traindata[timestamp])
    time = [timelist[-1]]
    for t in range(0,time_diff*4):
        time.append(time[-1]+timedelta(minutes=15))
    del(time[0])

    device_dict = {}
    
    for device in device_list:
        temp_dict={}
        temp_df = traindata[traindata[module]==device]
        for metric in metric_list:            
            temp_dict[metric] = temp_df[metric]
        device_dict[device] = temp_dict
                 
    for device in device_list:
        df=pd.DataFrame()
        df[timestamp] = time
        df[module] = [device]*time_diff*4
        for metric in metric_list:
            history = [x for x in device_dict[device][metric]]
            history = history[:96]
            pred=[]
            val=0
            tempdata=testdata[testdata['performance_metrics.module']==device]
            act = list(tempdata[metric])
            for t in range(0,time_diff):
                if device in list(pdq_dict.keys()):
                    p_val,d_val,q_val = pdq_dict[device][metric]
                else:
                    p_val,d_val,q_val = 5,0,0
                while p_val>0:
                    try:
                        res = []
                        model = ARIMA(history, order=(p_val,d_val,q_val))
                        model_fit = model.fit(disp=0)
                        res = model_fit.forecast(steps=4)[0]
                        for x in range(0,4):
                            history.append(act[val])
                            val=val+1
                        history = history[-96:]
                        p_val=0
                    except:
                        p_val=p_val-1
                        pass
                if np.isnan(res).any() or len(res)==0:
                    res = history[-4:]
                    pred.extend(res)
                else:
                    pred.extend(res)
            df[metric] = pred
        prediction_df = prediction_df.append(df,ignore_index=True)
        logger.info("Forecast completed for device : " + str(device))
        
    ad_obj = AnomalyDetection()
    fc = Forecast('forecast',config)
    prediction_df = prediction_df.apply(fc.anomaly, args=(ad_obj,), axis=1)
    
    prediction_df = fpHandling(prediction_df, device_list, module)
    old_forecast = pd.read_csv(pred_dir)
    old_forecast = old_forecast.append(prediction_df, ignore_index=True)
    alarm = prediction_df[prediction_df["anomaly"]==1][[timestamp,module, "anomaly"]]
    old_alarm = pd.read_csv(anomaly_dir)
    old_alarm = old_alarm.append(alarm,ignore_index=True)

    forecast_list = []
    alarm_list = []
    
    for device in list(old_forecast[module].unique()):
        tmp1 = old_forecast[old_forecast[module]==device]
        tmp1.drop_duplicates(subset=[timestamp], inplace=True)
        forecast_list.append(tmp1)
        
    for device in list(old_alarm[module].unique()):
        tmp2 = old_alarm[old_alarm[module]==device]
        tmp2.drop_duplicates(subset=[timestamp], inplace=True)
        alarm_list.append(tmp2)
    
    new_forecast = pd.concat(forecast_list) 
    new_alarm = pd.concat(alarm_list)
    
    new_forecast.to_csv(pred_dir,index=False)           
    new_alarm.to_csv(anomaly_dir, index=False)

    return prediction_df

if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    
    config_dir = os.getcwd() + '/WS_configfile.properties'
    config = configparser.RawConfigParser()
    config.read(config_dir)
    
    try:
        loggername = config.get('General', 'bulk.predict.log')
        actual_dir = config.get('BulkPredict', 'actual.data.dir')
        start_date = config.get('BulkPredict', 'start.date')
        end_date = config.get('BulkPredict', 'end.date')
        
        logging.basicConfig(level=logging.INFO, filename=loggername, filemode="a+",
                            format="%(asctime)-15s %(levelname)-8s %(message)s")
        
        logging.info("####################################################")
        logging.info("                 Started ...                          ")
        logging.info("####################################################\n")
        
        logging.info("Started forecasting the PM data")
        prediction_df = bulkPred(actual_dir,start_date,end_date,config,loggername)
        logging.info("Forecast is completed for all devices")
        
        logging.info("####################################################")
        logging.info("                 Completed                          ")
        logging.info("####################################################\n")
        
    except Exception as e:
        logging.error(str(e))
        
