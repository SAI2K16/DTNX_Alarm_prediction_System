# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 11:57:23 2018

@author:  Prodapt-ML Team
"""
from models.WS_AnomalyDetection import AnomalyDetection
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from datetime import timedelta
import pandas as pd
import numpy as np
import pickle
import logging


class Forecast():
    
    def __init__(self,loggername,config):
        self.logger = logging.getLogger(loggername)
        self.config = config

    
    def anomaly(self,x, ad_obj):
        """
             Anmaly prediction using KNN algorithm
    
             Parameters
             ----------
             x  : performance data
    
             Returns
             -------
             x : performance data
    
             Examples
             --------
             >> anomaly(x)
        """
        try:
            x["anomaly"] = ad_obj.knnAlgo_predict(x)
            return x
        except Exception as e:
            self.logger.error(str(e)+"-Occurred in anomaly detection")
            return (pd.DataFrame())
    
    def evaluate_arima_model(self,X, arima_order):
        train_size = int(len(X) * 0.96)
        train, test = X[0:train_size], X[train_size:]
        history = [x for x in train]
        predictions = list()
        for t in range(len(test)):
            model = ARIMA(history, order=arima_order)
            model_fit = model.fit(disp=0)
            yhat = model_fit.forecast()[0]
            predictions.append(yhat)
            history.append(test[t])
        error = mean_squared_error(test, predictions)
        return error
    
    def evaluate_models(self,dataset, p_values, d_values, q_values):
        dataset = dataset.astype('float32')
        best_score, best_pdq = float("inf"), None
        for p_val in p_values:
            for d_val in d_values:
                for q_val in q_values:
                    order = (p_val,d_val,q_val)
                    try:
                        mse = self.evaluate_arima_model(dataset, order)
                        if mse < best_score:
                            best_score, best_pdq = mse, order
                    except:
                        continue
        return(best_pdq)
    
    def selectPDQ(self,dev_dict,pqd_dir,metric_list, pdq_dict,return_dict):
        """
             Grid search to find p,q,d values
             
             Parameters
             ----------
             dev_dict  : performance data for each metric
             pqd_dir : Storage directory for the arima parameters
    
             Returns
             -------
             NA    
             
             Examples
             --------
             >> selectPDQ(dev_dict)
        """
        p_values = [4,5,6]
        d_values = [0,1]
        q_values = [0,1]
               
        for device,device_val in dev_dict.items():
            pdq_temp = {}
            for metric in metric_list:
                series = dev_dict[device][metric]
                pdq_temp[metric] = self.evaluate_models(series.values,p_values,d_values,q_values)
                if type(pdq_temp[metric])!=tuple:
                    pdq_temp[metric]=pdq_dict[device][metric]
            pdq_dict[device] = pdq_temp
            return_dict[device]=pdq_temp
            self.logger.info(f'Parameters have been calculated for the device {device}')
                    
     
                    
    def fpHandling(self,prediction_df, device_list, dev_metric):
        try:
            pm_list = []
            for device in device_list:
                tmp = prediction_df[prediction_df[dev_metric]==device]
                alerts = list(tmp["anomaly"])
                if alerts.count(1) > 1:
                    fp_alerts = [0,0,0,0]
                    fp_alerts[alerts.index(1)] = 1
                    tmp["anomaly"]=fp_alerts
                pm_list.append(tmp)
            prediction_df = pd.concat(pm_list)
            return prediction_df
        except Exception as e:
            self.logger.error(str(e)+"-Occurred in false positive handling")
            return (pd.DataFrame())
    
    def arimaForecast(self,input_df, pred_dir, pqd_dir):
        """
             Forecast the pm metrics for 30 devices - 1 hour
             
             Parameters
             ----------
             input_df  : performance data
             pred_dir : Storage directory for the predicted value
             pqd_dir : Storage directory for the arima parameters
    
             Returns
             -------
             prediction_df : Dataframe with forecasted value & anomaly prediction    
             
             Examples
             --------
             >> arimaForecast(input_df)
        """
        try:
            with open(pqd_dir, 'rb') as pdq:
                pdq_dict = pickle.load(pdq)            
    
            # For corresponding loggers        
            
            anomaly_dir = self.config.get('General', 'anomaly.op.dir')
            grid_search = self.config.get('General', 'grid.search')
            timestamp = self.config.get('Parameters', 'timestamp')
            module = self.config.get('Parameters', 'devices')   
            metric_val = self.config.get('Parameters', 'metrics')   
            metric_list = metric_val.split(',')
                
            input_df[timestamp]=pd.to_datetime(input_df[timestamp],infer_datetime_format=True)
            input_df.sort_values(timestamp,inplace=True)
               
            device_list = list(input_df[module].unique())
            
            prediction_df=pd.DataFrame(columns=[timestamp,module]+metric_list)
            timelist = list(input_df[timestamp])
            time = [timelist[-1]]
            for t in range(0,4):
                time.append(time[-1]+timedelta(minutes=15))
            del(time[0])
            
            device_dict = {}
            
            for device in device_list:
                temp_dict={}
                temp_df = input_df[input_df[module]==device]
                for metric in metric_list:            
                    temp_dict[metric] = temp_df[metric]
                device_dict[device] = temp_dict
            
            for device,device_val in device_dict.items():
                df=pd.DataFrame()
                df[timestamp] = time
                df[module] = [device]*4
                for metric in metric_list:
                    history = [x for x in device_dict[device][metric]]
                    history = history[-96:]
                    prediction=[]
                    if device in list(pdq_dict.keys()):
                        p_val,d_val,q_val = pdq_dict[device][metric]
                    else:
                        p_val,d_val,q_val = 5,0,0
                    while p_val > 0:
                            try:
                                model = ARIMA(history, order=(p_val,d_val,q_val))
                                model_fit = model.fit(disp=0)
                                result = model_fit.forecast(steps=4)[0]
                                prediction.extend(result)
                                p_val=0
                            except:                    
                                p_val=p_val-1
                                if p_val==0:                               
                                    prediction = history[-4:]
                                pass
                    if np.isnan(prediction).any():
                        prediction = history[-4:]
               
                    df[metric] = prediction
                self.logger.info("Forecast completed for device : " + str(device))
                prediction_df = prediction_df.append(df, ignore_index=True)       
    
            ad_obj = AnomalyDetection()
            prediction_df = prediction_df.apply(self.anomaly, args=(ad_obj,), axis=1)
        
            prediction_df = self.fpHandling(prediction_df, device_list, module)
            if len(prediction_df)!=0:
                self.logger.info("Alarm prediction completed successfully .. ")
            prediction_df.to_csv(pred_dir,index=False, mode='a', header=False)
            
            alarm = prediction_df[prediction_df["anomaly"]==1][[timestamp,module, "anomaly"]]
            
            alarm.to_csv(anomaly_dir, mode='a', header=False, index=False)
            
            if grid_search=="True":
                self.logger.info("Started computing the ARIMA parameters")
                self.selectPDQ(device_dict, pqd_dir, metric_list, pdq_dict)
                self.logger.info("Parametrs have been computed for all devices successfully")
            
            return prediction_df
        
        except Exception as e:
            self.logger.error(str(e)+"-Occurred in Forecast and anomaly detection script")
            return (pd.DataFrame())
            
