import pandas as pd
from pyhive import hive
import thrift
import csv
import configparser
import logging
import subprocess

class Utility:
    
    def __init__(self,loggername,config):
        self.logger = logging.getLogger(loggername)
        self.config = config

    def dataConversion(self,pm_data, dir_path, flag):
        
        try:
        
            timestamp = self.config.get('Parameters', 'timestamp')
            section_name = self.config.get('Parameters', 'section')
            module_name = self.config.get('Parameters', 'devices')
            val = self.config.get('Parameters', 'val')
            measure = self.config.get('Parameters', 'measure')
            common = self.config.get('Parameters', 'device.types')
            parent_columns = self.config.get('Parameters', 'parent.columns')
            actual = self.config.get('General','actual.dir')
            metric_val = self.config.get('Parameters', 'metrics')  
            metric_list = metric_val.split(',')   
            
            sections_list = list(pm_data[section_name].unique())   
            for section in sections_list:
                section_df = pm_data[pm_data[section_name]==section]
                modules_list = common.split(',')            
                converted_df = pd.DataFrame()
                for module in modules_list:
                    old_df = section_df[section_df[module_name]==module]
                    temp_df = old_df.pivot_table(val, [timestamp], measure) 
                    old_df.drop_duplicates([timestamp], inplace=True) 
                    old_df.set_index(old_df[timestamp],inplace = True)
                    new_df = old_df[parent_columns.split(',')].copy()
                    new_df = new_df.join(temp_df)
                    new_df.reset_index(inplace=True,drop=True)
                    converted_df = converted_df.append(new_df,ignore_index=True)
                converted_df = converted_df[[timestamp, module_name] + metric_list]   
                if flag == "modelbuilding":
                    converted_df.to_csv(dir_path,index = False,mode='a',header=False)
                else:
                    converted_df.to_csv(dir_path,index = False,header=False)
                    converted_df.to_csv(actual,index = False,mode='a',header=False)
                return converted_df
            
        except Exception as e:
            self.logger.error(str(e)+"-Occurred in Data Conversion")
            return (pd.DataFrame())


    def sendMail(self,message):
        try:
            from_addr = self.config.get('Mail','from.address')
            password = self.config.get('Mail','password')
            to_addr = self.config.get('Mail','to.address')
            subprocess.call(['bash','sendmail.sh',message,from_addr,password,to_addr])
        except Exception as e:
            self.logger.error(str(e)+"-Occurred while sending mail")
            

    def hiveReadData(self,flag):
        try:
            conn = hive.Connection(host=self.config.get('Hive','hostname'), port=self.config.get('Hive','port'), auth=self.config.get('Hive','auth'), username=self.config.get('Hive','user') ,password=self.config.get('Hive','pass'))
            if flag=="forecast":
                df = pd.read_sql("select * from dnapm.performance_metrics where nodename='CHCGILDTO6Y' and section='h_OCHCTP' and TO_UNIX_TIMESTAMP(ts) > (UNIX_TIMESTAMP() - 24*60*60)",conn).to_csv(self.config.get('Hive','test.dir'), index=False)
            else:
                df = pd.read_sql("select * from dnapm.performance_metrics where nodename='CHCGILDTO6Y' and section='h_OCHCTP' and TO_UNIX_TIMESTAMP(ts) > (UNIX_TIMESTAMP() - 24*60*60*180)",conn).to_csv(self.config.get('Hive','hive.model.dir'), index=False)
            return df
        except Exception as e:
            self.logger.error(str(e)+"-Occurred when reading hive data")
            return (pd.DataFrame())
        

