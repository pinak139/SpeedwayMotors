
# Install Libraries
# V_2_0 : Added reading directly from SQL view to pandas
#!pip install xgboost
#!pip install lightgbm

# Import Libraries

import pandas as pd
import numpy as np
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,HistGradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.linear_model import LinearRegression,Lasso
from datetime import date,datetime
import pyodbc
import sqlalchemy as sal
from sqlalchemy import create_engine
import urllib
import sqlalchemy

def getData(query):
    server = 'dw.speedway2.com'
    username = 'pdutta'
    password = 'Trance140'
    DB = 'DW'
    driver = 'ODBC Driver 17 for SQL Server'
    params = f'DRIVER={driver};SERVER={server};PORT=1433;DATABASE={DB};UID={username};PWD={password}'
    db_params = urllib.parse.quote_plus(params)
    engine = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect={}".format(db_params))
    df = pd.read_sql_query(query,engine)
    return df



 # Function for pre-processing lead time actuals for training
# Generates Random numbers  for different distributions for  monte carlo simulation using mean, median and 90th percentile

def preprocess(df):
    lead_time = df
    lead_time_sku = lead_time[['ixSKU','ActualLeadTime']]
    lead_time_ven = lead_time[['ixVendor','ActualLeadTime']]



    def poisson_mean(row):
        arr = np.random.poisson(row[1],1000)
        return np.mean(arr)

    def poisson_median(row):
        arr = np.random.poisson(row[1],1000)
        return np.median(arr)

    def poisson_90(row):
        arr = np.random.poisson(row[1],1000)
        return np.percentile(arr,90)

    def tri_mean(row):
        if row[1] == row[3]:
            arr = np.random.triangular(row[2],row[1],row[3]+1,1000)
        else:
            arr = np.random.triangular(row[2],row[1],row[3]+1,1000)

        return np.mean(arr)

    def tri_median(row):
        if row[1] == row[3]:
            arr = np.random.triangular(row[2],row[1],row[3]+1,1000)
        else:
            arr = np.random.triangular(row[2],row[1],row[3]+1,1000)
        return np.median(arr)


    def tri_90(row):
        if row[1] == row[3]:
            arr = np.random.triangular(row[2],row[1],row[3]+1,1000)
        else:
            arr = np.random.triangular(row[2],row[1],row[3]+1,1000)
        return np.percentile(arr,90)


    # Comnputing SKU Lead Time Estimates

    lt_med_sku = lead_time_sku.groupby(['ixSKU']).median().reset_index()
    lt_mean_sku = lead_time_sku.groupby(['ixSKU']).mean().reset_index()
    lt_min_sku= lead_time_sku.groupby(['ixSKU']).min().reset_index()
    lt_max_sku = lead_time_sku.groupby(['ixSKU']).max().reset_index()
    lt_count_sku = lead_time_sku.groupby(['ixSKU']).count().reset_index()

    # Comnputing SKU Lead Time Estimates

    lt_med_ven = lead_time_ven.groupby(['ixVendor']).median().reset_index()
    lt_mean_ven = lead_time_ven.groupby(['ixVendor']).mean().reset_index()
    lt_min_ven= lead_time_ven.groupby(['ixVendor']).min().reset_index()
    lt_max_ven = lead_time_ven.groupby(['ixVendor']).max().reset_index()
    lt_count_ven = lead_time_ven.groupby(['ixVendor']).count().reset_index()



    lt_sku = pd.merge(lt_med_sku,lt_min_sku, on = 'ixSKU', how = 'left')
    lt_sku = pd.merge(lt_sku,lt_max_sku, on = 'ixSKU', how = 'left')
    lt_sku = lt_sku.rename(columns = {'ActualLeadTime_x' : 'mean','ActualLeadTime_y' : 'min','ActualLeadTime' : 'max' })
    lt_sku = pd.merge(lt_sku,lt_count_sku, on = 'ixSKU', how = 'left')
    lt_sku = lt_sku.rename(columns = {'ActualLeadTime' : 'count' })

    lt_ven = pd.merge(lt_med_ven,lt_min_ven, on = 'ixVendor', how = 'left')
    lt_ven = pd.merge(lt_ven,lt_max_ven, on = 'ixVendor', how = 'left')
    lt_ven = lt_ven.rename(columns = {'ActualLeadTime_x' : 'mean','ActualLeadTime_y' : 'min','ActualLeadTime' : 'max' })
    lt_ven = pd.merge(lt_ven,lt_count_ven, on = 'ixVendor', how = 'left')
    lt_ven = lt_ven.rename(columns = {'ActualLeadTime' : 'count' })

    lt_sku['poisson_mean_sku'] = lt_sku.apply(poisson_mean,axis = 1)
    lt_sku['poisson_median_sku'] = lt_sku.apply(poisson_median,axis = 1)
    lt_sku['poisson_90_sku'] = lt_sku.apply(poisson_90,axis = 1)
    lt_sku['tri_mean_sku'] = lt_sku.apply(tri_mean,axis = 1)
    lt_sku['tri_median_sku'] = lt_sku.apply(tri_median,axis = 1)
    lt_sku['tri_90_sku'] = lt_sku.apply(tri_90,axis = 1)

    lt_ven['poisson_mean_ven'] = lt_ven.apply(poisson_mean,axis = 1)
    lt_ven['poisson_median_ven'] = lt_ven.apply(poisson_median,axis = 1)
    lt_ven['poisson_90_ven'] = lt_ven.apply(poisson_90,axis = 1)
    lt_ven['tri_mean_ven'] = lt_ven.apply(tri_mean,axis = 1)
    lt_ven['tri_median_ven'] = lt_ven.apply(tri_median,axis = 1)
    lt_ven['tri_90_ven'] = lt_ven.apply(tri_90,axis = 1)

    stats = lead_time[['ixPO','ixVendor','ixSKU','ExpectedLeadTime','ActualLeadTime']]
    stats = pd.merge(stats,lt_sku, on = 'ixSKU', how = 'left')
    stats = pd.merge(stats,lt_ven, on = 'ixVendor', how = 'left')

    stats = stats.dropna()

    stats = stats[stats['ActualLeadTime']> 0]

    return stats

# LEad Time Forecasting Function

def estimateLeadTimes(df):
    stats = df
    cols_x = [
           'ExpectedLeadTime',  'mean_x',
           'min_x', 'max_x', 'count_x', 'poisson_mean_sku', 'poisson_median_sku',
           'poisson_90_sku', 'tri_mean_sku', 'tri_median_sku', 'tri_90_sku',
           'mean_y', 'min_y', 'max_y', 'count_y', 'poisson_mean_ven',
           'poisson_median_ven', 'poisson_90_ven', 'tri_mean_ven',
           'tri_median_ven', 'tri_90_ven']

    X = stats[cols_x].values
    y = stats['ActualLeadTime'].values

    forest = RandomForestRegressor(n_estimators = 100,criterion = 'squared_error',random_state = 1, n_jobs = -1)
    ada = AdaBoostRegressor(n_estimators = 100,random_state = 1)
    hgb = HistGradientBoostingRegressor(random_state = 1)
    xgb = XGBRegressor(random_state = 1)
    lgb = LGBMRegressor(random_state = 1)
    lr = LinearRegression()
    lasso = Lasso(alpha = 1.0)


    lr.fit(X,y)
    y_pred_lr = lr.predict(X)
    print ('LR Done')
    lgb.fit(X,y)
    y_pred_lgb = lgb.predict(X)
    print ('LGB Done')
    xgb.fit(X,y)
    y_pred_xgb = xgb.predict(X)
    print ('XGB Done')
    ada.fit(X,y)
    y_pred_ada = ada.predict(X)
    print ('Ada Done')
    hgb.fit(X,y)
    y_pred_hgb = hgb.predict(X)
    print ('HGB Done')
    forest.fit(X,y)
    y_pred_forest = forest.predict(X)
    print ('RF Done')
    lasso.fit(X,y)
    y_pred_lasso = lasso.predict(X)
    print ('Lasso Done')

    stats['pred_lasso'] = y_pred_lasso
    stats['pred_rforest'] = y_pred_forest
    stats['pred_hgb'] = y_pred_hgb
    stats['pred_ada'] = y_pred_ada
    stats['pred_xgb'] = y_pred_xgb
    stats['pred_lgb'] = y_pred_lgb
    stats['pred_lr'] = y_pred_lr
    lead_time_preds = stats[['ixPO','ixVendor','ixSKU','ExpectedLeadTime','ActualLeadTime','pred_rforest']]

    lead_time_summ_mean = lead_time_preds[['ixSKU','pred_rforest']].groupby('ixSKU').mean().reset_index()
    lead_time_summ_min = lead_time_preds[['ixSKU','pred_rforest']].groupby('ixSKU').min().reset_index()
    lead_time_summ_max = lead_time_preds[['ixSKU','pred_rforest']].groupby('ixSKU').max().reset_index()
    temp = pd.merge(lead_time_summ_min,lead_time_summ_mean,on = 'ixSKU')
    temp = pd.merge(temp,lead_time_summ_max,on = 'ixSKU')
    temp = temp.rename(columns = {'pred_rforest_x':'min','pred_rforest_y': 'expected', 'pred_rforest':'max'})
    temp = temp.round(0)
    temp['SafetyDaysofSupply'] = temp['max'] - temp['expected']
    temp['Prediction_Date'] = date.today()
    lead_time_forecast = temp
    lead_time_forecast['x'] = lead_time_forecast['Prediction_Date'].astype(str)
    lead_time_forecast[['Year','Month','Date']] = lead_time_forecast['x'].str.split('-',expand = True)
    lead_time_forecast['Prediction_Beg_Month'] = lead_time_forecast['Year'] + '-' + lead_time_forecast['Month'].astype(str) + '-'+ '01'
    lead_time_forecast['Prediction_Beg_Month'] = pd.to_datetime(lead_time_forecast['Prediction_Beg_Month'])
    lead_time_forecast['Prediction_Date'] = pd.to_datetime(lead_time_forecast['Prediction_Date'])
    lead_time_forecast['Minimum'] = lead_time_forecast['min']
    lead_time_forecast['Expected'] = lead_time_forecast['expected']
    lead_time_forecast['Maximum'] = lead_time_forecast['max']
    lead_time_forecast = lead_time_forecast[['ixSKU','Minimum','Expected','Maximum','Prediction_Date','Prediction_Beg_Month']]

    return lead_time_forecast

# Compute Error Metrics

def computeStats(df):
    stats = df
    stats['error_lasso'] = np.abs(stats['ActualLeadTime'] - stats['pred_lasso'])/stats['ActualLeadTime']
    stats['error_rf'] = np.abs(stats['ActualLeadTime'] - stats['pred_rforest'])/stats['ActualLeadTime']
    stats['error_hgb'] = np.abs(stats['ActualLeadTime'] - stats['pred_hgb'])/stats['ActualLeadTime']
    stats['error_ada'] = np.abs(stats['ActualLeadTime'] - stats['pred_ada'])/stats['ActualLeadTime']
    stats['error_xgb'] = np.abs(stats['ActualLeadTime'] - stats['pred_xgb'])/stats['ActualLeadTime']
    stats['error_lgb'] = np.abs(stats['ActualLeadTime'] - stats['pred_lgb'])/stats['ActualLeadTime']
    stats['error_lr'] = np.abs(stats['ActualLeadTime'] - stats['pred_lr'])/stats['ActualLeadTime']
    mape_lasso = np.mean(stats['error_lasso'] )
    mape_rf = np.mean(stats['error_rf'] )
    mape_hgb = np.mean(stats['error_hgb'] )
    mape_ada = np.mean(stats['error_ada'])
    mape_xgb = np.mean(stats['error_xgb'])
    mape_lgb = np.mean(stats['error_lgb'])
    mape_lr = np.mean(stats['error_lr'])
    algo = ['rf','lasso','hgb','ada','xgb','lgb','lr']
    mape = [np.mean(stats['error_rf'] ),np.mean(stats['error_lasso'] ),np.mean(stats['error_hgb'] ),np.mean(stats['error_ada']),
                np.mean(stats['error_xgb']),np.mean(stats['error_lgb']),np.mean(stats['error_lr'])]
    error = pd.DataFrame(list(zip(algo,mape)),columns = ['algo','mape'])

    return error


 
def write2table(df,table):
    df.to_sql(name = table , con = engine, if_exists = 'append')
    print ('Complete')


# Main Body
query = 'select * from dbo.Lead_Time_Forecast_Data_Pipeline'
lead_time = getData(query)
stats = preprocess(lead_time)
preds = estimateLeadTimes(stats)         
