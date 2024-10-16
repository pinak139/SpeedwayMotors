
Open In Colab
In [ ]:	

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 20:32:22 2021

@author: Pinak
"""

#from neuralprophet import NeuralProphet
from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoNHITS,AutoNBEATS
from neuralforecast.models import NBEATS,NHITS, TimesNet,LSTM
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, SeasonalExponentialSmoothingOptimized,
import os
import pandas as pd
from neuralforecast.tsdataset import TimeSeriesDataset
from ray.tune.search.hyperopt import HyperOptSearch
from neuralforecast.losses.pytorch import MAE
from ray import tune
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
#import matplotlib.pyplot as plt
#plt.rcParams["axes.grid"]=True


sales = pd.read_csv('/Users/pinakdutta/Documents/SupplyChainOptimization/forecasting/sales_smooth.csv')
df = sales
df['unique_id'] = df['ixSKU']
df['ds'] = pd.to_datetime(df['iYearMonth'])
df['y'] = df['Shipped']
df['Year'] = pd.to_datetime(df['iYearMonth']).dt.year
train = df[df['Year'] < 2023]
val = df[df['Year'] >= 2023]


part_list =list(sales['sSEMAPart'].unique())
part = part_list[0]


train = train[train['sSEMAPart'] == part ]
val = val[val['sSEMAPart'] == part ]


train = train[['unique_id','ds','y']]
val = val[['unique_id','ds','y']]
comb = pd.concat([train,val])
horizon = len(val['ds'].unique())



def gen_neural_forecasts(horizon,df,freq):

    model_NHITS = NHITS (h = horizon,
                         input_size = 2*horizon,
                         dropout_prob_theta=0.0,
                         activation='ReLU',
                         loss=MAE(),
                         valid_loss=None,
                         max_steps =1000,
                         stack_types =['identity', 'identity','identity'],
                         n_blocks =[1, 1, 1],
                         mlp_units =[[512, 512], [512, 512],[512, 512]],
                         n_pool_kernel_size =[2, 2, 1],
                         n_freq_downsample =[4, 2, 1],
                         pooling_mode ='MaxPool1d',
                         interpolation_mode ='linear',


                         )


    model_NBEATS = NBEATS (h = horizon,
                           input_size = 2*horizon,
                           n_harmonics =2,
                           n_polynomials =2,
                           stack_types =['identity', 'trend', 'seasonality'],
                           n_blocks =[1, 1, 1],
                           mlp_units =[[512, 512], [512, 512],[512, 512]],
                           dropout_prob_theta =0.0,
                           activation ='ReLU',
                           loss=MAE(),
                           max_steps =1000,
                           learning_rate =0.001,
                           num_lr_decays =3,
                           early_stop_patience_steps =-1,
                           val_check_steps =100,
                           batch_size =32,
                           valid_batch_size =None,
                           inference_windows_batch_size =-1,


                         )


    model_LSTM =  LSTM (h = horizon,
                        input_size = 2*horizon,
                        inference_input_size = -1,
                        encoder_n_layers =2,
                        encoder_hidden_size =200,
                        loss=MAE(),
                        valid_loss=None,
                        max_steps =1000,
                        learning_rate =0.001,
           )


    models_nf = [model_LSTM, model_NBEATS, model_NHITS]

    nf = NeuralForecast(models_nf, freq = freq)

    nf.fit(df = df)

    pred_df = nf.predict().reset_index()

    sf = StatsForecast(
    models = [AutoARIMA(season_length = 12),
              SeasonalExponentialSmoothingOptimized(season_length = 12),
              SeasonalNaive(season_length = 12)
              ],
    freq = 'MS')

    sf.fit(df = train)
    forecast_sf = sf.predict(h = horizon).reset_index()


    return pred_df,forecast_sf


def forecast(df,horizon,freq):

    model_NHITS = NHITS (h = horizon,
                         input_size = 2*horizon,
                         dropout_prob_theta=0.0,
                         activation='ReLU',
                         loss=MAE(),
                         valid_loss=None,
                         max_steps =1000,
                         stack_types =['identity', 'identity','identity'],
                         n_blocks =[1, 1, 1],
                         mlp_units =[[512, 512], [512, 512],[512, 512]],
                         n_pool_kernel_size =[2, 2, 1],
                         n_freq_downsample =[4, 2, 1],
                         pooling_mode ='MaxPool1d',
                         interpolation_mode ='linear',


                         )


    model_NBEATS = NBEATS (h = horizon,
                           input_size = 2*horizon,
                           n_harmonics =2,
                           n_polynomials =2,
                           stack_types =['identity', 'trend', 'seasonality'],
                           n_blocks =[1, 1, 1],
                           mlp_units =[[512, 512], [512, 512],[512, 512]],
                           dropout_prob_theta =0.0,
                           activation ='ReLU',
                           loss=MAE(),
                           max_steps =1000,
                           learning_rate =0.001,
                           num_lr_decays =3,
                           early_stop_patience_steps =-1,
                           val_check_steps =100,
                           batch_size =32,
                           valid_batch_size =None,
                           inference_windows_batch_size =-1,


                         )


    model_TimesNet = TimesNet(h = horizon,
                               input_size = 2*horizon,
                               exclude_insample_y=False,
                               hidden_size =64,
                               dropout =0.1,
                               conv_hidden_size =64,
                               top_k =5,
                               num_kernels =6,
                               encoder_layers =2,
                               loss=MAE(),
                               valid_loss=None,
                               max_steps =1000,
                               learning_rate =0.0001,
                               num_lr_decays =-1,
                               early_stop_patience_steps =-1,
                               val_check_steps =100,
                               windows_batch_size=64,
                               inference_windows_batch_size= 3*horizon,
                               start_padding_enabled=False,
                               step_size =1,
                               scaler_type ='standard',
                               random_seed =1,
               )


    model_LSTM =  LSTM (h = horizon,
                        input_size = 2*horizon,
                        inference_input_size = -1,
                        encoder_n_layers =2,
                        encoder_hidden_size =200,
                        loss=MAE(),
                        valid_loss=None,
                        max_steps =1000,
                        learning_rate =0.001,
           )


    models_nf = [model_LSTM, model_NBEATS, model_NHITS]

    nf = NeuralForecast(models_nf, freq = freq)

    nf.fit(df = comb)

    pred_df = nf.predict().reset_index()


    sf = StatsForecast(
    models = [AutoARIMA(season_length = 12),
              SeasonalExponentialSmoothingOptimized(season_length = 12),
              SeasonalNaive(season_length = 12)
              ],
    freq = freq)

    sf.fit(df = comb)
    forecast_sf = sf.predict(h = horizon).reset_index()
    forecast_ens = pred_df.merge(forecast_sf, on = ['unique_id','ds'])

    return forecast_ens





pred_nf,pred_sf = gen_neural_forecasts(horizon,train,freq = 'MS')
eval_df = pred_nf.merge(val, on = ['unique_id','ds'])
eval_df = eval_df.merge(pred_sf, on = ['unique_id','ds'])

y_val = eval_df['y'].to_numpy()
X_val = eval_df[['LSTM','NBEATS','NHITS','AutoARIMA','SeasESOpt','SeasonalNaive']].to_numpy()

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_val, y_val)




###############



forecast_ens = forecast(comb,horizon,freq = 'MS')
#y_val = pred_df['y'].to_numpy()
X_fcst = forecast_ens[['LSTM','NBEATS','NHITS','AutoARIMA','SeasESOpt','SeasonalNaive']].to_numpy()
y_ens = regr.predict(X_fcst)
y_unique_id = forecast_ens['unique_id']
y_ds = forecast_ens['ds']

forecast_ens = pd.DataFrame(list(zip(y_unique_id,y_ds,y_ens)), columns = ['unique_id','ds','Ensemble'])


     