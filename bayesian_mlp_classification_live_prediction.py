
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.regularizers import L1L2
import bayesian_mlp_classification as nn
import os 
import keras
from keras.models import model_from_yaml

# np.random.seed(777)

# Load dataset
directory = os.path.dirname(os.path.abspath(__file__))
file = '/data/TEST_60m_EURUSD_2015-2018_midpoint.csv'
path = directory + file
# path = '/Users/jan/Desktop/...'
output_dir = directory +'/output/'#output directory. 
# format: date,open,high,low,close,volume,other, format of date doesn't matter because it's
# removed in the next step anyway but date has to be ascending. 
dataset = pd.read_csv(path, header=0, index_col=0)
#remove index
dataset.reset_index(drop=True, inplace=True)
dataset.columns = range(1,len(dataset.columns)+1)
periods_in_year = 250*24 #for annualization of Sharpe and CAGR


#HYPERPARAMETERS
#Use values selected by validation procedure
n_lags = 1#lookback window length
n_features = len(dataset.columns) # How many features are there?
# at what return is profit taken in period. #the price has to be strictly higher (lower) 
# for a sell (buy) order to be assumed filled
return_threshold = 0.0005 #0.001
threshold = [0.4] #softmax treshold
bayesian_threshold = [1] #obtained from best validation set model 
p_out = 2 #how many periods out do you want to perdict?
plot=True



#load selected model
name = 'LGS1-EPCHS200-BTCH512-NRNS256-LAY3-BL1_0.00,L2_0.00-KL1_0.00,L2_0.00-\
RL1_0.00,L2_0.00-LR0.0010-LRD0.0000-DO0.50-PTT0.0005'

yaml_file = open('%s%s.yaml' %(output_dir,name), 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights('%s%s.h5' %(output_dir,name))
print(">Loaded model from disk")

#load scaler
scaler = pd.read_csv('scalers.csv', header=0, index_col=0)

n_obs = (n_features)*(n_lags)
pct_change_cols = [1,2,3,4]# taking pct_change of ohlc since they are nonstationary
dataset_returns = nn.get_returns(dataset,  pct_change_cols)
values = dataset_returns#.values.astype('float32')
reframed = nn.multivariate_ts_to_supervised_extra_lag(values, dataset.iloc[:,[0,1,2,3]],
	n_lags, p_out,return_threshold)
	
test = reframed.values
# print(test)
test_X, test_y = test[:, :n_obs], test[:, -3:]

#scale test_X
test_X = test_X = (test_X - np.full(test_X.shape, scaler["train_X_mean"])) / \
        np.full(test_X.shape, scaler["train_X_std"])

periodic_return = test[:,-6]
low_return = test[:,-5]
high_return = test[:,-4]


out_of_sample_dataset = nn.out_of_sample_test(test_X,test_y,periodic_return,
	low_return,high_return,loaded_model)
equity_curve_data = nn.equity_curve(out_of_sample_dataset, name,
	periods_in_year, plot, threshold, return_threshold, bayesian_threshold,p_out)
# save equity curve for further analysis
equity_curve_data.to_csv('%s%s_equity_curve.csv' %(output_dir,name),
header = True, index=True, encoding='utf-8')

live_down_prediction = equity_curve_data['down_prediction'].values[-1]
live_flat_prediction = equity_curve_data['flat_prediction'].values[-1]
live_up_prediction = equity_curve_data['up_prediction'].values[-1]
live_signal = equity_curve_data['signal_%.2f_sigma' % threshold[0]].values[-1]

print(">live_down_prediction: %s " %live_down_prediction)
print(">live_flat_prediction: %s" %live_flat_prediction)
print(">live_up_prediction: %s " %live_up_prediction)
print(">live_signal: %s" %live_signal)
