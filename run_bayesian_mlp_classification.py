
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
file = '/data/TRAIN_60m_EURUSD_2003-2014_midpoint.csv'
path = directory + file
# path = '/Users/jan/Desktop/intraday_data/stats_@ES#C_60m.csv'
output_dir = directory +'/output/'#output directory. 
#format: date,open,high,low,close,volume,other, format of date  doesn't matter because
# it's removed in the next step anyway but date has to be ascending. 
dataset = pd.read_csv(path, header=0, index_col=0)
#remove index
dataset.reset_index(drop=True, inplace=True)
dataset.columns = range(1,len(dataset.columns)+1)
periods_in_year = 250*24#*6#*24*6#*52*24*6#330756#for annualization of Sharpe and CAGR


#HYPERPARAMETERS
#Define hyperparameters. If list, grid search is performed over them to find best value
train_pct = 0.5 #as a percentage of the whole dataset length
n_lags = [1]#lookback window length
n_features = len(dataset.columns) # How many features are there?
# List of **differenced** epochs. For specific value, for example 300, set to [300].
# If you want to test 300 and 600 epochs set to [300, 300]. (600=300+300)
n_epochs = [100,100]
n_batch = [512]# Batch size
n_neurons = [256]# List of number of neurons for each layer. 
n_hidden_dense_layers = [3]# >=0
bias_regularizers = [L1L2(l1=0.0, l2=0.0)]
kernel_regularizers = [L1L2(l1=0.00, l2=0.00)]
recurrent_regularizers = [L1L2(l1=0.0, l2=0.0)]# only applicable for recurrent layers 
learning_rates = [0.00100]#, 0.001, 0.005] 
learning_rate_decay = [0.00]#, 0.00, 0.005]#for adam optimization only
dropout = [0.5]
# at what return is profit taken in period. #the price has to be strictly higher (lower) 
# for a sell (buy) order to be assumed filled
return_threshold = [0.001]#,0.001
threshold = [0,0.2,0.4,0.6,0.8,0.9]#softmax treshold
bayesian_threshold = [0,0.1,0.2,0.4,0.8,1,2,4,8,16,32]
p_out = 3#6*3#12*6#how many periods out do you want to perdict?
plot=False



#loop over all hyperparameters you want to test
models = {}

for lags in n_lags:
    name_lags = ('LGS%i-' % (lags))
    for batch in n_batch:
        name_batch = ('BTCH%i-' % (batch))
        for neurons in n_neurons:
            name_neurons = ('NRNS%i-' % (neurons))
            for layers in n_hidden_dense_layers:
                name_layers = ('LAY%i-' % (layers))
                for breg in bias_regularizers:
                    name_breg = ('BL1_%.2f,L2_%.2f-' % (breg.l1, breg.l2))
                    for kreg in kernel_regularizers:
                        name_kreg = ('KL1_%.2f,L2_%.2f-' % (kreg.l1, kreg.l2))
                        for rreg in recurrent_regularizers:
                            name_rreg = ('RL1_%.2f,L2_%.2f-' % (rreg.l1, rreg.l2))
                            for lr in learning_rates:
                                name_lr = ('LR%.4f-' % (lr))
                                for lrd in learning_rate_decay:
                                    name_lrd = ('LRD%.4f-' % (lrd))
                                    for do in dropout:
                                        name_do = ('DO%.2f-' % (do))
                                        for rt in return_threshold:
                                            name_rt = ('PTT%s' % (rt))
                                            if 'name' in locals():
                                                del name
                                            cum_epochs = 0    
                                            for epochs in n_epochs:
                                                cum_epochs +=epochs
                                                name_epochs = ('EPCHS%i-' % (cum_epochs))
                                                if 'name' in locals():
                                                    l_name = name
                                                    
                                                    model = models[l_name]#must be removed
                                                else: model = None

                                                name = (name_lags + name_epochs + name_batch
                                                    + name_neurons+ name_layers 
                                                    + name_breg + name_kreg + name_rreg 
                                                    + name_lr + name_lrd + name_do + name_rt)

                                                test_X, test_y, periodic_return, low_return,\
                                                 high_return, models[name]= nn.train(
                                                    model, dataset, train_pct, lags, epochs,
                                                    batch, neurons, layers, n_features, breg, kreg, rreg,
                                                    lr, lrd, do, p_out,rt)

                                                out_of_sample_dataset = nn.out_of_sample_test(test_X,test_y,
                                                    periodic_return,low_return,high_return,models[name])
                                                equity_curve_data = nn.equity_curve(out_of_sample_dataset,
                                                    name, periods_in_year,plot, threshold, rt,
                                                    bayesian_threshold, p_out)
                                                # save equity curve for further analysis
                                                # equity_curve_data.to_csv('%s%s_equity_curve.csv' \
                                                # %(output_dir,name),
                                                 # header = True, index=True, encoding='utf-8')
                                                model_yaml = models[name].to_yaml()
                                                with open('%s%s.yaml' %(output_dir,name), "w") as yaml_file:
                                                    yaml_file.write(model_yaml)
                                                # serialize weights to HDF5
                                                models[name].save_weights('%s%s.h5' %(output_dir,name))
                                                print(">Saved model to disk")



