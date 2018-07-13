# Bayesian deep neural network for trading

This project implements a Bayesian deep neural network for trading. Gal and Ghahramani (2016) show that “a neural network with arbitrary depth and non-linearities, with dropout applied before every weight layer, is mathematically equivalent to an approximation to the probabilistic deep Gaussian process.” Simply training a neural network (NN) with dropout and then, contrary to common practice, using dropout at test time, performing a large number of stochastic forward passes approximates Bayesian inference. Using this method, uncertainty estimates can be obtained simply by computing the sample standard deviation of the predictions. These estimates, in turn, can be used to ignore trading signals that are highly uncertain, thereby increasing the hit ratio and reducing transaction costs. As can be seen in the training_log.txt file, the hit ratio does indeed increase with the uncertainty threshold the signal has to overcome. Why don't we use the softmax output to evaluate the uncertainty of the model? The softmax output can be high even if the model is very uncertain. 

![alt text](https://github.com/jpwoeltjen/BayesianDNN/blob/master/softmax_output.png)

Source: Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning, Gal and Ghahramani (2016). 

The model might learn a mean reversion strategy. If there is a large down move due to some event, the model might output a large softmax value even if such a move has never occurred during training and is actually justified. In this case we don't want to buy into the market crash because we have no idea what will happen next since we never observed such a thing. Another example of the usefulness of the Bayesian uncertainty estimate is that the model may ignore predictions based on outliers due to data errors that creep through our detection process. 



Inputs are ohlc and in fact any time series that is desired. As an example, the user can input price series statistics, fundamentals, intermarket statistics, order book information or trading signals from other models or based on the trader's own discretion, etc.  
The model predicts the next period's return subject to a profit taking threshold. If the high (low) in the period is strictly higher (lower) than the target price of a long (short) trade, the profit is taken. The inequality is strict to err on the side of conservatism with respect to fill assumptions. The profit taking threshold is also used to separate the classes during training. 

![alt text](https://github.com/jpwoeltjen/BayesianDNN/blob/master/open_box_prediction_white.png)
 
Transaction costs are configurable in the equity_curve() function. They are currently set to $2 ($1 commissions + $1 slippage) per $100000 trading volume. 

The model is fit on the training set. A grid search over the user defined hyper parameters is performed including the profit taking threshold, softmax threshold and Bayesian uncertainty threshold. The models are saved to the disk. The best model and hyperparameters are selected based on the validation set performance. This model is used to predict on the test set providing an unbiased estimate of expected future performance. The testing on the test set is necessary because the act of choosing the best performing model biases the expected future performance based on the validation set upwards. The highest care was taken to prevent any information leakage into the test set. 

As a demonstration the model is fit on the EUR.USD currency pair. A 0.0005 profit taking threshold is tested against a 0.001 threshold. Each model if fit for 200 epochs. 1000 stochastic forward passes are performed to obtain uncertainty estimates and mean predictions. 

Selecting the best model based on validation set Sharpe ratio after transaction costs:

```
FOR MODEL: LGS1-EPCHS200-BTCH512-NRNS256-LAY3-BL1_0.00,L2_0.00-KL1_0.00,L2_0.00-RL1_0.00,L2_0.00-LR0.0010-LRD0.0000-DO0.50-PTT0.0005
return > 1  x std: 
Percent correct 0.40_sigma: 67.03962960921568 %
percentage of periods betting up 0.40_sigma : 49.3652649034 %;
percentage of periods betting down: 0.40_sigma  49.5229292957 %; 
percentage of periods staying out of the market: 0.40_sigma  1.11180580096 %
There were 15788 total trades for 0.40_sigma.
The annualised_sharpe for 0.40_sigma. is: 3.61.
The CAGR for 0.40_sigma. is: 24.72 percent.
The annualised_sharpe for 0.40_sigma. after commissions is: 1.98.
The CAGR for 0.40_sigma. is: 12.76 percent. after commissions
average_gain: 0.000447643873660592
average_loss: -0.0007962537265445978
average_trade: 3.7650617350822874e-05
```

On the test set which ranges from 2015 to 2018, the performance is as follows:

```
FOR MODEL: LGS1-EPCHS200-BTCH512-NRNS256-LAY3-BL1_0.00,L2_0.00-KL1_0.00,L2_0.00-RL1_0.00,L2_0.00-LR0.0010-LRD0.0000-DO0.50-PTT0.0005
return > 1  x std: 
Percent correct 0.40_sigma: 64.33427211809266 %
percentage of periods betting up 0.40_sigma : 50.0481695568 %; 
percentage of periods betting down: 0.40_sigma  49.3256262042 %; 
percentage of periods staying out of the market: 0.40_sigma  0.626204238921 %
There were 8838 total trades for 0.40_sigma.
The annualised_sharpe for 0.40_sigma. is: 2.23.
The CAGR for 0.40_sigma. is: 14.17 percent.
The annualised_sharpe for 0.40_sigma. after commissions is: 0.55.
The CAGR for 0.40_sigma. is: 3.21 percent. after commissions
average_gain: 0.0004424762514773753
average_loss: -0.0007348571815196322
average_trade: 2.2571713001966453e-05
```
![alt text](https://github.com/jpwoeltjen/BayesianDNN/blob/master/Equity_curves/equity_curve_0.40_softmax_1.00_Bayesian_z_score.png)

The Sharpe ratio of 2.23 is highly statistically significant. After transaction costs the performance is not outstanding but still positive.


Upon further inspection the model apparently does poorly on the sort side, with an insignificant Sharpe ratio for short trades. On the contrary, the long side does very well :

```
FOR MODEL: LGS1-EPCHS200-BTCH512-NRNS256-LAY3-BL1_0.00,L2_0.00-KL1_0.00,L2_0.00-RL1_0.00,L2_0.00-LR0.0010-LRD0.0000-DO0.50-PTT0.0005
return > 1  x std: 
Percent correct 0.40_sigma: 65.07554614570302 %
percentage of periods betting up 0.40_sigma : 50.1156069364 %;
percentage of periods betting down: 0.40_sigma  0.0 %;
percentage of periods staying out of the market: 0.40_sigma  49.8843930636 %
There were 8765 total trades for 0.40_sigma.
The annualised_sharpe for 0.40_sigma. is: 3.17.
The CAGR for 0.40_sigma. is: 13.91 percent.
The annualised_sharpe for 0.40_sigma. after commissions is: 1.95.
The CAGR for 0.40_sigma. is: 8.29 percent. after commissions
average_gain: 0.0004436128288106771
average_loss: -0.0007015657522902119
average_trade: 4.366546370479117e-05
```
![alt text](https://github.com/jpwoeltjen/BayesianDNN/blob/master/Equity_curves/equity_curve_0.40_softmax_1.00_Bayesian_z_score_long_only.png)

The Sharpe ratio of 3.17 is highly statistically significant and even the performance after transaction costs is more than satisfactory. 


To check whether the Bayesian NN improves over a NN with standard dropout the model performance over various numbers of stochastic forward passes T is evaluated:

No dropout at test time

```
FOR MODEL: LGS1-EPCHS200-BTCH512-NRNS256-LAY3-BL1_0.00,L2_0.00-KL1_0.00,L2_0.00-RL1_0.00,L2_0.00-LR0.0010-LRD0.0000-DO0.50-PTT0.0005
return > 1  x std: 
Percent correct 0.40_sigma: 64.46369426751592 %
percentage of periods betting up 0.40_sigma : 40.7851637765 %; 
percentage of periods betting down: 0.40_sigma  53.9210019268 %; 
percentage of periods staying out of the market: 0.40_sigma  5.29383429672 %
There were 9367 total trades for 0.40_sigma.
The annualised_sharpe for 0.40_sigma. is: 1.83.
The CAGR for 0.40_sigma. is: 11.30 percent.
The annualised_sharpe for 0.40_sigma. after commissions is: 0.17.
The CAGR for 0.40_sigma. is: 0.85 percent. after commissions
average_gain: 0.00044528413854955584
average_loss: -0.0007537539848906185
average_trade: 1.91902851548334e-05
```

Averaging over T = 100

```
FOR MODEL: LGS1-EPCHS200-BTCH512-NRNS256-LAY3-BL1_0.00,L2_0.00-KL1_0.00,L2_0.00-RL1_0.00,L2_0.00-LR0.0010-LRD0.0000-DO0.50-PTT0.0005
return > 1  x std: 
Percent correct 0.40_sigma: 64.19489469086673 %
percentage of periods betting up 0.40_sigma : 50.2071290944 %; 
percentage of periods betting down: 0.40_sigma  49.2244701349 %; 
percentage of periods staying out of the market: 0.40_sigma  0.568400770713 %
There were 8827 total trades for 0.40_sigma.
The annualised_sharpe for 0.40_sigma. is: 2.10.
The CAGR for 0.40_sigma. is: 13.24 percent.
The annualised_sharpe for 0.40_sigma. after commissions is: 0.42.
The CAGR for 0.40_sigma. is: 2.37 percent. after commissions
average_gain: 0.00044229780406157527
average_loss: -0.0007338288879659513
average_trade: 2.1184403412315518e-05
```

Averaging over T = 5000 

```
FOR MODEL: LGS1-EPCHS200-BTCH512-NRNS256-LAY3-BL1_0.00,L2_0.00-KL1_0.00,L2_0.00-RL1_0.00,L2_0.00-LR0.0010-LRD0.0000-DO0.50-PTT0.0005
return > 1  x std: 
Percent correct 0.40_sigma: 64.39081241198465 %
percentage of periods betting up 0.40_sigma : 49.9903660886 %; 
percentage of periods betting down: 0.40_sigma  49.3786127168 %; 
percentage of periods staying out of the market: 0.40_sigma  0.631021194605 %
There were 8818 total trades for 0.40_sigma.
The annualised_sharpe for 0.40_sigma. is: 2.37.
The CAGR for 0.40_sigma. is: 15.04 percent.
The annualised_sharpe for 0.40_sigma. after commissions is: 0.69.
The CAGR for 0.40_sigma. is: 4.02 percent. after commissions
average_gain: 0.0004426380996794876
average_loss: -0.0007334596160295251
average_trade: 2.3839257874323972e-05
```

As can be seen the test performance improves with the number of stochastic forward passes T over which we average the predictive signal. Performing 5000 stochastic passes yields a substantially higher Sharpe ratio than regular prediction (2.37 vs. 1.83).  

To improve the model one should consider adding potentially predictive data, predicting more than one period into the future, and trying many hyperparameter combinations to find out which ones work best. 
