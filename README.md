# Introduction
Predicting stocks is a mammoth task, and requires years of experience to correctly gauge their trends in future. The **ARIMA**(**A**uto-**R**egressive-**M**oving-**A**verage) model is the most widely used machine learning algorithm to forecast stock prices. Though its highly accurate, it's predicting capacity is only limited for shorter periods of time hence rendering it unsuitable for real time depolyment. Therefore we propose a recurrent neural network solution coupled with differencing strategy that stays stable and consistent for many days. We use Gated Recurrent Units as the neural network variant due their low computational overhead. The results are considerably more accurate than standalone recurrent neural networks.


# Method: 
## Explanation of ARIMA and GRU:
### 1. ARIMA:

It is made up of three different components, AR(Auto Regression) model, MA(Moving Average) model, and the degree of differencing. 

<p align = "center">
<img src = "/ims/eq1.png" width = 700>
</p>

**Where,**
<p align = "center">
<img src = "/ims/desp1.png" width = 480>
</p>

The backshift operator B is defined to perform shifting of time series data (Y) by one period,  

<p align = "center">
<img src = "/ims/eq2.png" width = 100>
</p>

Multiplication with higher degree of B, gives a backward shift value more than 1 period,

<p align = "center">
<img src = "/ims/eq3.png" width = 100>
</p>

The first-difference of the series has been shown according to the shift operator B. Let us assume y is the first difference of Y. So at time t, 

<p align = "center">
<img src = "/ims/eq4.png" width = 400>
</p>

The original series Y is multiplied with the factor of 1-B to obtain differenced series y. Now, if assume that z is the first difference of y, which makes z as the second difference of Y, which gives, 

<p align = "center">
<img src = "/ims/eq5.png" width = 500>
</p>

By multiplying factor of (1-B)^2,the second difference of Y is obtained.In general the dth difference of Y would be obtained by multiplying by a factor of (1-B)^d.  
The ARIMA modeling procedures are determined through the Box- Jenkins model building methodology:
1. Identifying the degree of differencing to transform the time series data into stationary.  
2. Estimating the model parameters by auto correlation function (ACF) and partial ACF (PACF).  
3. Checking the degree of fitting on R square maximum principle and Bayesian Information Criterion (BIC) minimum principle then achieved predicted data and noise residuals.  

### 2. GRU:

Engaging technique to resolve in machine learning tasks have recently shown by recurrent neural networks (RNNs). RNN is continuation of a conventional neural network, which may handle a variable-length sequence input. In formally, given input layer of sequence,

<p align = "center">
<img src = "/ims/eq6.png" width = 200>
</p>

<p align = "center">
<img src = "/ims/eq7.png" width = 300>
</p>

<p align = "center">
<img src = "/ims/desp2.png" width = 500>
</p>

The output layer is computed as below formula,
<p align = "center">
<img src = "/ims/eq8.png" width = 200>
</p>

To train RNN, Back Propagation Through Time(BPTT) algorithmic rule is employed. However, it becomes troublesome to coach typical RNNs to capture long-run changes as a result of the vanishing gradient drawback. Therefore Gated Recurrent Unit(GRU) is used, it addresses the vanishing problem by replacing hidden node in traditional RNN by GRU node. Every GRU node consists of 2 gates, update gate zt and reset gate rt. Update gate decides up to what quantity the unit updates its activation, or content. It is computed in equation (1). Reset gate permits to forget the previously computed state, is calculated by equation (2). The hidden layer is computed by equation (4) using Ht which is calculated by equation (3). In the GRU-RNN we use model parameters including,

<p align = "center">
<img src = "/ims/eq_last.png" width = 500>
</p>

## Motivation and Proposed Method:

### Motivation:
* The problem with ARIMA is that, in one iteration of training, we can only predict stock price of particular type(Opening, Closing, Adjusted, Volumes) upto next day. So to predict upto *N* numbers of days, the model must be fit with every new data point that is created until (N-1)th day. We simply cannot use (N-3)th day, (N-2)th day, (N-1)th day data to predict (N+1)th day's price.

* But this is not the case with recurrent neural networks, they're known to be stable for longer time. Hence using them to understand future trend of a stock value might prove beneficial.

### Proposed Method:

* Traditionally, time series only requires one variable that changes according to time, but here we introduced an additonal variable to the input pipeline, and that is the quantifible difference between today's and previous day's price. 

* By doing this we were able to improve the performance of the model by 30%-53%; adjusted closing price of Google stocks from three timelines were considered, 2000-2020, 2010-2020, 2015-2020.

* The input data is time framed in 60 days format, the model has three GRU units with 2 of them returning states and relu activation for all three layers. Mean squred error is the loss with adam loss updation policy. 


#### Walkthrough in Google Colab:

```python 
import pandas as pd
import numpy as np
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tensorflow as tf
import yfinance as yf  
import matplotlib.pyplot as plt
import time
```

```python
df = yf.download('GOOGL','2000-01-01','2020-01-01') # Download Google stocks from 2000-2020

# Setup time series plots
fig, ax = plt.subplots(1, figsize=(30, 7), constrained_layout=True) 
ax.plot(df.index, df['Adj Close'])
ax.xaxis.set_major_locator(mdates.YearLocator(3))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.grid(True)
ax.set_ylabel('Open')
```
```python
# Taking adjusted price variable from loaded dataframe
df = df['Adj Close']
df = pd.concat([df, df.diff()], axis=1)
df.columns = ["Adj", "Diff"]
df["Diff"][0] = 0.0
print(df)

# Preprocessing using Min-Max Scalar
values = df.values
scaler_1 =  MinMaxScaler(feature_range=(0, 1)) 
scaled_1 = scaler_1.fit_transform(values.reshape(-1,1))
scaled = pd.DataFrame(scaled_1)
scaled = scaled.T
```

```python
# Creating 60 days timeframes for actual and difference data
X = []
y = []
for i in range(scaled.iloc[:-60, :].shape[0]):
  X.append(scaled.iloc[i:i+60, :].values)
  y.append(np.array(scaled.iloc[i+60, 0]).reshape(1,-1))

X = np.array(X)
y = np.array(y)[:,0,:]

l = int(len(X)*0.80)
train_X, train_y = X[:l,:], y[:l]
test_X, test_y = X[l:,:], y[l:]

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
```
```python
# Three units of GRU, two of them returns cell states
model_gru = Sequential()
model_gru.add(GRU(70, activation="relu", return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
model_gru.add(GRU(units=70, activation="relu", return_sequences=True))
model_gru.add(GRU(units=20, activation="relu"))
model_gru.add(Dense(units=1))
model_gru.compile(loss='mse', optimizer='adam')
model_gru.summary()
```
```python
# fit network
history_gru = model_gru.fit(train_X, train_y, epochs=20, batch_size=64, validation_data=(test_X, test_y), shuffle=False) 
```
```python
# Plot loss
pyplot.plot(history_gru.history['loss'], label='GRU train', color='red')
pyplot.plot(history_gru.history['val_loss'], label='GRU test', color= 'green')
```
```python
# Predict test data and visualise trends
preds= []
start = time.time() 
for i in range(len(test_y)):
  preds.append(model_gru.predict(test_X[np.newaxis,i])[0])
end = time.time()
print("Time taken to predict:",end-start)
preds_1 = np.array(preds)
act = scaler_1.inverse_transform(test_y[:,0].reshape(-1,1))
preds_2 = scaler_1.inverse_transform(np.array(preds_1[:,0]).reshape(-1,1))

# Plot predicted and actual data
pyplot.figure(figsize=(40, 20))
pyplot.plot(act)
pyplot.plot(preds_2, color='red')
pyplot.savefig('trend')
```

# Experimental Results:

1. 
| Category | Values |
| ------------- | ------------- | 
| Year | 2000-2020  |
| Ours Trend| <img src = "/ims/google_2000.png">  |
| GRU without difference feature Trend | <img src = "/ims/20_year_60_lag_normal.png">  |
| Loss Ours | 0.00018  | 
| Loss Standalone | 0.00036 | 
| Improvement | 50% |

2.
| Category | Values |
| ------------- | ------------- | 
| Year | 2010-2020  | 
| Ours Trend | <img src = "/ims/google_2010.png">  | 
| GRU without difference feature Trend | <img src = "/ims/10_year_60_lag_normal.png">  | 
| Loss Ours | 0.00028  | 
| Loss Standalone | 0.00041 |
| Improvement| 31% |

3.
| Category | Values |
| ------------- | ------------- |
| Year | 2015-2020  | 
| Ours Trend | <img src = "/ims/google_2020.png">  | 
| GRU without difference feature Trend | <img src = "/ims/5_year_60_lag_normal.png">  | \
| Loss Ours | 0.0010 | 
| Loss Standlone | 0.0021 |
| Improvement | 52% |

